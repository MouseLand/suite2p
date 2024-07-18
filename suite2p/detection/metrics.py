"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import numpy as np
import cv2
import logging 
logger = logging.getLogger(__name__)

from .utils import mask_ious, temporal_high_pass_filter, standard_deviation_over_time, _intersection_over_union
from .sparsedetect import neuropil_subtraction
from .denoise import pca_denoise
from ..io import BinaryFile


def compute_gt_matches(img, masks, stat_func, settings=None, reg_file=None, threshold=0.5):
    """ anatomical img and masks matched to functional ROIs in stat_func """
    Ly, Lx = masks.shape
    stat_anat, iorig = extend_anatomical(img, masks, settings=settings, reg_file=reg_file)
    iou, iout, preds, ap = match_func_anat(stat_func, stat_anat, Ly, Lx, threshold)

    chosen_cells = iout > threshold
    func_ids = preds[chosen_cells] - 1
    overlaps = iout[chosen_cells]

    return stat_anat, iorig, iou, func_ids, overlaps


def match_func_anat(stat_func, stat_anat, Ly, Lx, threshold=0.5):
    """ match functional ROIs to anatomical ROIs by correlation"""
    iou = np.zeros((len(stat_anat), len(stat_func)))
    ly = 15
    for i, sf in enumerate(stat_func):
        if sf["ypix"].size < 20:
            continue
        ypix, xpix, lam = sf["ypix"].copy(), sf["xpix"].copy(), sf["lam"].copy()
        lam /= (lam**2).sum()**0.5
        # box around ROI
        ymed, xmed = sf["med"][0], sf["med"][1]
        inds = (slice(max(0, ymed - ly),
                      min(ymed + ly, Ly)), slice(max(0, xmed - ly), min(xmed + ly, Lx)))
        mf = np.zeros((Ly, Lx), np.float32)
        mf[ypix, xpix] = lam
        mfc = mf[inds].flatten()
        mfc /= (mfc**2).sum()**0.5

        # matched anatomical masks (will not compute IOU for all masks)
        for j, sa in enumerate(stat_anat):
            ypix_a, xpix_a = sa["ypix"], sa["xpix"]
            if (np.logical_and(ypix_a > inds[0].start, ypix_a < inds[0].stop).sum() > 0
                    and np.logical_and(xpix_a > inds[1].start, xpix_a
                                       < inds[1].stop).sum() > 0):
                lam_a = sa["lam"].copy()
                lam_a /= (lam_a**2).sum()**0.5
                ma = np.zeros((Ly, Lx), np.float32)
                ma[ypix_a, xpix_a] = lam_a
                mac = ma[inds].flatten()
                mac /= ((mac**2).sum()**0.5 + 1e-10)
                iou[j, i] = (mac * mfc).sum()
        if i % 1000 == 0:
            logger.info("%d ROIs processed" % i)
    logger.info("%d ROIs processed" % i)

    n_true = len(stat_anat)
    n_pred = len(stat_func)
    iout, preds = mask_ious(iou)
    tp = (iout > threshold).sum()
    logger.info((iout > threshold).sum())
    fn = n_true - tp
    fp = n_pred - tp
    ap = tp / (fn + tp + fp)
    logger.info("TP: %d, FN: %d, FP: %d, AP: %0.3f" % (tp, fn, fp, ap))

    return iou, iout, preds, ap


def extend_anatomical(img_anat, masks_anat, mov=None, settings=None, reg_file=None):
    if mov is None:
        if reg_file is None:
            reg_file = settings["reg_file"]

        bin_size = int(
            max(1, settings["nframes"] // settings["nbinned"], np.round(settings["tau"] * settings["fs"])))
        t0 = time.time()
        with BinaryFile(filename=reg_file, Ly=settings["Ly"], Lx=settings["Lx"]) as f:
            mov = f.bin_movie(
                bin_size=bin_size,
                bad_frames=settings.get("badframes"),
                y_range=settings["yrange"],
                x_range=settings["xrange"],
            )
        logger.info("Binned movie [%d,%d,%d] in %0.2f sec." %
              (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    nt, Lyc, Lxc = mov.shape

    if settings is not None:
        # process movie
        mov = pca_denoise(mov, [settings["block_size"][0] // 2, settings["block_size"][1] // 2],
                          0.5)
        mov = temporal_high_pass_filter(mov=mov, width=int(settings["high_pass"]))
        sdmov = standard_deviation_over_time(mov, batch_size=settings["batch_size"])
        mov = neuropil_subtraction(
            mov=mov / sdmov,
            filter_size=settings["spatial_hp_detect"])  # subtract low-pass filtered movie
    else:
        settings = {"yrange": [0, Lyc], "xrange": [0, Lxc]}
        sdmov = np.ones(mov.shape[1:])

    redimg = img_anat[settings["yrange"][0]:settings["yrange"][-1],
                      settings["xrange"][0]:settings["xrange"][-1]]
    redmasks = masks_anat[settings["yrange"][0]:settings["yrange"][-1],
                          settings["xrange"][0]:settings["xrange"][-1]]
    ly = 10
    stat_anat = []
    iorig = []
    for i in range(masks_anat.max()):
        ypix, xpix = np.nonzero(redmasks == (i + 1))
        if ypix.size < 10:
            continue

        # create box around ROI to grow ROI
        ymed, xmed = int(np.median(ypix)), int(np.median(xpix))
        inds = (slice(max(0, ymed - ly),
                      min(ymed + ly, Lyc)), slice(max(0, xmed - ly),
                                                  min(xmed + ly, Lxc)))
        maskb = np.zeros((Lyc, Lxc), "bool")
        maskb[ypix, xpix] = 1
        maskb = maskb[inds].astype(np.float32)
        maskb /= (maskb.sum())**0.5
        bx = mov[:, inds[0], inds[1]]

        ### get activity mask
        # find active frames
        lam = redimg[ypix, xpix]
        F = mov[:, ypix, xpix] @ lam  #.sum(axis=1)
        active_frames = F > np.percentile(F, 99)
        # activity of pixels in box on active_frames
        cc = bx[active_frames].sum(axis=0)
        cc_threshold = max(0, cc.max() / 5.0)
        cc_mask = cc > cc_threshold

        # get connected components
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            (cc_mask).astype(np.uint8), connectivity=4)
        npix = stats[1:, -1]
        if (npix > 15).sum() == 0:
            continue

        # get overlap of connected components with original mask, take one with largest overlap
        iou = _intersection_over_union((maskb > 0).astype(np.uint16),
                                       output.astype(np.uint16))[1, 1:]
        max_label = np.nonzero(npix > 15)[0][iou[npix > 15].argmax()]
        cc_mask = (output == (max_label + 1))
        cc[~cc_mask] = 0

        # correlation of activity mask with original mask
        mfunc = cc.flatten() / ((cc**2).sum()**0.5)
        corr = (mfunc * maskb.flatten()).sum()
        if corr < 0.65:
            continue

        # mask pix and weights
        ypix, xpix = np.nonzero(cc_mask)
        ypix += max(0, ymed - ly)
        xpix += max(0, xmed - ly)
        lam = cc[cc_mask] * sdmov[ypix, xpix]
        # ypix, xpix in full coordinates
        ypix += settings["yrange"][0]
        xpix += settings["xrange"][0]
        stat_anat.append({"ypix": ypix, "xpix": xpix, "lam": lam})
        iorig.append(i)

    return stat_anat, iorig
