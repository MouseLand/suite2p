"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from typing import Any, Dict
from scipy.ndimage import find_objects, gaussian_filter
import time
import cv2
import os
import torch
import logging 
logger = logging.getLogger(__name__)


from . import utils
from .stats import roi_stats

try:
    from cellpose.models import CellposeModel
    from cellpose import transforms, dynamics, core
    from cellpose.utils import fill_holes_and_remove_small_masks
    from cellpose.transforms import normalize99
    CELLPOSE_INSTALLED = True
except Exception as e:
    CELLPOSE_INSTALLED = False
    cellpose_error = e


def mask_stats(mask):
    """ median and diameter of mask """
    y, x = np.nonzero(mask)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    ymed = np.median(y)
    xmed = np.median(x)
    imin = np.argmin((x - xmed)**2 + (y - ymed)**2)
    xmed = x[imin]
    ymed = y[imin]
    diam = len(y)**0.5
    diam /= (np.pi**0.5) / 2
    return ymed, xmed, diam
    

def mask_centers(masks):
    centers = np.zeros((masks.max(), 2), np.int32)
    diams = np.zeros(masks.max(), np.float32)
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            ymed, xmed, diam = mask_stats(masks[sr, sc] == (i + 1))
            centers[i] = np.array([ymed, xmed])
            diams[i] = diam
    return centers, diams


def patch_detect(patches, diam):
    """ anatomical detection of masks from top active frames for putative cell """
    logger.info("refining masks using cellpose")
    npatches = len(patches)
    ly = patches[0].shape[0]
    model = CellposeModel(gpu=True if core.use_gpu() else False)
    imgs = np.zeros((npatches, ly, ly, 2), np.float32)
    for i, m in enumerate(patches):
        imgs[i, :, :, 0] = transforms.normalize99(m)
    rsz = 30. / diam
    imgs = transforms.resize_image(imgs, rsz=rsz).transpose(0, 3, 1, 2)
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)

    pmasks = np.zeros((npatches, ly, ly), np.uint16)
    batch_size = 8 * 224 // ly
    tic = time.time()
    for j in np.arange(0, npatches, batch_size):
        y = model.net(imgs[j:j + batch_size])[0]
        y = y[:, :, ysub[0]:ysub[-1] + 1, xsub[0]:xsub[-1] + 1]
        y = y.asnumpy()
        for i, yi in enumerate(y):
            cellprob = yi[-1]
            dP = yi[:2]
            niter = 1 / rsz * 200
            p = dynamics.follow_flows(-1 * dP * (cellprob > 0) / 5., niter=niter)
            maski = dynamics.get_masks(p, iscell=(cellprob > 0), flows=dP,
                                       threshold=1.0)
            maski = fill_holes_and_remove_small_masks(maski)
            maski = transforms.resize_image(maski, ly, ly,
                                            interpolation=cv2.INTER_NEAREST)
            pmasks[j + i] = maski
        if j % 5 == 0:
            logger.info("%d / %d masks created in %0.2fs" %
                  (j + batch_size, npatches, time.time() - tic))
    return pmasks


def refine_masks(stats, patches, seeds, diam, Lyc, Lxc):
    nmasks = len(patches)
    patch_masks = patch_detect(patches, diam)
    ly = patches[0].shape[0] // 2
    igood = np.zeros(nmasks, "bool")
    for i, (patch_mask, stat, (yi, xi)) in enumerate(zip(patch_masks, stats, seeds)):
        mask = np.zeros((Lyc, Lxc), np.float32)
        ypix0, xpix0 = stat["ypix"], stat["xpix"]
        mask[ypix0, xpix0] = stat["lam"]
        func_mask = utils.square_mask(mask, ly, yi, xi)
        ious = utils.mask_ious(patch_mask.astype(np.uint16), (func_mask
                                                              > 0).astype(np.uint16))[0]
        if len(ious) > 0 and ious.max() > 0.45:
            mask_id = np.argmax(ious) + 1
            patch_mask = patch_mask[max(0, ly - yi):min(2 * ly, Lyc + ly - yi),
                                    max(0, ly - xi):min(2 * ly, Lxc + ly - xi)]
            func_mask = func_mask[max(0, ly - yi):min(2 * ly, Lyc + ly - yi),
                                  max(0, ly - xi):min(2 * ly, Lxc + ly - xi)]
            ypix0, xpix0 = np.nonzero(patch_mask == mask_id)
            lam0 = func_mask[ypix0, xpix0]
            lam0[lam0 <= 0] = lam0.min()
            ypix0 = ypix0 + max(0, yi - ly)
            xpix0 = xpix0 + max(0, xi - ly)
            igood[i] = True
            stat["ypix"] = ypix0
            stat["xpix"] = xpix0
            stat["lam"] = lam0
            stat["anatomical"] = True
        else:
            stat["anatomical"] = False
    return stats


def roi_detect(mproj, diameter=None, settings=None,
               pretrained_model=None, device=torch.device("cuda"), chan2=False):
    """ detect ROIs in image using cellpose """
    Lyc, Lxc = mproj.shape
    diameter = [30., 30. ] if diameter is None else diameter
    diameter = [diameter, diameter] if np.isscalar(diameter) else diameter
    
    rescale = diameter[1] / diameter[0]
    if rescale != 1.0:
        img = cv2.resize(img, (Lxc, int(Lyc * rescale)))
    logger.info("!NOTE! diameter set to %0.2f for cell detection with cellpose" %
                diameter[1])

    pretrained_model = "cpsam" if pretrained_model is None else pretrained_model
    model = CellposeModel(pretrained_model=pretrained_model, gpu=True if core.use_gpu() else False)
    params = settings["params"] if not chan2 else settings["chan2_params"]
    masks = model.eval(mproj, diameter=diameter[1],
                       cellprob_threshold=settings.get("cellprob_threshold", 0.0),
                       flow_threshold=settings.get("flow_threshold", 0.4),
                       **params)[0]
    shape = masks.shape
    _, masks = np.unique(np.int32(masks), return_inverse=True)
    masks = masks.reshape(shape)

    if rescale != 1.0:
        masks = cv2.resize(masks, (Lxc, Lyc), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (Lxc, Lyc))
    
    centers, mask_diams = mask_centers(masks)
    median_diam = np.median(mask_diams)
    logger.info(">>>> %d masks detected, median diameter = %0.2f " %
          (masks.max(), median_diam))
    
    if device.type == "cuda":
        del model
        torch.cuda.empty_cache()

    return masks, centers, median_diam, mask_diams.astype(np.int32)


def masks_to_stats(masks, weights):
    stats = []
    slices = find_objects(masks)
    for i, si in enumerate(slices):
        sr, sc = si
        ypix0, xpix0 = np.nonzero(masks[sr, sc] == (i + 1))
        ypix0 = ypix0.astype(int) + sr.start
        xpix0 = xpix0.astype(int) + sc.start
        ymed = np.median(ypix0)
        xmed = np.median(xpix0)
        imin = np.argmin((xpix0 - xmed)**2 + (ypix0 - ymed)**2)
        xmed = xpix0[imin]
        ymed = ypix0[imin]
        stats.append({
            "ypix": ypix0,
            "xpix": xpix0,
            "lam": weights[ypix0, xpix0],
            "med": [ymed, xmed],
            "footprint": 1
        })
    stats = np.array(stats)
    return stats


def select_rois(mean_img, max_proj, settings: Dict[str, Any], 
                diameter=[12., 12.],
                device=torch.device("cuda")):
    """ find ROIs in static frames
    
    Parameters:

        settings: dictionary
            requires keys "high_pass", "anatomical_only"
        
        mov: ndarray t x Lyc x Lxc, binned movie
    
    Returns:
        stats: list of dicts
    
    """
    Lyc, Lxc = mean_img.shape
    if settings["img"] == 'max_proj / meanImg':
        img = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
        weights = max_proj
    elif settings["img"] == 'meanImg':
        img = mean_img
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1)) /
            (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)), 0, 1)
    else:
        img = max_proj.copy()
        weights = max_proj

    t0 = time.time()

    if settings.get("highpass_spatial", 0):
        img = np.clip(normalize99(img), 0, 1)
        img -= gaussian_filter(img, diameter[1] * settings["highpass_spatial"])

    masks, centers, median_diam, mask_diams = roi_detect(
        img, diameter=diameter, settings=settings, device=device)
    
    stats = masks_to_stats(masks, weights)
    logger.info("Detected %d ROIs, %0.2f sec" % (len(stats), time.time() - t0))

    new_settings = {
        "diameter": median_diam,
        "Vmax": 0,
        "ihop": 0,
        "Vsplit": 0,
        "Vcorr": img,
        "Vmap": 0,
        "spatscale_pix": 0
    }

    return new_settings, stats


# def run_assist():
#     nmasks, diam = 0, None
#     if anatomical:
#         try:
#             logger.info(">>>> CELLPOSE estimating spatial scale and masks as seeds for functional algorithm")
#             from . import anatomical
#             mproj = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
#             masks, centers, diam, mask_diams = anatomical.roi_detect(mproj)
#             nmasks = masks.max()
#         except:
#             logger.info("ERROR importing or running cellpose, continuing without anatomical estimates")
#         if tj < nmasks:
#             yi, xi = centers[tj]
#             ls = mask_diams[tj]
#             imap = np.ravel_multi_index((yi, xi), (Lyc, Lxc))
# if nmasks > 0:
#         stats = anatomical.refine_masks(stats, patches, seeds, diam, Lyc, Lxc)
#         for stat in stats:
#             if stat["anatomical"]:
#                 stat["lam"] *= sdmov[stat["ypix"], stat["xpix"]]
