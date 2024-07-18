"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

import logging 
logger = logging.getLogger(__name__)

from ..extraction import masks
from . import utils
"""
identify cells with channel 2 brightness (aka red cells)

main function is detect
takes from settings: "meanImg", "meanImg_chan2", "Ly", "Lx"
takes from stat: "ypix", "xpix", "lam"
"""


def quadrant_mask(Ly, Lx, ny, nx, sT):
    mask = np.zeros((Ly, Lx), np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask


def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), np.float32)
    weights = np.zeros((nblks, nblks), np.float32)
    yb = np.linspace(0, Ly, nblks + 1).astype(int)
    xb = np.linspace(0, Lx, nblks + 1).astype(int)
    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy + 1]).astype(int)
            nx = np.arange(xb[ix], xb[ix + 1]).astype(int)
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)
            x = mimg[np.ix_(ny, nx)].flatten()
            x2 = mimg2[np.ix_(ny, nx)].flatten()
            # predict chan2 from chan1
            a = (x * x2).sum() / (x * x).sum()
            weights[iy, ix] = a
    mask /= mask.sum(axis=-1).sum(axis=-1)[:, :, np.newaxis, np.newaxis]
    mask *= weights
    mask *= mimg[:, :, np.newaxis, np.newaxis]
    mimg2 -= mask.sum(axis=-1).sum(axis=-1)
    mimg2 = np.maximum(0, mimg2)
    return mimg2


def intensity_ratio(mimg2, stats, chan2_threshold=0.65):
    """ compute pixels in cell and in area around cell (including overlaps)
        (exclude pixels from other cells) """
    Ly, Lx = mimg2.shape
    cell_pix = masks.create_cell_pix(stats, Ly=Ly, Lx=Lx)
    cell_masks0 = [
        masks.create_cell_mask(stat, Ly=Ly, Lx=Lx, allow_overlap=True) for stat in stats
    ]
    neuropil_ipix = masks.create_neuropil_masks(
        ypixs=[stat["ypix"] for stat in stats],
        xpixs=[stat["xpix"] for stat in stats],
        cell_pix=cell_pix,
    )
    cell_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    neuropil_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    for cell_mask, cell_mask0, neuropil_mask, neuropil_mask0 in zip(
            cell_masks, cell_masks0, neuropil_masks, neuropil_ipix):
        cell_mask[cell_mask0[0]] = cell_mask0[1]
        neuropil_mask[neuropil_mask0.astype(np.int64)] = 1. / len(neuropil_mask0)

    inpix = cell_masks @ mimg2.flatten()
    extpix = neuropil_masks @ mimg2.flatten()
    inpix = np.maximum(1e-3, inpix)
    redprob = inpix / (inpix + extpix)
    redcell = redprob > chan2_threshold
    
    return np.stack((redcell, redprob), axis=-1)


def cellpose_overlap(stats, mimg2, chan2_threshold=0.25, device=torch.device("cuda")):
    from . import anatomical
    masks = anatomical.roi_detect(mimg2, device=device)[0]
    Ly, Lx = masks.shape
    redstats = np.zeros((len(stats), 2),
                        np.float32)  #changed the size of preallocated space
    for i in range(len(stats)):
        smask = np.zeros((Ly, Lx), np.uint16)
        ypix0, xpix0 = stats[i]["ypix"], stats[i]["xpix"]
        smask[ypix0, xpix0] = 1
        ious = utils.mask_ious(masks, smask)[0]
        iou = ious.max()
        redstats[
            i,
        ] = np.array([iou > chan2_threshold, iou])  #this had the wrong dimension
    return redstats, masks


def detect(meanImg, meanImg_chan2, stats, cellpose_chan2=True, chan2_threshold=0.65,
           device=torch.device("cuda")):
    mimg = meanImg.copy()
    mimg2 = meanImg_chan2.copy()

    redstats = None
    if cellpose_chan2:
        try:
            logger.info(">>>> CELLPOSE estimating masks in anatomical channel")
            redstats, masks = cellpose_overlap(stats, mimg2, 
                                               chan2_threshold=chan2_threshold,
                                               device=device)
        except:
            logger.info(
                "ERROR importing or running cellpose, continuing with intensity-based anatomical estimates"
            )

    if redstats is None:
        # subtract bleedthrough of green into red channel
        # non-rigid regression with nblks x nblks pieces
        nblks = 3
        #Ly, Lx = settings["Ly"], settings["Lx"]
        #mimg2_corr = correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2)
        redstats = intensity_ratio(mimg2, stats, chan2_threshold=chan2_threshold)
    
    return masks, redstats
