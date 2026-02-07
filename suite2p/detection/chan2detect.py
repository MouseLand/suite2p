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
    """
    Create a smoothed binary mask for a rectangular block of the image.

    Sets a rectangular region to 1 and applies Gaussian smoothing to create
    soft edges for blending in the bleedthrough correction.

    Parameters
    ----------
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    ny : numpy.ndarray
        Y-indices defining the block rows.
    nx : numpy.ndarray
        X-indices defining the block columns.
    sT : float
        Standard deviation for Gaussian smoothing of the mask.

    Returns
    -------
    mask : numpy.ndarray
        Smoothed mask of shape (Ly, Lx), dtype float32.
    """
    mask = np.zeros((Ly, Lx), np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask


def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    """
    Subtract bleedthrough of the green channel into the red channel.

    Uses non-rigid regression with nblks x nblks spatial blocks to estimate
    and remove the green-to-red bleedthrough, producing a corrected red
    channel mean image.

    Parameters
    ----------
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    nblks : int
        Number of spatial blocks along each axis for piecewise regression.
    mimg : numpy.ndarray
        Green channel mean image of shape (Ly, Lx).
    mimg2 : numpy.ndarray
        Red channel mean image of shape (Ly, Lx).

    Returns
    -------
    mimg2 : numpy.ndarray
        Corrected red channel mean image with bleedthrough subtracted,
        clipped to non-negative values, shape (Ly, Lx).
    """
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


def intensity_ratio(mimg2, stats, chan2_threshold=0.65, inner_neuropil_radius=2,
                    min_neuropil_pixels=350):
    """
    Classify cells as red using the intensity ratio of cell to surround.

    Computes the ratio of channel 2 intensity inside each cell mask to the
    intensity in the surrounding neuropil, excluding other cells. Cells with
    a ratio above the threshold are classified as red.

    Parameters
    ----------
    mimg2 : numpy.ndarray
        Red channel mean image of shape (Ly, Lx).
    stats : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix", "xpix",
        and "lam".
    chan2_threshold : float, optional (default 0.65)
        Threshold on the intensity ratio for red cell classification.
    inner_neuropil_radius : int, optional (default 2)
        Radius in pixels of the inner exclusion zone around each cell for
        neuropil mask creation.
    min_neuropil_pixels : int, optional (default 350)
        Minimum number of pixels in each neuropil mask.

    Returns
    -------
    redcell : numpy.ndarray
        Array of shape (n_cells, 2) where column 0 is the binary red cell
        label and column 1 is the red probability (intensity ratio).
    """
    Ly, Lx = mimg2.shape
    cell_pix = masks.create_cell_pix(stats, Ly=Ly, Lx=Lx)
    cell_masks0 = [
        masks.create_cell_mask(stat, Ly=Ly, Lx=Lx, allow_overlap=True) for stat in stats
    ]
    neuropil_ipix = masks.create_neuropil_masks(
        ypixs=[stat["ypix"] for stat in stats],
        xpixs=[stat["xpix"] for stat in stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=inner_neuropil_radius,
        min_neuropil_pixels=min_neuropil_pixels
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


def cellpose_overlap(stats, mimg2, diameter, chan2_threshold=0.25, device=torch.device("cuda"),
                     settings=None):
    """
    Classify cells as red by computing overlap with Cellpose-detected masks.

    Runs Cellpose on the red channel mean image to detect anatomical masks,
    then computes the intersection-over-union (IOU) between each ROI and the
    Cellpose masks. ROIs with IOU above the threshold are classified as red.

    Parameters
    ----------
    stats : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix" and "xpix".
    mimg2 : numpy.ndarray
        Red channel mean image of shape (Ly, Lx).
    diameter : float or list of float
        Expected cell diameter in pixels for Cellpose detection.
    chan2_threshold : float, optional (default 0.25)
        IOU threshold for red cell classification.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for Cellpose and GPU cache cleanup.
    settings : dict, optional
        Detection settings dictionary passed to Cellpose.

    Returns
    -------
    redstats : numpy.ndarray
        Array of shape (n_cells, 2) where column 0 is the binary red cell
        label and column 1 is the maximum IOU with Cellpose masks.
    masks : numpy.ndarray
        Integer label image of Cellpose-detected masks in the red channel,
        shape (Ly, Lx).
    """
    from . import anatomical
    masks = anatomical.roi_detect(mimg2, diameter=diameter, device=device, settings=settings, chan2=True)[0]
    Ly, Lx = masks.shape
    redstats = np.zeros((len(stats), 2),
                        np.float32)  #changed the size of preallocated space
    for i in range(len(stats)):
        smask = np.zeros((Ly, Lx), np.uint16)
        ypix0, xpix0 = stats[i]["ypix"], stats[i]["xpix"]
        smask[ypix0, xpix0] = 1
        ious = utils.mask_ious(masks, smask)[0]
        if ious.size > 0:
            iou = ious.max()
        else:
            iou = 0.0
        redstats[
            i,
        ] = np.array([iou > chan2_threshold, iou])  #this had the wrong dimension
    return redstats, masks


def detect(meanImg, meanImg_chan2, stats, diameter, cellpose_chan2=False, chan2_threshold=0.65,
           device=torch.device("cuda"), settings=None, inner_neuropil_radius=2,
           min_neuropil_pixels=350):
    """
    Identify red cells using the second channel (e.g. tdTomato).

    Attempts Cellpose-based overlap detection first if ``cellpose_chan2`` is
    True. Falls back to intensity-ratio detection if Cellpose fails or is
    disabled.

    Parameters
    ----------
    meanImg : numpy.ndarray
        Green channel mean image of shape (Ly, Lx).
    meanImg_chan2 : numpy.ndarray
        Red channel mean image of shape (Ly, Lx).
    stats : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix", "xpix",
        and "lam".
    diameter : float or list of float
        Expected cell diameter in pixels for Cellpose detection.
    cellpose_chan2 : bool, optional (default False)
        If True, attempt Cellpose-based red cell detection first.
    chan2_threshold : float, optional (default 0.65)
        Threshold for red cell classification. Meaning depends on method:
        IOU threshold for Cellpose, intensity ratio for fallback.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for Cellpose and GPU cache cleanup.
    settings : dict, optional
        Detection settings dictionary passed to Cellpose.
    inner_neuropil_radius : int, optional (default 2)
        Radius in pixels of the inner exclusion zone for neuropil masks,
        used in intensity-ratio fallback.
    min_neuropil_pixels : int, optional (default 350)
        Minimum number of neuropil pixels, used in intensity-ratio fallback.

    Returns
    -------
    masks : numpy.ndarray or None
        Cellpose-detected masks in the red channel if Cellpose was used,
        otherwise None.
    redstats : numpy.ndarray
        Array of shape (n_cells, 2) where column 0 is the binary red cell
        label and column 1 is the red probability.
    """
    mimg = meanImg.copy()
    mimg2 = meanImg_chan2.copy()

    redstats = None
    masks = None
    if cellpose_chan2:
        try:
            logger.info(">>>> CELLPOSE estimating masks in anatomical channel")
            redstats, masks = cellpose_overlap(stats, mimg2, diameter=diameter,
                                               chan2_threshold=chan2_threshold,
                                               device=device, settings=settings)
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
        redstats = intensity_ratio(mimg2, stats, chan2_threshold=chan2_threshold,
                                  inner_neuropil_radius=inner_neuropil_radius,
                                  min_neuropil_pixels=min_neuropil_pixels)

    return masks, redstats
