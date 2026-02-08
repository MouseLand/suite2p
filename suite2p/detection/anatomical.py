"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
import time
import cv2
import os
import torch
import logging 
logger = logging.getLogger(__name__)

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
    """
    Compute the median center and diameter of a single binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        2D binary mask for a single ROI.

    Returns
    -------
    ymed : int
        Y-coordinate of the pixel closest to the median center.
    xmed : int
        X-coordinate of the pixel closest to the median center.
    diam : float
        Estimated diameter of the mask, computed from the number of pixels.
    """
    y, x = np.nonzero(mask)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    ymed = np.median(y)
    xmed = np.median(x)
    imin = ((x - xmed)**2 + (y - ymed)**2).argmin()
    xmed = x[imin]
    ymed = y[imin]
    diam = len(y)**0.5
    diam /= (np.pi**0.5) / 2
    return ymed, xmed, diam
    

def mask_centers(masks):
    """
    Compute the centers and diameters for all masks in a label image.

    Parameters
    ----------
    masks : numpy.ndarray
        2D integer label image where each unique positive value is an ROI.

    Returns
    -------
    centers : numpy.ndarray
        Median center coordinates of shape (n_masks, 2), as [ymed, xmed].
    diams : numpy.ndarray
        Estimated diameters for each mask, shape (n_masks,).
    """
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


def roi_detect(mproj, diameter=None, settings=None,
               pretrained_model=None, device=torch.device("cuda"), chan2=False):
    """
    Detect ROIs in an image using Cellpose.

    Runs a Cellpose model on the input image to segment cells. Optionally
    rescales the image if the diameter aspect ratio is not 1.

    Parameters
    ----------
    mproj : numpy.ndarray
        2D image to segment, shape (Ly, Lx).
    diameter : float or list of float, optional
        Expected cell diameter in pixels. If scalar, used for both axes.
        If list, [dy, dx]. Defaults to [30., 30.].
    settings : dict, optional
        Detection settings dictionary. Used to get "params", "chan2_params",
        "cellprob_threshold", and "flow_threshold" for Cellpose.
    pretrained_model : str, optional
        Name of the Cellpose pretrained model. Defaults to "cpsam".
    device : torch.device, optional (default torch.device("cuda"))
        Torch device, used for GPU cache cleanup after detection.
    chan2 : bool, optional (default False)
        If True, use "chan2_params" from settings instead of "params".

    Returns
    -------
    masks : numpy.ndarray
        Integer label image of shape (Ly, Lx), where each ROI has a
        unique positive integer label.
    centers : numpy.ndarray
        Median center coordinates of shape (n_masks, 2).
    median_diam : float
        Median diameter across all detected masks.
    mask_diams : numpy.ndarray
        Diameters for each detected mask, shape (n_masks,), dtype int32.
    """
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
    params = {} if params is None else params
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
    """
    Convert a label image and weight map into an array of ROI statistics dictionaries.

    For each mask, extracts the pixel coordinates, pixel weights from the weight
    image, and computes the median center.

    Parameters
    ----------
    masks : numpy.ndarray
        2D integer label image where each unique positive value is an ROI.
    weights : numpy.ndarray
        2D weight image of the same shape as `masks` (e.g. max projection),
        used to assign pixel weights ("lam") to each ROI.

    Returns
    -------
    stats : numpy.ndarray
        Array of dictionaries, one per ROI, each containing "ypix", "xpix",
        "lam", "med", and "footprint".
    """
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


def select_rois(mean_img, max_proj, settings, 
                diameter=[12., 12.],
                device=torch.device("cuda")):
    """
    Find ROIs in static images using Cellpose anatomical detection.

    Prepares an image for segmentation based on the "img" setting (max
    projection / mean image ratio, mean image, or max projection), optionally
    applies spatial high-pass filtering, then runs Cellpose to detect ROIs.

    Parameters
    ----------
    mean_img : numpy.ndarray
        2D mean image of shape (Ly, Lx).
    max_proj : numpy.ndarray
        2D maximum projection image of shape (Ly, Lx).
    settings : dict
        Detection settings dictionary. Must contain "img" (str specifying
        which image to segment) and optionally "highpass_spatial" (float).
    diameter : list of float, optional (default [12., 12.])
        Expected cell diameter [dy, dx] in pixels.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for Cellpose and GPU cache cleanup.

    Returns
    -------
    new_settings : dict
        Dictionary with detection metadata including "diameter", "Vcorr",
        and placeholder keys "Vmax", "ihop", "Vsplit", "Vmap",
        "spatscale_pix".
    stats : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix",
        "xpix", "lam", "med", and "footprint".
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
