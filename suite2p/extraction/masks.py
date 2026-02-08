"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from itertools import count
import numpy as np
from scipy.ndimage import percentile_filter

from ..detection.sparsedetect import extendROI
from .. import default_settings


def create_masks(stats, Ly, Lx, lam_percentile=50., 
                 allow_overlap=False, neuropil_extract=True, inner_neuropil_radius=2,
                 min_neuropil_pixels=350, circular_neuropil=False):
    """
    Create cell and neuropil masks for all ROIs.

    Parameters
    ----------
    stats : list of dict
        List of ROI statistics dictionaries, each containing "ypix", "xpix",
        "lam", "radius", and "overlap".
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    lam_percentile : float, optional (default 50.)
        Percentile filter threshold for excluding low-weight cell pixels from
        neuropil masks. Set to 0 to disable.
    allow_overlap : bool, optional (default False)
        If True, include overlapping pixels in cell masks.
    neuropil_extract : bool, optional (default True)
        If True, compute neuropil masks.
    inner_neuropil_radius : int, optional (default 2)
        Number of pixels to extend around each ROI as an exclusion zone for
        the neuropil mask.
    min_neuropil_pixels : int, optional (default 350)
        Minimum number of pixels in the neuropil mask.
    circular_neuropil : bool, optional (default False)
        If True, extend neuropil masks circularly rather than as rectangles (SLOW).

    Returns
    -------
    cell_masks : list of tuple
        Each element is a tuple (pixel_indices, weights) for one ROI.
    neuropil_masks : list of numpy.ndarray or None
        Each element is an array of flattened pixel indices for the neuropil
        mask. None if neuropil_extract is False.
    """

    cell_pix = create_cell_pix(stats, Ly=Ly, Lx=Lx,
                               lam_percentile=lam_percentile)
    cell_masks = [
        create_cell_mask(stat, Ly=Ly, Lx=Lx, allow_overlap=allow_overlap)
        for stat in stats
    ]
    if neuropil_extract:
        neuropil_masks = create_neuropil_masks(
            ypixs=[stat["ypix"] for stat in stats],
            xpixs=[stat["xpix"] for stat in stats], cell_pix=cell_pix,
            inner_neuropil_radius=inner_neuropil_radius,
            min_neuropil_pixels=min_neuropil_pixels,
            circular=circular_neuropil)
    else:
        neuropil_masks = None
    return cell_masks, neuropil_masks


def create_cell_pix(stats, Ly, Lx, lam_percentile = 50.0):
    """
    Create a binary image indicating which pixels contain a cell.

    Pixels with low cell weights can be excluded using lam_percentile;
    set lam_percentile to 0 to disable filtering.

    Parameters
    ----------
    stats : list of dict
        List of ROI statistics dictionaries, each containing "ypix", "xpix",
        "lam", and "radius".
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    lam_percentile : float, optional (default 50.0)
        Percentile filter threshold for excluding low-weight pixels.

    Returns
    -------
    cell_pix : numpy.ndarray
        Boolean array of shape (Ly, Lx), True where a cell pixel exists.
    """
    cell_pix = np.zeros((Ly, Lx))
    lammap = np.zeros((Ly, Lx))
    radii = np.zeros(len(stats))
    for ni, stat in enumerate(stats):
        radii[ni] = stat["radius"]
        ypix = stat["ypix"]
        xpix = stat["xpix"]
        lam = stat["lam"]
        lammap[ypix, xpix] = np.maximum(lammap[ypix, xpix], lam)
    radius = np.median(radii)
    if lam_percentile > 0.0:
        filt = percentile_filter(lammap, percentile=lam_percentile,
                                 size=int(radius * 5))
        cell_pix = ~np.logical_or(lammap < filt, lammap == 0)
    else:
        cell_pix = lammap > 0.0

    return cell_pix


def create_cell_mask(stat, Ly, Lx, allow_overlap = False):
    """
    Create the cell mask for a single ROI.

    Parameters
    ----------
    stat : dict
        ROI statistics dictionary containing "ypix", "xpix", "lam", and
        "overlap".
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    allow_overlap : bool, optional (default False)
        If True, include overlapping pixels in the cell mask.

    Returns
    -------
    cell_mask : numpy.ndarray
        Flattened pixel indices for the ROI.
    lam_normed : numpy.ndarray
        Normalized pixel weights summing to 1.
    """
    mask = ... if allow_overlap else ~stat["overlap"]
    cell_mask = np.ravel_multi_index((stat["ypix"], stat["xpix"]), (Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat["lam"][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed


def create_neuropil_masks(ypixs, xpixs, cell_pix, inner_neuropil_radius=2,
                          min_neuropil_pixels=350, circular=False):
    """
    Create surround neuropil masks for ROIs by extending each ROI outward.

    Parameters
    ----------
    ypixs : list of numpy.ndarray
        Y-coordinates of the pixels for each ROI.
    xpixs : list of numpy.ndarray
        X-coordinates of the pixels for each ROI.
    cell_pix : numpy.ndarray
        Binary array of shape (Ly, Lx), True where a cell pixel exists.
        These pixels are excluded from neuropil masks.
    inner_neuropil_radius : int, optional (default 2)
        Number of pixels to extend around each ROI as an exclusion zone.
    min_neuropil_pixels : int, optional (default 350)
        Minimum number of pixels in each neuropil mask.
    circular : bool, optional (default False)
        If True, extend neuropil masks circularly (slower). If False, extend
        as rectangles.

    Returns
    -------
    neuropil_ipix : list of numpy.ndarray
        Each element is an array of flattened pixel indices for the neuropil
        mask of one ROI.
    """
    valid_pixels = lambda cell_pix, ypix, xpix: cell_pix[ypix, xpix] < .5
    extend_by = 5

    Ly, Lx = cell_pix.shape
    assert len(xpixs) == len(ypixs)
    neuropil_ipix = []
    idx = 0
    for ypix, xpix in zip(ypixs, xpixs):
        neuropil_mask = np.zeros((Ly, Lx), bool)
        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix,
                                    xpix))  # count how many pixels are valid

        nreps = count()
        ypix1, xpix1 = ypix.copy(), xpix.copy()
        while next(nreps) < 100 and np.sum(valid_pixels(
                cell_pix, ypix1, xpix1)) - nring <= min_neuropil_pixels:
            if circular:
                ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx,
                                         extend_by)  # keep extending
            else:
                ypix1, xpix1 = np.meshgrid(
                    np.arange(max(0,
                                  ypix1.min() - extend_by),
                              min(Ly,
                                  ypix1.max() + extend_by + 1), 1, int),
                    np.arange(max(0,
                                  xpix1.min() - extend_by),
                              min(Lx,
                                  xpix1.max() + extend_by + 1), 1, int), indexing="ij")

        ix = valid_pixels(cell_pix, ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = True
        neuropil_mask[ypix, xpix] = False
        neuropil_ipix.append(np.ravel_multi_index(np.nonzero(neuropil_mask), (Ly, Lx)))
        idx += 1

    return neuropil_ipix
