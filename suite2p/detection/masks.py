from typing import List, Tuple
from itertools import count
import numpy as np

from suite2p.detection.sparsedetect import extendROI


def create_cell_pix(stats, Ly, Lx, allow_overlap=False) -> np.ndarray:
    """Returns Ly x Lx array of whether it contains a cell (1) or not (0)."""
    cell_pix = np.zeros((Ly, Lx))
    for stat in stats:
        mask = ... if allow_overlap else ~stat['overlap']
        ypix = stat['ypix'][mask]
        xpix = stat['xpix'][mask]
        lam = stat['lam'][mask]
        if xpix.size:
            cell_pix[ypix[lam > 0], xpix[lam > 0]] = 1

    return cell_pix


def create_cell_masks(stats, Ly, Lx, allow_overlap=False) -> List[Tuple[np.ndarray, np.ndarray]]:
    """ creates cell masks for ROIs in stat and computes radii

    Parameters
    ----------

    stats : dictionary
        'ypix', 'xpix', 'lam'

    Ly : float
        y size of frame

    Lx : float
        x size of frame

    allow_overlap : bool (optional, default False)
        whether or not to include overlapping pixels in cell masks

    Returns
    -------
    cell_masks : list 
        len ncells, each has tuple of pixels belonging to each cell and weights

    """
    cell_masks = []
    for stat in stats:
        mask = ... if allow_overlap else ~stat['overlap']
        ypix = stat['ypix'][mask]
        xpix = stat['xpix'][mask]
        cell_mask = np.ravel_multi_index((ypix, xpix), (Ly, Lx))
        lam = stat['lam'][mask]
        lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
        cell_masks.append((cell_mask, lam_normed))

    return cell_masks


def create_neuropil_masks(ypixs, xpixs, cell_pix, inner_neuropil_radius, min_neuropil_pixels):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slow!)

    Parameters
    ----------

    cellpix : 2D array
        1 if ROI exists in pixel, 0 if not;
        pixels ignored for neuropil computation

    Returns
    -------

    neuropil_masks : 3D array
        size [ncells x Ly x Lx] where each pixel is weight of neuropil mask

    """
    valid_pixels = lambda cell_pix, ypix, xpix: cell_pix[ypix, xpix] < .5

    Ly, Lx = cell_pix.shape
    assert len(xpixs) == len(ypixs)
    neuropil_masks = np.zeros((len(xpixs), Ly, Lx), np.float32)
    for ypix, xpix, neuropil_mask in zip(ypixs, xpixs, neuropil_masks):

        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix, xpix))  # count how many pixels are valid

        nreps = count()
        ypix1, xpix1 = ypix, xpix
        while next(nreps) < 100 and np.sum(valid_pixels(cell_pix, ypix1, xpix1)) - nring <= min_neuropil_pixels:
            ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx, 5)  # keep extending

        ix = valid_pixels(cell_pix, ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = 1.
        neuropil_mask[ypix, xpix] = 0

    neuropil_masks /= np.sum(neuropil_masks, axis=(1, 2), keepdims=True)
    neuropil_masks = np.reshape(neuropil_masks, (-1, Ly * Lx))
    return neuropil_masks
