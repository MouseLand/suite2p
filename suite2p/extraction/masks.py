from typing import List, Tuple, Dict, Any
from itertools import count
import numpy as np
from scipy.ndimage import percentile_filter

from ..detection.sparsedetect import extendROI


def create_masks(ops: Dict[str, Any], stats: List[Dict[str, Any]]):
    """ create cell and neuropil masks """

    cell_pix = create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'], 
                               lam_percentile=ops.get('lam_percentile', 50.0))
    cell_masks = [create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stats]
    if ops.get('neuropil_extract', True):
        neuropil_masks = create_neuropil_masks(
            ypixs=[stat['ypix'] for stat in stats],
            xpixs=[stat['xpix'] for stat in stats],
            cell_pix=cell_pix,
            inner_neuropil_radius=ops['inner_neuropil_radius'],
            min_neuropil_pixels=ops['min_neuropil_pixels'],
            circular=ops.get('circular_neuropil', False)
        )
    else:
        neuropil_masks = None
    return cell_masks, neuropil_masks

def create_cell_pix(stats: List[Dict[str, Any]], Ly: int, Lx: int, 
                    lam_percentile: float = 50.0) -> np.ndarray:
    """Returns Ly x Lx array of whether pixel contains a cell (1) or not (0).
    
    lam_percentile allows some pixels with low cell weights to be used, 
    disable with lam_percentile=0.0

    """
    cell_pix = np.zeros((Ly, Lx))
    lammap = np.zeros((Ly, Lx))
    radii = np.zeros(len(stats))
    for ni,stat in enumerate(stats):
        radii[ni] = stat['radius']
        ypix = stat['ypix']
        xpix = stat['xpix']
        lam = stat['lam']
        lammap[ypix, xpix] = np.maximum(lammap[ypix, xpix], lam)
    radius = np.median(radii)
    if lam_percentile > 0.0:
        filt = percentile_filter(lammap, percentile=lam_percentile, size=int(radius*5))
        cell_pix = ~np.logical_or(lammap < filt, lammap==0)
    else:
        cell_pix = lammap > 0.0

    return cell_pix


def create_cell_mask(stat: Dict[str, Any], Ly: int, Lx: int, allow_overlap: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    creates cell masks for ROIs in stat and computes radii

    Parameters
    ----------

    stat : dictionary 'ypix', 'xpix', 'lam'
    Ly : y size of frame
    Lx : x size of frame
    allow_overlap : whether or not to include overlapping pixels in cell masks

    Returns
    -------

    cell_masks : len ncells, each has tuple of pixels belonging to each cell and weights
    lam_normed
    """
    mask = ... if allow_overlap else ~stat['overlap']
    cell_mask = np.ravel_multi_index((stat['ypix'], stat['xpix']), (Ly, Lx))
    cell_mask = cell_mask[mask]
    lam = stat['lam'][mask]
    lam_normed = lam / lam.sum() if lam.size > 0 else np.empty(0)
    return cell_mask, lam_normed



def create_neuropil_masks(ypixs, xpixs, cell_pix, inner_neuropil_radius, min_neuropil_pixels, circular=False):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slower if circular)

    Parameters
    ----------

    cellpix : 2D array
        1 if ROI exists in pixel, 0 if not;
        pixels ignored for neuropil computation

    Returns
    -------

    neuropil_masks : list
        each element is array of pixels in mask in (Ly*Lx) coordinates

    """
    valid_pixels = lambda cell_pix, ypix, xpix: cell_pix[ypix, xpix] < .5
    extend_by = 5

    Ly, Lx = cell_pix.shape
    assert len(xpixs) == len(ypixs)
    neuropil_ipix = []
    idx=0
    for ypix, xpix in zip(ypixs, xpixs):
        neuropil_mask = np.zeros((Ly, Lx), bool)
        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix, xpix))  # count how many pixels are valid

        nreps = count()
        ypix1, xpix1 = ypix.copy(), xpix.copy()
        while next(nreps) < 100 and np.sum(valid_pixels(cell_pix, ypix1, xpix1)) - nring <= min_neuropil_pixels:
            if circular:
                ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx, extend_by)  # keep extending
            else:
                ypix1, xpix1 = np.meshgrid(np.arange(max(0, ypix1.min() - extend_by), min(Ly, ypix1.max() + extend_by + 1), 1, int), 
                                           np.arange(max(0, xpix1.min() - extend_by), min(Lx, xpix1.max() + extend_by + 1), 1, int),
                                           indexing='ij')
            
        ix = valid_pixels(cell_pix, ypix1, xpix1)
        neuropil_mask[ypix1[ix], xpix1[ix]] = True
        neuropil_mask[ypix, xpix] = False
        neuropil_ipix.append(np.ravel_multi_index(np.nonzero(neuropil_mask), (Ly, Lx)))
        idx+=1

    return neuropil_ipix