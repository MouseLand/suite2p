from typing import List
import numpy as np

from suite2p.detection.sparsedetect import extendROI

def count_overlaps(Ly: int, Lx: int, ypixs, xpixs) -> np.ndarray:
    overlap = np.zeros((Ly, Lx))
    for xpix, ypix in zip(xpixs, ypixs):
        overlap[ypix, xpix] += 1
    return overlap


def get_overlaps(overlaps, ypixs: List[np.ndarray], xpixs: List[np.ndarray]) -> List[np.ndarray]:
    """computes overlapping pixels from ROIs"""
    return [overlaps[ypix, xpix] > 1 for ypix, xpix in zip(ypixs, xpixs)]


def remove_overlappers(ypixs, xpixs, max_overlap: float, Ly: int, Lx: int) -> List[int]:
    """returns ROI indices are remain after removing those that overlap more than fraction max_overlap with other ROIs"""
    overlaps = count_overlaps(Ly=Ly, Lx=Lx, ypixs=ypixs, xpixs=xpixs)
    ix = []
    for i, (ypix, xpix) in reversed(list(enumerate(zip(ypixs, xpixs)))):  # todo: is there an ordering effect here that affects which rois will be removed and which will stay?
        if np.mean(overlaps[ypix, xpix] > 1) > max_overlap:  # note: fancy indexing returns a copy
            overlaps[ypix, xpix] -= 1
        else:
            ix.append(i)
    return ix[::-1]


def create_cell_masks(stat, Ly, Lx, allow_overlap=False):
    """ creates cell masks for ROIs in stat and computes radii

    Parameters
    ----------

    stat : dictionary
        'ypix', 'xpix', 'lam'

    Ly : float
        y size of frame

    Lx : float
        x size of frame

    allow_overlap : bool (optional, default False)
        whether or not to include overlapping pixels in cell masks

    Returns
    -------
    
    cell_pix : 2D array
        size [Ly x Lx] where 1 if pixel belongs to cell
    
    cell_masks : list 
        len ncells, each has tuple of pixels belonging to each cell and weights

    """

    ncells = len(stat)
    cell_pix = np.zeros((Ly,Lx))
    cell_masks = []

    for n in range(ncells):
        if allow_overlap:
            overlap = np.zeros((stat[n]['npix'],), bool)
        else:
            overlap = stat[n]['overlap']
        ypix = stat[n]['ypix'][~overlap]
        xpix = stat[n]['xpix'][~overlap]
        lam  = stat[n]['lam'][~overlap]
        if xpix.size:
            # add pixels of cell to cell_pix (pixels to exclude in neuropil computation)
            cell_pix[ypix[lam>0],xpix[lam>0]] += 1
            ipix = np.ravel_multi_index((ypix, xpix), (Ly,Lx)).astype('int')
            #utils.sub2ind((Ly,Lx), ypix, xpix)
            cell_masks.append((ipix, lam/lam.sum()))
        else:
            cell_masks.append((np.zeros(0).astype('int'), np.zeros(0)))

    cell_pix = np.minimum(1, cell_pix)
    return cell_pix, cell_masks


def create_neuropil_masks(stats, cell_pix, inner_neuropil_radius, min_neuropil_pixels):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slow!)
    
    Parameters
    ----------

    ops : dictionary
        'inner_neuropil_radius', 'min_neuropil_pixels'

    stat : dictionary
        'ypix', 'xpix', 'lam'

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
    neuropil_masks = np.zeros((len(stats), Ly, Lx), np.float32)
    for stat, neuropil_mask in zip(stats, neuropil_masks):

        # extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(stat['ypix'], stat['xpix'], Ly, Lx, niter=inner_neuropil_radius)
        nring = np.sum(valid_pixels(cell_pix, ypix, xpix))  # count how many pixels are valid
        inner_ring = neuropil_masks[ypix, xpix]

        for _ in range(100):
            ypix, xpix = extendROI(ypix, xpix, Ly, Lx, 5)  # keep extending
            if np.sum(valid_pixels(cell_pix, ypix, xpix)) - nring > min_neuropil_pixels:
                break  # break if there are at least a minimum number of valid pixels
        ix = valid_pixels(cell_pix, ypix, xpix)
        neuropil_mask[ypix[ix], xpix[ix]] = 1.
        inner_ring[:] = 0

    return neuropil_masks / np.sum(neuropil_masks, axis=(1, 2), keepdims=True)
