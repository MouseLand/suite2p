"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from cellpose.metrics import _intersection_over_union, mask_ious


def square_mask(mask, ly, yi, xi):
    """ crop from mask a square of size ly at position yi,xi """
    Lyc, Lxc = mask.shape
    mask0 = np.zeros((2 * ly, 2 * ly), mask.dtype)
    yinds = [max(0, yi - ly), min(yi + ly, Lyc)]
    xinds = [max(0, xi - ly), min(xi + ly, Lxc)]
    mask0[max(0, ly - yi):min(2 * ly, Lyc + ly - yi),
          max(0, ly - xi):min(2 * ly, Lxc + ly - xi)] = mask[yinds[0]:yinds[1],
                                                             xinds[0]:xinds[1]]
    return mask0


def circleMask(d0):
    """
    creates array with indices which are the radius of that x,y point

    Parameters
    ----------
    d0
        (patch of (-d0,d0+1) over which radius computed

    Returns
    -------
    rs:
        array (2*d0+1,2*d0+1) of radii
    dx:
        indices in rs where the radius is less than d0
    dy:
        indices in rs where the radius is less than d0
    """
    dy = np.tile(np.arange(-d0[0], d0[0] + 1) / d0[0], (2 * d0[1] + 1, 1))
    dy = dy.transpose()
    dx = np.tile(np.arange(-d0[1], d0[1] + 1) / d0[1], (2 * d0[0] + 1, 1))
    
    rs = (dy**2 + dx**2)**0.5
    dx = dx[rs <= 1.]
    dy = dy[rs <= 1.]
    return rs, dx, dy

def hp_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array "mov" using a gaussian kernel.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The kernel width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered video
    """
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def hp_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array "mov" using a non-overlapping rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def temporal_high_pass_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns hp-filtered mov over time, selecting an algorithm for computational performance based on the kernel width.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames
    """

    return hp_gaussian_filter(mov, width) if width < 10 else hp_rolling_mean_filter(
        mov, width)  # gaussian is slower


def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Ly x Lx
        The statistics for each pixel
    """
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), "float32")
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix + batch_size, :, :], axis=0)**2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov
