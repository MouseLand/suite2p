"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter1d
from cellpose.metrics import _intersection_over_union, mask_ious


def square_mask(mask, ly, yi, xi):
    """
    Crop a square patch from a 2D mask centered at a given position.

    Parameters
    ----------
    mask : numpy.ndarray
        2D array to crop from, shape (Lyc, Lxc).
    ly : int
        Half-width of the square patch (output is 2*ly x 2*ly).
    yi : int
        Y-coordinate of the center.
    xi : int
        X-coordinate of the center.

    Returns
    -------
    mask0 : numpy.ndarray
        Cropped square patch of shape (2*ly, 2*ly).
    """
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
    Create a normalized distance array and return indices within a unit circle.

    Parameters
    ----------
    d0 : list of float
        Two-element list [d0_y, d0_x] giving the radius in each dimension.

    Returns
    -------
    rs : numpy.ndarray
        Normalized distance array of shape (2*d0_y+1, 2*d0_x+1).
    dx : numpy.ndarray
        Normalized X-coordinates of pixels within the unit circle.
    dy : numpy.ndarray
        Normalized Y-coordinates of pixels within the unit circle.
    """
    d00 = int(np.round(d0[0]))
    d01 = int(np.round(d0[1]))
    dy = np.tile(np.arange(-d00, d00 + 1) / d00, (2 * d01 + 1, 1))
    dy = dy.transpose()
    dx = np.tile(np.arange(-d01, d01 + 1) / d01, (2 * d00 + 1, 1))

    rs = (dy**2 + dx**2)**0.5
    dx = dx[rs <= 1.]
    dy = dy[rs <= 1.]
    return rs, dx, dy

def hp_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered `mov` by subtracting off the movie smoothed by a gaussian kernel.

    Parameters
    ----------
    mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx).
    width: int
        The standard deviation of the Gaussian filter in time

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered video
    """
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter1d(mov[:, j, :], width, axis=0)
    return mov


def hp_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns high-pass-filtered `mov` by subtracting off the rolling mean in window of `width`.

    Parameters
    ----------
    mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx).
    width: int
        The filter width in time.

    Returns
    -------
    filtered_mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx), high-pass filtered in time.
    
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
    mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx).
    width: int
        The filter width in time.

    Returns
    -------
    filtered_mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx), high-pass filtered in time.
        
    """

    return hp_gaussian_filter(mov, width) if width < 10 else hp_rolling_mean_filter(
        mov, width)  # gaussian is slower


def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: numpy.ndarray
        Movie of shape (nframes, Ly, Lx).
    batch_size: int
        The batch size for computing the standard deviation of the difference.

    Returns
    -------
    sdmov: Ly x Lx
        The standard deviation of the difference across time for each pixel.
    """
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), "float32")
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix + batch_size, :, :], axis=0)**2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov
