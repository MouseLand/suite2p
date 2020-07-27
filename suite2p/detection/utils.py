import numpy as np
from scipy.ndimage import gaussian_filter


def hp_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a gaussian kernel."""
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def hp_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a non-overlapping rolling mean kernel over time."""
    mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def temporal_high_pass_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns hp-filtered mov over time, selecting an algorithm for computational performance based on the kernel width."""
    return hp_gaussian_filter(mov, width) if width < 10 else hp_rolling_mean_filter(mov, width)  # gaussian is slower


def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """Returns standard deviation of difference between pixels across time, computed in batches of batch_size."""
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), 'float32')
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix+batch_size, :, :], axis=0) ** 2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov


def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """Returns a pixel-downsampled movie from 'mov', tapering the edges of 'taper_edge' is True."""
    n_frames, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, int(np.ceil(Ly / 2)), Lx), 'float32')
    movd[:, :Ly//2, :] = np.mean([mov[:, 0:-1:2, :], mov[:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, -1, :] = mov[:, -1, :] / 2 if taper_edge else mov[:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), 'float32')
    mov2[:, :, :Lx//2] = np.mean([movd[:, :, 0:-1:2], movd[:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:, :, -1] = movd[:, :, -1] / 2 if taper_edge else movd[:, :, -1]

    return mov2


def threshold_reduce(mov: np.ndarray, intensity_threshold: float) -> np.ndarray:
    """Returns standard deviation of pixels, thresholded by 'intensity_threshold'.
    Run in a loop to reduce memory footprint.
    """
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp,Lxp), 'float32')
    for t in range(nbinned):
        Vt += mov[t]**2 * (mov[t] > intensity_threshold)
    Vt = Vt**.5
    return Vt

