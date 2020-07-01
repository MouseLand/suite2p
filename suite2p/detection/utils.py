from typing import Tuple, NamedTuple, Sequence, Optional

import numpy as np
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter

from ..io.binary import BinaryFile

def binned_mean(mov: np.ndarray, bin_size) -> np.ndarray:
    """Returns an array with the mean of each time bin (of size 'bin_size')."""
    n_frames, Ly, Lx = mov.shape
    return mov.reshape(-1, bin_size, Ly, Lx).mean(axis=1)


def reject_frames(mov: np.ndarray, bad_indices: Sequence[int], mov_indices: Optional[Sequence[int]] = None, reject_threshold: float = 0.):
    """
    Returns only the frames of 'mov' not in 'bad_indices', if the percentage of bad_indices is higher than reject_threshold.
    Uses the indices of 'mov' by default, but can use alternate indices in 'mov_indices' to match with bad_indices.
    """
    n_frames, Ly, Lx = mov.shape
    indices = mov_indices if mov_indices is not None else np.arange(n_frames)
    if len(indices) != n_frames:
        raise TypeError("'mov_indices' must be the same length as the movie, in order to match them up properly.")
    good_frames = np.setdiff1d(bad_indices, indices, assume_unique=True)
    good_mov = mov[good_frames, :, :] if len(good_frames) / len(indices) > reject_threshold else mov
    return good_mov


def crop(mov: np.ndarray, y_range: Tuple[int, int], x_range: Tuple[int, int]) -> np.ndarray:
    """Returns cropped frames of 'mov' encompassed by y_range and x_range."""
    return mov[:, slice(*y_range), slice(*x_range)]


def bin_movie(filename: str, Ly: int, Lx: int, bin_size: int, n_frames: int, x_range: Tuple[int, int] = (), y_range: Tuple[int, int] = (), bad_frames: Sequence[int] = ()):
    """Returns binned movie [nbins x Ly x Lx] from filename that has bad frames rejected and cropped to (y_range, x_range)"""
    with BinaryFile(Ly=Ly, Lx=Lx, read_file=filename) as f:
        batches = []
        batch_size = min(n_frames - len(bad_frames), 500) // bin_size * bin_size
        for indices, data in f.iter_frames(batch_size=batch_size):

            if len(x_range) and len(y_range):
                data = crop(mov=data, y_range=y_range, x_range=x_range)

            if len(bad_frames) > 0:
                data = reject_frames(mov=data, bad_indices=bad_frames, mov_indices=indices, reject_threshold=0.5)

            if data.shape[0] >= batch_size:  # todo: drops the end of the movie
                dbin = binned_mean(mov=data, bin_size=bin_size)
                batches.append(dbin)

    mov = np.vstack(batches)
    return mov


def high_pass_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a gaussian kernel."""
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def high_pass_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a non-overlapping rolling mean kernel over time."""
    mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """Returns standard deviation of difference between pixels across time, computed in batches of batch_size."""
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), 'float32')
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix+batch_size, :, :], axis=0) ** 2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov


class EllipseData(NamedTuple):
    mu: float
    cov: float
    radii: Tuple[float, float]
    ellipse: np.ndarray

    @property
    def area(self):
        return (self.radii[0] * self.radii[1]) ** 0.5 * np.pi


def fitMVGaus(y, x, lam, thres=2.5, npts: int = 100) -> EllipseData:
    """ computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations.

    Parameters
    ----------
    y : float, array
        pixel locations in y
    x : float, array
        pixel locations in x
    lam : float, array
        weights of each pixel
    """

    # normalize pixel weights
    lam /= lam.sum()

    # mean of gaussian
    yx = np.stack((y, x))
    mu = (lam * yx).sum(axis=1)
    yx = (yx - mu[:, np.newaxis]) * lam ** .5
    cov = yx @ yx.T

    # radii of major and minor axes
    radii, evec = np.linalg.eig(cov)
    radii = thres * np.maximum(0, np.real(radii)) ** .5

    # compute pts of ellipse
    theta = np.linspace(0, 2 * np.pi, npts)
    p = np.stack((np.cos(theta), np.sin(theta)))
    ellipse = (p.T * radii) @ evec.T + mu
    radii = np.sort(radii)[::-1]

    return EllipseData(mu=mu, cov=cov, radii=radii, ellipse=ellipse)


def distance_kernel(radius: int) -> np.ndarray:
    """ Returns 2D array containing geometric distance from center, with radius 'radius'"""
    d = np.arange(-radius, radius + 1)
    dists_2d = norm(np.meshgrid(d, d), axis=0)
    return dists_2d


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
    """Returns time-normed movie values, thresholded by 'intensity_threshold'."""
    return norm(np.where(mov > intensity_threshold, mov, 0), axis=0)
