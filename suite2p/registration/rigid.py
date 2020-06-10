import warnings
from typing import Tuple

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)
from . import utils


def compute_masks(refImg, maskSlope) -> Tuple[np.ndarray, np.ndarray]:
    """Returns maskMul and maskOffset from an image and slope parameter"""
    Ly, Lx = refImg.shape
    maskMul = utils.spatial_taper(maskSlope, Ly, Lx)
    maskOffset = refImg.mean() * (1. - maskMul)
    return maskMul.astype('float32'), maskOffset.astype('float32')


def apply_masks(data: np.ndarray, maskMul: np.ndarray, maskOffset: np.ndarray) -> np.ndarray:
    """Returns a 3D image 'data', multiplied by 'maskMul' and then added 'maskOffet'."""
    return utils.addmultiplytype(data, maskMul, maskOffset)


def phasecorr_reference(refImg: np.ndarray, smooth_sigma=None, pad_fft: bool = False) -> np.ndarray:
    """
    Returns reference image fft'ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation 'smooth_sigma' computes fft'ed reference image for phasecorr.

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns
    -------
    cfRefImg : 2D array, complex64
    """
    Ly, Lx = refImg.shape
    cfRefImg = np.conj(fft2(refImg, (next_fast_len(Ly), next_fast_len(Lx)))) if pad_fft else np.conj(fft2(refImg))
    cfRefImg /= (1e-5 + np.absolute(cfRefImg))
    cfRefImg *= utils.gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.astype('complex64')


def phasecorr(data, cfRefImg, maxregshift, smooth_sigma_time):
    """ compute phase correlation between data and reference image

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    lcorr : int
        maximum shift in pixels
    smooth_sigma_time : float
        how many frames to smooth in time

    Returns
    -------
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame

    """

    # maximum registration shift allowed
    min_dim = np.minimum(*data.shape[1:])
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))

    # shifts and corrmax
    X = data
    fft2(X, overwrite_x=True)
    X = utils.apply_dotnorm(X, cfRefImg)
    ifft2(X, overwrite_x=True)
    cc = np.real(
        np.block(
            [[X[:,  -lcorr:, -lcorr:], X[:,  -lcorr:, :lcorr+1]],
             [X[:, :lcorr+1, -lcorr:], X[:, :lcorr+1, :lcorr+1]]]
        )
    )
    if smooth_sigma_time > 0:
        cc = gaussian_filter1d(cc, smooth_sigma_time, axis=0)

    ymax, xmax = np.zeros(data.shape[0], np.int32), np.zeros(data.shape[0], np.int32)
    for t in np.arange(X.shape[0]):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
    cmax = cc[np.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax.astype(np.float32)


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """returns frame, shifted by dy and dx"""
    return np.roll(frame, (-dy, -dx), axis=(0, 1))
