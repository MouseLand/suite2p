from typing import Tuple

import numpy as np

from .utils import convolve, complex_fft2, spatial_taper, addmultiply, gaussian_fft, temporal_smooth


def compute_masks(refImg, maskSlope) -> Tuple[np.ndarray, np.ndarray]:
    """Returns maskMul and maskOffset from an image and slope parameter"""
    Ly, Lx = refImg.shape
    maskMul = spatial_taper(maskSlope, Ly, Lx)
    maskOffset = refImg.mean() * (1. - maskMul)
    return maskMul.astype('float32'), maskOffset.astype('float32')


def apply_masks(data: np.ndarray, maskMul: np.ndarray, maskOffset: np.ndarray) -> np.ndarray:
    """Returns a 3D image 'data', multiplied by 'maskMul' and then added 'maskOffet'."""
    return addmultiply(data, maskMul, maskOffset)


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
    cfRefImg = complex_fft2(img=refImg, pad_fft=pad_fft)
    cfRefImg /= (1e-5 + np.absolute(cfRefImg))
    cfRefImg *= gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.astype('complex64')


def phasecorr(data, cfRefImg, maxregshift, smooth_sigma_time) -> Tuple[int, int, float]:
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
    
    data = convolve(data, cfRefImg)

    min_dim = np.minimum(*data.shape[1:])  # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))
    cc = np.real(
        np.block(
            [[data[:,  -lcorr:, -lcorr:], data[:,  -lcorr:, :lcorr+1]],
             [data[:, :lcorr+1, -lcorr:], data[:, :lcorr+1, :lcorr+1]]]
        )
    )
    cc = temporal_smooth(cc, smooth_sigma_time) if smooth_sigma_time > 0 else cc

    ymax, xmax = np.zeros(data.shape[0], np.int32), np.zeros(data.shape[0], np.int32)
    for t in np.arange(data.shape[0]):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
    cmax = cc[np.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax.astype(np.float32)


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """returns frame, shifted by dy and dx"""
    return np.roll(frame, (-dy, -dx), axis=(0, 1))
