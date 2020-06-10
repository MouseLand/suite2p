import warnings

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)
from . import utils

def phasecorr_reference(refImg, maskSlope, smooth_sigma=None, pad_fft=None):
    """ computes masks and fft'ed reference image for phasecorr

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns
    -------
    maskMul : 2D array
        mask that is multiplied to spatially taper
    maskOffset : 2D array
        shifts in x from cfRefImg to data for each frame
    cfRefImg : 2D array, complex64
        reference image fft'ed and complex conjugate and multiplied by gaussian
        filter in the fft domain with standard deviation 'smooth_sigma'
    """

    Ly, Lx = refImg.shape
    maskMul = utils.spatial_taper(maskSlope, Ly, Lx)

    refImg = refImg.squeeze()

    maskOffset = refImg.mean() * (1. - maskMul)

    # reference image in fourier domain
    cfRefImg = np.conj(fft2(refImg, (next_fast_len(Ly), next_fast_len(Lx)))) if pad_fft else np.conj(fft2(refImg))

    absRef = np.absolute(cfRefImg)
    cfRefImg = cfRefImg / (1e-5 + absRef)

    # gaussian filter in space
    fhg = utils.gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    cfRefImg *= fhg

    maskMul = maskMul.astype('float32')
    maskOffset = maskOffset.astype('float32')
    cfRefImg = cfRefImg.astype('complex64')
    cfRefImg = np.reshape(cfRefImg, (1, cfRefImg.shape[0], cfRefImg.shape[1]))
    return maskMul, maskOffset, cfRefImg


def clip(X, lcorr):
    """ perform 2D fftshift and crop with lcorr """
    x00 = X[:, :lcorr+1, :lcorr+1]
    x11 = X[:, -lcorr:, -lcorr:]
    x01 = X[:, :lcorr+1, -lcorr:]
    x10 = X[:, -lcorr:, :lcorr+1]
    return x00, x01, x10, x11


def phasecorr(data, maskMul, maskOffset, cfRefImg, maxregshift, smooth_sigma_time):
    """ compute phase correlation between data and reference image

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    refAndMasks : list
        maskMul, maskOffset and cfRefImg (from prepare_refAndMasks)
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

    nimg, Ly, Lx = data.shape

    # maximum registration shift allowed
    maxregshift = np.round(maxregshift * np.minimum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly, Lx) / 2.)))

    # shifts and corrmax
    ymax = np.zeros((nimg,), np.int32)
    xmax = np.zeros((nimg,), np.int32)
    cmax = np.zeros((nimg,), np.float32)

    X = utils.addmultiplytype(data, maskMul, maskOffset)
    for t in range(X.shape[0]):
        fft2(X[t], overwrite_x=True)
    X = utils.apply_dotnorm(X, cfRefImg)
    for t in np.arange(nimg):
        ifft2(X[t], overwrite_x=True)
    x00, x01, x10, x11 = clip(X, lcorr)
    cc = np.real(np.block([[x11, x10], [x01, x00]]))
    if smooth_sigma_time > 0:
        cc = gaussian_filter1d(cc, smooth_sigma_time, axis=0)
    for t in np.arange(nimg):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
        cmax[t] = cc[t, ymax[t], xmax[t]]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """returns frame, shifted by dy and dx"""
    return np.roll(frame, (-dy, -dx), axis=(0, 1))
