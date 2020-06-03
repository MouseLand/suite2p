import warnings

import numpy as np
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)
from . import utils

def phasecorr_reference(refImg0, spatial_taper=None, smooth_sigma=None, pad_fft=None, reg_1p=None, spatial_hp=None, pre_smooth=None):
    """ computes masks and fft'ed reference image for phasecorr

    Parameters
    ----------
    refImg0 : 2D array, int16
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
    if reg_1p:
        data = refImg0
        if pre_smooth and pre_smooth % 2:
            raise ValueError("if set, pre_smooth must be a positive even integer.")
        if spatial_hp % 2:
            raise ValueError("spatial_hp must be a positive even integer.")
        data = data.astype(np.float32)

        if pre_smooth:
            data = utils.spatial_smooth(data, int(pre_smooth))
        data = utils.spatial_high_pass(data, int(spatial_hp))
        refImg0 = data

    refImg = refImg0.copy()

    maskSlope = spatial_taper if reg_1p else 3 * smooth_sigma
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


def phasecorr_cpu(data, maskMul, maskOffset, cfRefImg, lcorr, smooth_sigma_time):
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
    nimg = data.shape[0]

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
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2*lcorr+1, 2*lcorr+1))
        cmax[t] = cc[t, ymax[t], xmax[t]]
    ymax, xmax = ymax-lcorr, xmax-lcorr
    return ymax, xmax, cmax


def phasecorr(data, maskMul, maskOffset, cfRefImg, maxregshift, smooth_sigma_time):
    """ compute registration offsets """

    nimg, Ly, Lx = data.shape

    # maximum registration shift allowed
    maxregshift = np.round(maxregshift * np.minimum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly, Lx) / 2.)))

    ymax, xmax, cmax = phasecorr_cpu(data, maskMul, maskOffset, cfRefImg, lcorr, smooth_sigma_time)
    return ymax, xmax, cmax


def shift_data(X, ymax, xmax):
    """ rigid shift X by integer shifts ymax and xmax in place (no return)

    Parameters
    ----------
    X : int16
        array that's frames x Ly x Lx
    ymax : np.ndarray
        shifts in y from cfRefImg to data for each frame
    xmax : np.ndarray
        shifts in x from cfRefImg to data for each frame

    """
    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]
    nimg, Ly, Lx = X.shape
    for n in range(nimg):
        X[n] = np.roll(X[n].copy(), (-ymax[n], -xmax[n]), axis=(0,1))

