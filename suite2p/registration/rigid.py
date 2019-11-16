import time, os
import numpy as np
from scipy.fftpack import next_fast_len
from numpy import fft
from numba import vectorize, complex64, float32, int16
import math
from scipy.ndimage import gaussian_filter1d
from mkl_fft import fft2, ifft2
from . import utils

def phasecorr_reference(refImg0, ops):
    """ computes masks and fft'ed reference image for phasecorr

    Parameters
    ----------
    refImg0 : int16
        reference image
    ops : dictionary
        requires 'smooth_sigma'
        (if ```ops['1Preg']```, need 'spatial_taper', 'spatial_hp', 'pre_smooth')

    Returns
    -------
    maskMul : float32
        mask that is multiplied to spatially taper frames
    maskOffset : float32
        shifts in x from cfRefImg to data for each frame
    cfRefImg : complex64
        reference image fft'ed and complex conjugate and multiplied by gaussian
        filter in the fft domain with standard deviation 'smooth_sigma'

    """
    refImg = refImg0.copy()
    if '1Preg' in ops and ops['1Preg']:
        maskSlope    = ops['spatial_taper'] # slope of taper mask at the edges
    else:
        maskSlope    = 3 * ops['smooth_sigma'] # slope of taper mask at the edges
    Ly,Lx = refImg.shape
    maskMul = utils.spatial_taper(maskSlope, Ly, Lx)

    if ops['1Preg']:
        refImg = utils.one_photon_preprocess(refImg[np.newaxis,:,:], ops).squeeze()
    maskOffset = refImg.mean() * (1. - maskMul);

    # reference image in fourier domain
    if 'pad_fft' in ops and ops['pad_fft']:
        cfRefImg   = np.conj(fft2(refImg,
                            (next_fast_len(Ly), next_fast_len(Lx))))
    else:
        cfRefImg   = np.conj(fft2(refImg))

    absRef     = np.absolute(cfRefImg)
    cfRefImg   = cfRefImg / (1e-5 + absRef)

    # gaussian filter in space
    fhg = utils.gaussian_fft(ops['smooth_sigma'], cfRefImg.shape[0], cfRefImg.shape[1])
    cfRefImg *= fhg

    maskMul = maskMul.astype('float32')
    maskOffset = maskOffset.astype('float32')
    cfRefImg = cfRefImg.astype('complex64')
    cfRefImg = np.reshape(cfRefImg, (1, cfRefImg.shape[0], cfRefImg.shape[1]))
    return maskMul, maskOffset, cfRefImg


@vectorize(['complex64(int16, float32, float32)', 'complex64(float32, float32, float32)'], nopython=True, target = 'parallel')
def addmultiplytype(x,y,z):
    return np.complex64(np.float32(x)*y + z)

def clip(X, lcorr):
    """ perform 2D fftshift and crop with lcorr """
    x00 = X[:,  :lcorr+1, :lcorr+1]
    x11 = X[:,  -lcorr:, -lcorr:]
    x01 = X[:,  :lcorr+1, -lcorr:]
    x10 = X[:,  -lcorr:, :lcorr+1]
    return x00, x01, x10, x11

def phasecorr_cpu(data, refAndMasks, lcorr, smooth_sigma_time):
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
    maskMul    = refAndMasks[0]
    maskOffset = refAndMasks[1]
    cfRefImg   = refAndMasks[2].squeeze()

    nimg = data.shape[0]
    ly,lx = cfRefImg.shape[-2:]
    lyhalf = int(np.floor(ly/2))
    lxhalf = int(np.floor(lx/2))

    # shifts and corrmax
    ymax = np.zeros((nimg,), np.int32)
    xmax = np.zeros((nimg,), np.int32)
    cmax = np.zeros((nimg,), np.float32)

    X = addmultiplytype(data, maskMul, maskOffset)
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


def phasecorr(data, refAndMasks, ops):
    """ compute registration offsets """
    nimg, Ly, Lx = data.shape

    # maximum registration shift allowed
    maxregshift = np.round(ops['maxregshift'] *np.maximum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly,Lx)/2.)))

    # preprocessing for 1P recordings
    if ops['1Preg']:
        #data = data.copy().astype(np.float32)
        X = utils.one_photon_preprocess(data.copy().astype(np.float32), ops).astype(np.int16)

    ymax, xmax, cmax = phasecorr_cpu(data, refAndMasks, lcorr, ops['smooth_sigma_time'])

    return ymax, xmax, cmax

def shift_data(X, ymax, xmax):
    """ rigid shift X by integer shifts ymax and xmax in place (no return)

    Parameters
    ----------
    X : int16
        array that's frames x Ly x Lx
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame

    """

    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]
    nimg, Ly, Lx = X.shape
    for n in range(nimg):
        X[n] = np.roll(X[n].copy(), (-ymax[n], -xmax[n]), axis=(0,1))
        #yrange = np.arange(0, Ly,1,int) + ymax[n]
        #xrange = np.arange(0, Lx,1,int) + xmax[n]
        #yrange = yrange[np.logical_or(yrange<0, yrange>Ly-1)] - ymax[n]
        #xrange = xrange[np.logical_or(xrange<0, xrange>Lx-1)] - xmax[n]
        #X[n][yrange, :] = m0
        #X[n][:, xrange] = m0


def shift_data_subpixel(inputs):
    ''' rigid shift of X by ymax and xmax '''
    ''' allows subpixel shifts '''
    ''' ** not being used ** '''
    X, ymax, xmax, pad_fft = inputs
    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]

    nimg, Ly0, Lx0 = X.shape
    if pad_fft:
        X = fft2(X.astype('float32'), (next_fast_len(Ly0), next_fast_len(Lx0)))
    else:
        X = fft2(X.astype('float32'))
    nimg, Ly, Lx = X.shape
    Ny = fft.ifftshift(np.arange(-np.fix(Ly/2), np.ceil(Ly/2)))
    Nx = fft.ifftshift(np.arange(-np.fix(Lx/2), np.ceil(Lx/2)))
    [Nx,Ny] = np.meshgrid(Nx,Ny)
    Nx = Nx.astype('float32') / Lx
    Ny = Ny.astype('float32') / Ly
    dph = Nx * np.reshape(xmax, (-1,1,1)) + Ny * np.reshape(ymax, (-1,1,1))
    Y = np.real(ifft2(X * np.exp((2j * np.pi) * dph)))
    # crop back to original size
    if Ly0<Ly or Lx0<Lx:
        Lyhalf = int(np.floor(Ly/2))
        Lxhalf = int(np.floor(Lx/2))
        Y = Y[np.ix_(np.arange(0,nimg,1,int),
                     np.arange(-np.fix(Ly0/2), np.ceil(Ly0/2),1,int) + Lyhalf,
                     np.arange(-np.fix(Lx0/2), np.ceil(Lx0/2),1,int) + Lxhalf)]
    return Y
