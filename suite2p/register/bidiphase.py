import time, os
import numpy as np
from scipy.fftpack import next_fast_len
from numpy import random as rnd
import multiprocessing
#import scipy.fftpack as fft
from numpy import fft
from numba import vectorize, complex64, float32, int16
import math
from scipy.signal import medfilt
from scipy.ndimage import laplace, gaussian_filter1d
from suite2p import register, utils
from skimage.external.tifffile import TiffWriter
from mkl_fft import fft2, ifft2

def compute(frames):
    """ computes the bidirectional phase offset

    sometimes in line scanning there will be offsets between lines;
    if ops['do_bidiphase'], then bidiphase is computed and applied

    Parameters
    ----------
    frames : int16
        random subsample of frames in binary (frames x Ly x Lx)

    Returns
    -------
    bidiphase : int
        bidirectional phase offset in pixels

    """

    Ly = frames.shape[1]
    Lx = frames.shape[2]
    # lines scanned in 1 direction
    yr1 = np.arange(1, np.floor(Ly/2)*2, 2, int)
    # lines scanned in the other direction
    yr2 = np.arange(0, np.floor(Ly/2)*2, 2, int)

    # compute phase-correlation between lines in x-direction
    d1 = fft.fft(frames[:, yr1, :], axis=2)
    d2 = np.conj(fft.fft(frames[:, yr2, :], axis=2))
    d1 = d1 / (np.abs(d1) + eps0)
    d2 = d2 / (np.abs(d2) + eps0)

    #fhg =  gaussian_fft(1, int(np.floor(Ly/2)), Lx)
    cc = np.real(fft.ifft(d1 * d2 , axis=2))#* fhg[np.newaxis, :, :], axis=2))
    cc = cc.mean(axis=1).mean(axis=0)
    cc = fft.fftshift(cc)
    ix = np.argmax(cc[(np.arange(-10,11,1) + np.floor(Lx/2)).astype(int)])
    ix -= 10
    bidiphase = -1*ix

    return bidiphase

def shift(frames, bidiphase):
    """ shift frames by bidirectional phase offset, bidiphase

    sometimes in line scanning there will be offsets between lines;
    shifts last axis by bidiphase

    Parameters
    ----------
    frames : int16
        frames from binary (frames x Ly x Lx)
    bidiphase : int
        bidirectional phase offset in pixels

    Returns
    -------
    frames : int16
        shifted frames from binary (frames x Ly x Lx)

    """

    bidiphase = int(bidiphase)
    nt, Ly, Lx = frames.shape
    yr = np.arange(1, np.floor(Ly/2)*2, 2, int)
    ntr = np.arange(0, nt, 1, int)
    if bidiphase > 0:
        xr = np.arange(bidiphase, Lx, 1, int)
        xrout = np.arange(0, Lx-bidiphase, 1, int)
        frames[np.ix_(ntr, yr, xr)] = frames[np.ix_(ntr, yr, xrout)]
    else:
        xr = np.arange(0, bidiphase+Lx, 1, int)
        xrout = np.arange(-bidiphase, Lx, 1, int)
        frames[np.ix_(ntr, yr, xr)] = frames[np.ix_(ntr, yr, xrout)]
