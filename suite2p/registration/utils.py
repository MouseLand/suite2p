import warnings
from functools import lru_cache

import numpy as np
from numba import vectorize, complex64
from numpy import fft
from scipy.fftpack import next_fast_len
from scipy.ndimage import gaussian_filter1d

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)


@vectorize([complex64(complex64, complex64)], nopython=True, target='parallel')
def apply_dotnorm(Y, cfRefImg):
    return Y / (np.complex64(1e-5) + np.abs(Y)) * cfRefImg


@vectorize(['complex64(int16, float32, float32)', 'complex64(float32, float32, float32)'], nopython=True, target='parallel', cache=True)
def addmultiply(x, mul, add):
    return np.complex64(np.float32(x) * mul + add)


def combine_offsets_across_batches(offset_list, rigid):
    yoff, xoff, corr_xy = [], [], []
    for batch in offset_list:
        yoff.append(batch[0])
        xoff.append(batch[1])
        corr_xy.append(batch[2])
    if rigid:
        return np.hstack(yoff), np.hstack(xoff), np.hstack(corr_xy)
    else:
        return np.vstack(yoff), np.vstack(xoff), np.vstack(corr_xy)


def meshgrid_mean_centered(x, y):
    x = np.arange(0, x)
    y = np.arange(0, y)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def gaussian_fft(sig, Ly, Lx):
    ''' gaussian filter in the fft domain with std sig and size Ly,Lx '''
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg


def spatial_taper(sig, Ly, Lx):
    ''' spatial taper  on edges with gaussian of std sig '''
    xx, yy = meshgrid_mean_centered(x=Lx, y=Ly)
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1. / (1. + np.exp((yy - mY) / sig))
    maskX = 1. / (1. + np.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul

def temporal_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """returns Gaussian filtered 'frames' ndarray over first dimension"""
    return gaussian_filter1d(data, sigma=sigma, axis=0)


def spatial_smooth(data, window):
    """spatially smooth data using cumsum over axis=1,2 with window N"""
    if window and window % 2:
        raise ValueError("Filter window must be an even integer.")
    if data.ndim == 2:
        data = data[np.newaxis, : ,:]

    half_pad = window // 2
    data_padded = np.pad(data, ((0, 0), (half_pad, half_pad), (half_pad, half_pad)), mode='constant', constant_values=0)

    data_summed = data_padded.cumsum(axis=1).cumsum(axis=2, dtype=np.float32)
    data_summed = (data_summed[:, window:, :] - data_summed[:, :-window, :])  # in X
    data_summed = (data_summed[:, :, window:] - data_summed[:, :, :-window])  # in Y
    data_summed /= window ** 2
    
    return data_summed.squeeze()


def spatial_high_pass(data, N):
    """high pass filters data over axis=1,2 with window N"""
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    data_filtered = data - (spatial_smooth(data, N) / spatial_smooth(np.ones((1, data.shape[1], data.shape[2])), N))
    return data_filtered.squeeze()


def convolve(mov: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Returns the 3D array 'mov' convolved by a 2D array 'img'."""
    return ifft2(apply_dotnorm(fft2(mov), img))


def complex_fft2(img: np.ndarray, pad_fft: bool = False) -> np.ndarray:
    """Returns the complex conjugate of the fft-transformed 2D array 'img', optionally padded for speed."""
    Ly, Lx = img.shape
    return np.conj(fft2(img, (next_fast_len(Ly), next_fast_len(Lx)))) if pad_fft else np.conj(fft2(img))


def kernelD(xs: np.ndarray, ys: np.ndarray, sigL: float = 0.85) -> np.ndarray:
    """Gaussian kernel from xs (1D array) to ys (1D array), with the 'sigL' smoothing width for up-sampling kernels, (best between 0.5 and 1.0)"""
    xs0, xs1 = np.meshgrid(xs, xs)
    ys0, ys1 = np.meshgrid(ys, ys)
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    K = np.exp(-(dxs ** 2 + dys ** 2) / (2 * sigL ** 2))
    return K


def kernelD2(xs: int, ys: int) -> np.ndarray:
    ys, xs = np.meshgrid(xs, ys)
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = np.exp(-((ys - ys.T) ** 2 + (xs - xs.T) ** 2))
    R = R / np.sum(R, axis=0)
    return R


@lru_cache(maxsize=5)
def mat_upsample(lpad, subpixel: int = 10):
    """ upsampling matrix using gaussian kernels """
    lar = np.arange(-lpad, lpad + 1)
    larUP = np.arange(-lpad, lpad + .001, 1. / subpixel)
    nup = larUP.shape[0]
    Kmat = np.linalg.inv(kernelD(lar, lar)) @ kernelD(lar, larUP)
    return Kmat, nup