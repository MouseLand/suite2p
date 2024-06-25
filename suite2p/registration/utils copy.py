"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import warnings
from typing import Tuple

import numpy as np
from numpy.fft import ifftshift  as ifftshift0 #, fft2, ifft2
from scipy.ndimage import gaussian_filter1d
import torch

try:
    # pytorch > 1.7
    from torch.fft import fft, fft2, ifft, ifft2, fftshift, ifftshift
except:
    # pytorch <= 1.7
    raise ImportError("pytorch version > 1.7 required")

eps = torch.complex(torch.tensor(1e-5), torch.tensor(0.0))

def convolve(mov: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Returns the 3D array "mov" convolved by a 2D array "img".

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to process
    img: 2D array
        The convolution kernel
    lcorr: int (optional)
        amount to crop cross-correlation

    Returns
    -------
    convolved_data: nImg x Ly x Lx
    """
    mov = fft2(mov)
    mov /= (eps + torch.abs(mov))
    mov *= img
    mov = torch.real(ifft2(mov))
    return mov


def meshgrid_mean_centered(x: int, y: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns a mean-centered meshgrid

    Parameters
    ----------
    x: int
        The height of the meshgrid
    y: int
        The width of the mehgrid

    Returns
    -------
    xx: int array
    yy: int array
    """
    
    x = np.arange(0, x)
    y = np.arange(0, y)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    return xx, yy




def spatial_taper(sig, Ly, Lx):
    """
    Returns spatial taper  on edges with gaussian of std sig

    Parameters
    ----------
    sig
    Ly: int
        frame height
    Lx: int
        frame width

    Returns
    -------
    maskMul


    """
    y = torch.arange(0, Ly, dtype=torch.float)
    y -= y.mean()
    x = torch.arange(0, Lx, dtype=torch.float)
    x -= x.mean()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    mY = ((Ly - 1) / 2) - 2 * sig
    mX = ((Lx - 1) / 2) - 2 * sig
    maskY = 1. / (1. + torch.exp((yy - mY) / sig))
    maskX = 1. / (1. + torch.exp((xx - mX) / sig))
    maskMul = maskY * maskX
    return maskMul


def temporal_smooth(data: np.ndarray, sigma: float) -> np.ndarray:
    """
    Returns Gaussian filtered "frames" ndarray over first dimension

    Parameters
    ----------
    data: nImg x Ly x Lx
    sigma: float
        windowing parameter

    Returns
    -------
    smoothed_data: nImg x Ly x Lx
        Smoothed data

    """
    return gaussian_filter1d(data, sigma=sigma, axis=0)


def spatial_smooth(data: np.ndarray, window: int):
    """
    Spatially smooth data using cumsum over axis=1,2 with window N

    Parameters
    ----------
    data: Ly x Lx
        The image to smooth.
    window: int
        The window size

    Returns
    -------
    smoothed_data: Ly x Lx
        The smoothed frame

    """
    if window and window % 2:
        raise ValueError("Filter window must be an even integer.")
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    half_pad = window // 2
    data_padded = np.pad(data, ((0, 0), (half_pad, half_pad), (half_pad, half_pad)),
                         mode="constant", constant_values=0)

    data_summed = data_padded.cumsum(axis=1).cumsum(axis=2, dtype=np.float32)
    data_summed = (data_summed[:, window:, :] - data_summed[:, :-window, :])  # in X
    data_summed = (data_summed[:, :, window:] - data_summed[:, :, :-window])  # in Y
    data_summed /= window**2

    return data_summed.squeeze()


def spatial_high_pass(data, N):
    """
    high pass filters data over axis=1,2 with window N

    Parameters
    ----------
    data: Ly x Lx
        The image to smooth.
    N: int
        The window size

    Returns
    -------
    smoothed_data: Ly x Lx
        The smoothed frame
    """
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    data_filtered = data - (spatial_smooth(data, N) /
                            spatial_smooth(np.ones(
                                (1, data.shape[1], data.shape[2])), N))
    return data_filtered.squeeze()


def complex_fft2(img: np.ndarray) -> np.ndarray:
    """
    Returns the complex conjugate of the fft-transformed 2D array "img", optionally padded for speed.

    Parameters
    ----------
    img: Ly x Lx
        The image to process
    pad_fft: bool
        Whether to pad the image


    """
    Ly, Lx = img.shape
    return torch.conj(fft2(img)) 

def gaussian_kernel(sigma_y, sigma_x, Ly, Lx, device=torch.device("cpu")):
    """
    Generates a 2D Gaussian kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        Ly (int): Number of pixels in the y-axis.
        Lx (int): Number of pixels in the x-axis.
        device (torch.device, optional): Device to store the kernel tensor. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: 2D Gaussian kernel tensor.

    """
    y = torch.arange(0, Ly, device=device, dtype=torch.float)
    y -= y.mean()
    x = torch.arange(0, Lx, device=device, dtype=torch.float)
    x -= x.mean()
    ky = torch.exp(-y**2 / (2 * sigma_y**2)) 
    kx = torch.exp(-x**2 / (2 * sigma_x**2))
    kernel = ky[:, None] * kx
    kernel /= kernel.sum()
    return kernel

def gaussian_fft(sig, Ly: int, Lx: int):
    kernel = gaussian_kernel(sig, sig, Ly, Lx)
    return torch.real(fft2(ifftshift(kernel)))

def ref_smooth_fft(refImg: np.ndarray, smooth_sigma=None) -> np.ndarray:
    """
    Returns reference image fft"ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation "smooth_sigma" computes fft"ed reference image for phasecorr.

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns
    -------
    cfRefImg : 2D array, complex64
    """
    cfRefImg = complex_fft2(img=refImg)
    cfRefImg /= (1e-5 + torch.abs(cfRefImg))
    cfRefImg *= gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
    return cfRefImg.type(torch.complex64)

def kernelD(xs: np.ndarray, ys: np.ndarray, sigL: float = 0.85) -> np.ndarray:
    """
    Gaussian kernel from xs (1D array) to ys (1D array), with the "sigL" smoothing width for up-sampling kernels, (best between 0.5 and 1.0)

    Parameters
    ----------
    xs:
    ys
    sigL

    Returns
    -------

    """
    xs0, xs1 = torch.meshgrid(xs, xs, indexing="ij")
    ys0, ys1 = torch.meshgrid(ys, ys, indexing="ij")
    dxs = xs0.reshape(-1, 1) - ys0.reshape(1, -1)
    dys = xs1.reshape(-1, 1) - ys1.reshape(1, -1)
    K = torch.exp(-(dxs**2 + dys**2) / (2 * sigL**2))
    return K

def kernelD2(xs: int, ys: int) -> np.ndarray:
    """
    Parameters
    ----------
    xs
    ys

    Returns
    -------

    """
    xs, ys = torch.meshgrid(xs, ys, indexing="ij")
    ys = ys.flatten().reshape(1, -1)
    xs = xs.flatten().reshape(1, -1)
    R = torch.exp(-((ys - ys.T)**2 + (xs - xs.T)**2))
    R = R / torch.sum(R, axis=0)
    return R


# def mat_upsample(lpad: int, subpixel: int = 10):
#     """
#     upsampling matrix using gaussian kernels

#     Parameters
#     ----------
#     lpad: int
#     subpixel: int

#     Returns
#     -------
#     Kmat: np.ndarray
#     nup: int
#     """
#     #kernel0 gaussian_kernel(sigma_y=0.85, sigma_x=0.85, Ly=2 * lpad + 1, Lx=2 * lpad + 1)
#     lar = np.arange(-lpad, lpad + 1)
#     larUP = np.arange(-lpad, lpad + .001, 1. / subpixel)
#     nup = larUP.shape[0]
#     Kmat = np.linalg.inv(kernelD(lar, lar)) @ kernelD(lar, larUP)
#     return Kmat, nup


def mat_upsample(lpad: int, subpixel: int = 10, device=torch.device("cpu")):
    xs = torch.arange(-lpad, lpad + 1, device=device)
    xs_up = torch.arange(-lpad, lpad + .001, 1. / subpixel, device=device)
    kernel0 = kernelD(xs, xs)
    kernel_up = kernelD(xs, xs_up)
    #Ly *= subpixel
    #kernelUp = utils.gaussian_kernel(sigma_y=0.85*subpixel, sigma_x=0.85*subpixel, Ly=Ly, Lx=Ly)
    Kmat = torch.linalg.solve(kernel0, kernel_up)
    nup = len(xs_up)
    return Kmat, nup


def highpass_mean_image(I, aspect=1.):
    """ computes enhanced mean image

    """
    Ly, Lx = I.shape
    sigma_y, sigma_x = 3., 3.
    if aspect != 1.:
        sigma_y *= aspect

    mimg = I.copy()
    # high-pass filter
    mimg = torch.from_numpy(mimg)
    kernel = -1 * gaussian_kernel(sigma_y, sigma_x, Ly, Lx)
    kernel[Ly // 2, Lx // 2] = 1
    
    fhp = fft2(kernel)
    img_filt = torch.real(ifft2(
                fft2(mimg) * torch.conj(fhp)))
    img_filt = fftshift(img_filt)
    img_filt = img_filt.numpy()
    i01, i99 = np.percentile(img_filt, [1, 99])
    img_filt = (img_filt - i01) / (i99 - i01)
    img_filt = np.clip(img_filt, 0, 1)

    return img_filt
