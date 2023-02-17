import warnings
from typing import Tuple

import numpy as np
from numba import float32, njit, prange
from numpy import fft
from scipy.fftpack import next_fast_len

from .utils import addmultiply, spatial_taper, gaussian_fft, kernelD2, mat_upsample, convolve


def calculate_nblocks(L: int, block_size: int = 128) -> Tuple[int, int]:
    """
    Returns block_size and nblocks from dimension length and desired block size

    Parameters
    ----------
    L: int
    block_size: int

    Returns
    -------
    block_size: int
    nblocks: int
    """
    return (L, 1) if block_size >= L else (block_size, int(np.ceil(1.5 * L / block_size)))


def make_blocks(Ly, Lx, block_size=(128, 128)):
    """
    Computes overlapping blocks to split FOV into to register separately

    Parameters
    ----------
    Ly: int
        Number of pixels in the vertical dimension
    Lx: int
        Number of pixels in the horizontal dimension
    block_size: int, int
        block size

    Returns
    -------
    yblock: float array
    xblock: float array
    nblocks: int, int
    block_size: int, int
    NRsm: array
    """

    block_size_y, ny = calculate_nblocks(L=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(L=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    # todo: could rounding to int here over-represent some pixels over others?
    ystart = np.linspace(0, Ly - block_size[0], ny).astype('int')
    xstart = np.linspace(0, Lx - block_size[1], nx).astype('int')
    yblock = [np.array([ystart[iy], ystart[iy] + block_size[0]]) for iy in range(ny) for _ in range(nx)]
    xblock = [np.array([xstart[ix], xstart[ix] + block_size[1]]) for _ in range(ny) for ix in range(nx)]

    NRsm = kernelD2(xs=np.arange(nx), ys=np.arange(ny)).T

    return yblock, xblock, [ny, nx], block_size, NRsm


def phasecorr_reference(refImg0: np.ndarray, maskSlope, smooth_sigma, yblock: np.ndarray, xblock: np.ndarray):
    """
    Computes taper and fft'ed reference image for phasecorr.

    Parameters
    ----------
    refImg0: array
    maskSlope
    smooth_sigma
    yblock: float array
    xblock: float array
    
    Returns
    -------
    maskMul
    maskOffset
    cfRefImg

    """
    nb, Ly, Lx = len(yblock), yblock[0][1] - yblock[0][0], xblock[0][1] - xblock[0][0]
    dims = (nb, Ly, Lx)
    cfRef_dims = dims
    gaussian_filter = gaussian_fft(smooth_sigma, *cfRef_dims[1:])
    cfRefImg1 = np.zeros(cfRef_dims, 'complex64')

    maskMul = spatial_taper(maskSlope, *refImg0.shape)
    maskMul1 = np.zeros(dims, 'float32')
    maskMul1[:] = spatial_taper(2 * smooth_sigma, Ly, Lx)
    maskOffset1 = np.zeros(dims, 'float32')
    for yind, xind, maskMul1_n, maskOffset1_n, cfRefImg1_n in zip(yblock, xblock, maskMul1, maskOffset1, cfRefImg1):
        ix = np.ix_(np.arange(yind[0], yind[-1]).astype('int'), np.arange(xind[0], xind[-1]).astype('int'))
        refImg = refImg0[ix]

        # mask params
        maskMul1_n *= maskMul[ix]
        maskOffset1_n[:] = refImg.mean() * (1. - maskMul1_n)

        # gaussian filter
        cfRefImg1_n[:] = np.conj(fft.fft2(refImg))
        cfRefImg1_n /= 1e-5 + np.absolute(cfRefImg1_n)
        cfRefImg1_n[:] *= gaussian_filter

    return maskMul1[:, np.newaxis, :, :], maskOffset1[:, np.newaxis, :, :], cfRefImg1[:, np.newaxis, :, :]


def getSNR(cc: np.ndarray, lcorr: int, lpad: int) -> float:
    """
    Compute SNR of phase-correlation.

    Parameters
    ----------
    cc: nimg x Ly x Lx
        The frame data to analyze
    lcorr: int
    lpad: int
        border padding width

    Returns
    -------
    snr: float
    """
    cc0 = cc[:, lpad:-lpad, lpad:-lpad].reshape(cc.shape[0], -1)
    # set to 0 all pts +-lpad from ymax,xmax
    cc1 = cc.copy()
    for c1, ymax, xmax in zip(cc1, *np.unravel_index(np.argmax(cc0, axis=1), (2 * lcorr + 1, 2 * lcorr + 1))):
        c1[ymax:ymax + 2 * lpad, xmax:xmax + 2 * lpad] = 0

    snr = np.amax(cc0, axis=1) / np.maximum(1e-10, np.amax(cc1.reshape(cc.shape[0], -1), axis=1))  # ensure positivity for outlier cases
    return snr


def phasecorr(data: np.ndarray, maskMul, maskOffset, cfRefImg, snr_thresh, NRsm, xblock, yblock, maxregshiftNR, subpixel: int = 10, lpad: int = 3):
    """
    Compute phase correlations for each block
    
    Parameters
    ----------
    data : nimg x Ly x Lx
    maskMul: ndarray
        gaussian filter
    maskOffset: ndarray
        mask offset
    cfRefImg
        FFT of reference image
    snr_thresh : float
        signal to noise ratio threshold
    NRsm
    xblock: float array
    yblock: float array
    maxregshiftNR: int
    subpixel: int
    lpad: int
        upsample from a square +/- lpad

    Returns
    -------
    ymax1
    xmax1
    cmax1
    """
    
    Kmat, nup = mat_upsample(lpad=3)

    nimg = data.shape[0]
    ly, lx = cfRefImg.shape[-2:]

    # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshiftNR), np.floor(np.minimum(ly, lx) / 2.) - lpad))
    nb = len(yblock)

    # shifts and corrmax
    Y = np.zeros((nimg, nb, ly, lx), 'float32')
    for n in range(nb):
        yind, xind = yblock[n], xblock[n]
        Y[:,n] = data[:, yind[0]:yind[-1], xind[0]:xind[-1]]
    Y = addmultiply(Y, maskMul, maskOffset)
    batch = min(64, Y.shape[1]) #16
    for n in np.arange(0, nb, batch):
        nend = min(Y.shape[1], n+batch)
        Y[:,n:nend] = convolve(mov=Y[:,n:nend], img=cfRefImg[n:nend])

    # calculate ccsm
    lhalf = lcorr + lpad
    cc0 = np.real(
        np.block(
            [[Y[:, :, -lhalf:,    -lhalf:], Y[:, :, -lhalf:,    :lhalf + 1]],
             [Y[:, :, :lhalf + 1, -lhalf:], Y[:, :, :lhalf + 1, :lhalf + 1]]]
        )
    )
    cc0 = cc0.transpose(1, 0, 2, 3)
    cc0 = cc0.reshape(cc0.shape[0], -1)

    cc2 = [cc0, NRsm @ cc0, NRsm @ NRsm @ cc0]
    cc2 = [c2.reshape(nb, nimg, 2 * lcorr + 2 * lpad + 1, 2 * lcorr + 2 * lpad + 1) for c2 in cc2]
    ccsm = cc2[0]
    for n in range(nb):
        snr = np.ones(nimg, 'float32')
        for j, c2 in enumerate(cc2):
            ism = snr < snr_thresh
            if np.sum(ism) == 0:
                break
            cc = c2[n, ism, :, :]
            if j > 0:
                ccsm[n, ism, :, :] = cc
            snr[ism] = getSNR(cc, lcorr, lpad)

    # calculate ymax1, xmax1, cmax1
    mdpt = nup // 2
    ymax1 = np.empty((nimg, nb), np.float32)
    cmax1 = np.empty((nimg, nb), np.float32)
    xmax1 = np.empty((nimg, nb), np.float32)
    ymax = np.empty((nb,), np.int32)
    xmax = np.empty((nb,), np.int32)
    
    for t in range(nimg):
        ccmat = np.empty((nb, 2*lpad+1, 2*lpad+1), np.float32)
        for n in range(nb):
            ix = np.argmax(ccsm[n, t][lpad:-lpad, lpad:-lpad], axis=None)
            ym, xm = np.unravel_index(ix, (2 * lcorr + 1, 2 * lcorr + 1))
            ccmat[n] = ccsm[n, t][ ym:ym + 2 * lpad + 1, xm:xm + 2 * lpad + 1]
            ymax[n], xmax[n] = ym - lcorr, xm - lcorr
        ccb = ccmat.reshape(nb, -1) @ Kmat
        cmax1[t] = np.amax(ccb, axis=1)
        ymax1[t], xmax1[t] = np.unravel_index(np.argmax(ccb, axis=1), (nup, nup))
        ymax1[t] = (ymax1[t] - mdpt) / subpixel + ymax
        xmax1[t] = (xmax1[t] - mdpt) / subpixel + xmax

    return ymax1, xmax1, cmax1


@njit(['(int16[:, :],float32[:,:], float32[:,:], float32[:,:])', 
        '(float32[:, :],float32[:,:], float32[:,:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y) -> None:
    """
    In-place bilinear transform of image 'I' with ycoordinates yc and xcoordinates xc to Y
    
    Parameters
    -------------
    I : Ly x Lx
    yc : Ly x Lx
        new y coordinates
    xc : Ly x Lx
        new x coordinates
    Y : Ly x Lx
        shifted I
    """
    Ly,Lx = I.shape
    yc_floor = yc.astype(np.int32)
    xc_floor = xc.astype(np.int32)
    yc = yc - yc_floor
    xc = xc - xc_floor
    for i in range(yc_floor.shape[0]):
        for j in range(yc_floor.shape[1]):
            yf = min(Ly-1, max(0, yc_floor[i,j]))
            xf = min(Lx-1, max(0, xc_floor[i,j]))
            yf1= min(Ly-1, yf+1)
            xf1= min(Lx-1, xf+1)
            y = yc[i,j]
            x = xc[i,j]
            Y[i,j] = (np.float32(I[yf, xf]) * (1 - y) * (1 - x) +
                      np.float32(I[yf, xf1]) * (1 - y) * x +
                      np.float32(I[yf1, xf]) * y * (1 - x) +
                      np.float32(I[yf1, xf1]) * y * x )


@njit(['int16[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]',
       'float32[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]'], parallel=True, cache=True)
def shift_coordinates(data, yup, xup, mshy, mshx, Y):
    """
    Shift data into yup and xup coordinates

    Parameters
    ----------
    data : nimg x Ly x Lx
    yup : nimg x Ly x Lx
        y shifts for each coordinate
    xup : nimg x Ly x Lx
        x shifts for each coordinate
    mshy : Ly x Lx
        meshgrid in y
    mshx : Ly x Lx
        meshgrid in x
    Y : nimg x Ly x Lx
        shifted data
    """
    for t in prange(data.shape[0]):
        map_coordinates(data[t], mshy+yup[t], mshx+xup[t], Y[t])


@njit((float32[:, :,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:], float32[:,:,:]), parallel=True, cache=True)
def block_interp(ymax1, xmax1, mshy, mshx, yup, xup):
    """
    interpolate from ymax1 to mshy to create coordinate transforms

    Parameters
    ----------
    ymax1
    xmax1
    mshy: Ly x Lx
        meshgrid in y
    mshx: Ly x Lx
        meshgrid in x
    yup: nimg x Ly x Lx
        y shifts for each coordinate
    xup: nimg x Ly x Lx
        x shifts for each coordinate
    """
    for t in prange(ymax1.shape[0]):
        map_coordinates(ymax1[t], mshy, mshx, yup[t])  # y shifts for blocks to coordinate map
        map_coordinates(xmax1[t], mshy, mshx, xup[t])  # x shifts for blocks to coordinate map


def upsample_block_shifts(Lx, Ly, nblocks, xblock, yblock, ymax1, xmax1):
    """ upsample blocks of shifts into full pixel-wise maps for shifting

    this function upsamples ymax1, xmax1 so that they are nimg x Ly x Lx
    for later bilinear interpolation
        

    Parameters
    ----------
    Lx: int
        number of pixels in the horizontal dimension
    Ly: int
        number of pixels in the vertical dimension
    nblocks: (int, int)
    xblock: float array
    yblock: float array
    ymax1: nimg x nblocks
        y shifts of blocks
    xmax1: nimg x nblocks
        y shifts of blocks

    Returns
    -------
    yup : nimg x Ly x Lx
        y shifts for each coordinate
    xup : nimg x Ly x Lx
        x shifts for each coordinate

    """
    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1)  # this recovers the coordinates of the meshgrid from (yblock, xblock)
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1)

    iy = np.interp(np.arange(Ly), yb, np.arange(yb.size)).astype(np.float32)
    ix = np.interp(np.arange(Lx), xb, np.arange(xb.size)).astype(np.float32)
    mshx, mshy = np.meshgrid(ix, iy)

    # interpolate from block centers to all points Ly x Lx
    nimg = ymax1.shape[0]
    ymax1 = ymax1.reshape(nimg, nblocks[0], nblocks[1])
    xmax1 = xmax1.reshape(nimg, nblocks[0], nblocks[1])
    yup = np.zeros((nimg, Ly, Lx), np.float32)
    xup = np.zeros((nimg, Ly, Lx), np.float32)

    block_interp(ymax1, xmax1, mshy, mshx, yup, xup)

    return yup, xup


def transform_data(data, nblocks, xblock, yblock, ymax1, xmax1, bilinear=True):
    """
    Piecewise affine transformation of data using block shifts ymax1, xmax1
    
    Parameters
    ----------

    data : nimg x Ly x Lx
    nblocks: (int, int)
    xblock: float array
    yblock: float array
    ymax1 : nimg x nblocks
        y shifts of blocks
    xmax1 : nimg x nblocks
        y shifts of blocks
    bilinear: bool (optional, default=True)
        do bilinear interpolation, if False do nearest neighbor

    Returns
    -----------
    Y : float32, nimg x Ly x Lx
        shifted data
    """
    _, Ly, Lx = data.shape
    yup, xup = upsample_block_shifts(
        Lx=Lx,
        Ly=Ly,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )
    if not bilinear:
        yup = np.round(yup)
        xup = np.round(xup)

    # use shifts and do bilinear interpolation
    mshx, mshy = np.meshgrid(np.arange(Lx, dtype=np.float32), np.arange(Ly, dtype=np.float32))
    Y = np.zeros_like(data, dtype=np.float32)
    shift_coordinates(data, yup, xup, mshy, mshx, Y)
    return Y

