import math
import warnings

import numpy as np
from numba import vectorize, float32, int32, njit, prange
from numpy import fft
from scipy.fftpack import next_fast_len

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)
from . import utils

sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad
subpixel = 10

# smoothing kernel
def kernelD(a, b):
    """ Gaussian kernel from a to b """
    dxs = np.reshape(a[0], (-1,1)) - np.reshape(b[0], (1,-1))
    dys = np.reshape(a[1], (-1,1)) - np.reshape(b[1], (1,-1))
    ds = np.square(dxs) + np.square(dys)
    K = np.exp(-ds/(2*np.square(sigL)))
    return K

def mat_upsample(lpad):
    """ upsampling matrix using gaussian kernels """
    lar    = np.arange(-lpad, lpad+1)
    larUP  = np.arange(-lpad, lpad+.001, 1./subpixel)
    x, y   = np.meshgrid(lar, lar)
    xU, yU = np.meshgrid(larUP, larUP)
    Kx = kernelD((x,y),(x,y))
    Kx = np.linalg.inv(Kx)
    Kg = kernelD((x,y),(xU,yU))
    Kmat = np.dot(Kx, Kg)
    nup = larUP.shape[0]
    return Kmat, nup

Kmat, nup = mat_upsample(lpad)

def make_blocks(Ly, Lx, maxregshiftNR=5, block_size=(128, 128)):
    """ computes overlapping blocks to split FOV into to register separately"""
    ny = int(np.ceil(1.5 * float(Ly) / block_size[0]))
    nx = int(np.ceil(1.5 * float(Lx) / block_size[1]))

    if block_size[0] >= Ly:
        block_size[0] = Ly
        ny = 1
    if block_size[1] >= Lx:
        block_size[1] = Lx
        nx = 1
    nblocks = [ny, nx]

    ystart = np.linspace(0, Ly - block_size[0], ny).astype('int')
    xstart = np.linspace(0, Lx - block_size[1], nx).astype('int')
    yblock = []
    xblock = []
    for iy in range(ny):
        for ix in range(nx):
            yind = np.array([ystart[iy], ystart[iy] + block_size[0]])
            xind = np.array([xstart[ix], xstart[ix] + block_size[1]])
            yblock.append(yind)
            xblock.append(xind)

    ys, xs = np.meshgrid(np.arange(nx), np.arange(ny))
    ys = ys.flatten()
    xs = xs.flatten()
    ds = (ys - ys[:, np.newaxis]) ** 2 + (xs - xs[:, np.newaxis]) ** 2
    R = np.exp(-ds)
    R = R / np.sum(R, axis=0)
    NRsm = R.T

    return yblock, xblock, nblocks, maxregshiftNR, block_size, NRsm


def phasecorr_reference(refImg, reg_1p, maskSlope, smooth_sigma, spatial_hp, pre_smooth, yblock, xblock, pad_fft):
    """ computes taper and fft'ed reference image for phasecorr
    
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

    refImg0 = refImg.copy()
    Ly, Lx = refImg0.shape
    maskMul = utils.spatial_taper(maskSlope, Ly, Lx)

    if reg_1p:
        refImg0 = utils.one_photon_preprocess(data=refImg0[np.newaxis, :, :], spatial_hp=spatial_hp, pre_smooth=pre_smooth)

    # split refImg0 into multiple parts
    nb = len(yblock)

    #patch taper
    Ly = yblock[0][1] - yblock[0][0]
    Lx = xblock[0][1] - xblock[0][0]
    cfRefImg1 = np.zeros((nb,1,next_fast_len(Ly), next_fast_len(Lx)), 'complex64') if pad_fft else np.zeros((nb, 1, Ly, Lx), 'complex64')
    maskMul1 = np.zeros((nb,1,Ly,Lx),'float32')
    maskOffset1 = np.zeros((nb,1,Ly,Lx),'float32')
    for n in range(nb):
        yind = yblock[n]
        yind = np.arange(yind[0], yind[-1]).astype('int')
        xind = xblock[n]
        xind = np.arange(xind[0], xind[-1]).astype('int')

        refImg = refImg0.squeeze()[np.ix_(yind,xind)]
        maskMul2 = utils.spatial_taper(2 * smooth_sigma, Ly, Lx)
        maskMul1[n, 0, :, :] = maskMul[np.ix_(yind,xind)].astype('float32')
        maskMul1[n, 0, :, :] *= maskMul2.astype('float32')
        maskOffset1[n, 0, :, :] = (refImg.mean() * (1. - maskMul1[n, 0, :, :])).astype(np.float32)
        cfRefImg = np.conj(fft.fft2(refImg))
        absRef = np.absolute(cfRefImg)
        cfRefImg = cfRefImg / (1e-5 + absRef)

        # gaussian filter
        fhg = utils.gaussian_fft(smooth_sigma, cfRefImg.shape[0], cfRefImg.shape[1])
        cfRefImg *= fhg
        cfRefImg1[n, 0, :, :] = cfRefImg.astype('complex64')

    return maskMul1, maskOffset1, cfRefImg1

@vectorize([float32(float32, float32, float32)], nopython=True, target = 'parallel', cache=True)
def apply_masks(Y, maskMul, maskOffset):
    return Y*maskMul + maskOffset
@vectorize(['complex64(int16, float32, float32)', 'complex64(float32, float32, float32)'], nopython=True, target = 'parallel', cache=True)
def addmultiply(x,y,z):
    return np.complex64(x*y + z)

def getSNR(cc, Ls):
    """ compute SNR of phase-correlation - is it an accurate predicted shift? """
    (lcorr, lpad) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, lpad:-lpad, lpad:-lpad]
    cc0 = np.reshape(cc0, (nimg, -1))
    X1max  = np.amax(cc0, axis = 1)
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    # set to 0 all pts +-lpad from ymax,xmax
    cc0 = cc.copy()
    for j in range(nimg):
        cc0[j,ymax[j]:ymax[j]+2*lpad, xmax[j]:xmax[j]+2*lpad] = 0
    cc0 = np.reshape(cc0, (nimg, -1))
    Xmax  = np.maximum(0, np.amax(cc0, axis = 1))
    snr = X1max / Xmax # computes snr
    return snr


def clip(X, lhalf):
    x00 = X[:, :, :lhalf+1, :lhalf+1]
    x11 = X[:, :, -lhalf:, -lhalf:]
    x01 = X[:, :, :lhalf+1, -lhalf:]
    x10 = X[:, :, -lhalf:, :lhalf+1]
    return x00, x01, x10, x11


def phasecorr(data, refAndMasks, snr_thresh, NRsm, xblock, yblock, maxregshiftNR):
    """ compute phase correlations for each block 
    
    Parameters
    -------------

    data : int16 or float32, 3D array
        size [nimg x Ly x Lx]

    refAndMasks : list
        gaussian filter, mask offset, FFT of reference image

    ymax1 : 2D array
        size [nimg x nblocks], y shifts of blocks

    xmax1 : 2D array
        size [nimg x nblocks], y shifts of blocks

    cmax1 : 2D array
        size [nimg x nblocks], value of peak of phase correlation

    ccsm : 4D array
        size [nimg x nblocks x ly x lx], smoothed phase correlations


    """

    nimg, Ly, Lx = data.shape
    maskMul = refAndMasks[0].squeeze()
    maskOffset = refAndMasks[1].squeeze()
    cfRefImg = refAndMasks[2].squeeze()

    ly, lx = cfRefImg.shape[-2:]

    # maximum registration shift allowed
    maxregshift = np.round(maxregshiftNR)
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(ly, lx) / 2.) - lpad))
    nb = len(yblock)

    # shifts and corrmax
    ymax1 = np.zeros((nimg, nb), np.float32)
    cmax1 = np.zeros((nimg, nb), np.float32)
    xmax1 = np.zeros((nimg, nb), np.float32)

    cc0 = np.zeros((nimg, nb, 2*lcorr + 2*lpad + 1, 2*lcorr + 2*lpad + 1), np.float32)
    ymax = np.zeros((nb,), np.int32)
    xmax = np.zeros((nb,), np.int32)

    Y = np.zeros((nimg, nb, ly, lx), 'int16')
    for n in range(nb):
        yind, xind = yblock[n], xblock[n]
        Y[:,n] = data[:, yind[0]:yind[-1], xind[0]:xind[-1]]
    Y = addmultiply(Y, maskMul, maskOffset)
    for n in range(nb):
        for t in range(nimg):
            fft2(Y[t,n], overwrite_x=True)
    Y = utils.apply_dotnorm(Y, cfRefImg)
    for n in range(nb):
        for t in range(nimg):
            ifft2(Y[t,n], overwrite_x=True)
    x00, x01, x10, x11 = clip(Y, lcorr+lpad)
    cc0 = np.real(np.block([[x11, x10], [x01, x00]]))
    cc0 = np.transpose(cc0, (1,0,2,3))
    cc0 = cc0.reshape((cc0.shape[0], -1))
    cc2 = []
    R = NRsm
    cc2.append(cc0)
    for j in range(2):
        cc2.append(R @ cc2[j])
    for j in range(len(cc2)):
        cc2[j] = cc2[j].reshape((nb, nimg, 2*lcorr+2*lpad+1, 2*lcorr+2*lpad+1))
    ccsm = cc2[0]
    for n in range(nb):
        snr = np.ones((nimg,), 'float32')
        for j in range(len(cc2)):
            ism = snr < snr_thresh
            if np.sum(ism)==0:
                break
            cc = cc2[j][n,ism,:,:]
            if j>0:
                ccsm[n, ism, :, :] = cc
            snr[ism] = getSNR(cc, (lcorr,lpad))

    ccmat = np.zeros((nb, 2*lpad+1, 2*lpad+1), np.float32)
    for t in range(nimg):
        ccmat = np.zeros((nb, 2*lpad+1, 2*lpad+1), np.float32)
        for n in range(nb):
            ix = np.argmax(ccsm[n, t][lpad:-lpad, lpad:-lpad], axis=None)
            ym, xm = np.unravel_index(ix, (2*lcorr+1, 2*lcorr+1))
            ccmat[n] = ccsm[n,t][ym:ym+2*lpad+1, xm:xm+2*lpad+1]
            ymax[n], xmax[n] = ym-lcorr, xm-lcorr
        ccmat = np.reshape(ccmat, (nb,-1))
        ccb = np.dot(ccmat, Kmat)
        imax = np.argmax(ccb, axis=1)
        cmax = np.amax(ccb, axis=1)
        ymax1[t], xmax1[t] = np.unravel_index(imax, (nup,nup))
        cmax1[t] = cmax
        mdpt = np.floor(nup/2)
        ymax1[t], xmax1[t] = (ymax1[t] - mdpt)/subpixel, (xmax1[t] - mdpt)/subpixel
        ymax1[t], xmax1[t] = ymax1[t] + ymax, xmax1[t] + xmax
    #ccmat = np.reshape(ccmat, (nb, 2*lpad+1, 2*lpad+1))
    return ymax1, xmax1, cmax1, ccsm

def linear_interp(iy, ix, yb, xb, f):
    """ 2d interpolation of f on grid of yb, xb into grid of iy, ix 
        assumes f is 3D and last two dimensions are yb,xb """
    fup = f.copy().astype(np.float32)
    Lax = [iy.size, ix.size]
    for n in range(2):
        fup = np.transpose(fup,(1,2,0)).copy()
        if n==0:
            ds  = np.abs(iy[:,np.newaxis] - yb[:,np.newaxis].T)
        else:
            ds  = np.abs(ix[:,np.newaxis] - xb[:,np.newaxis].T)
        im1 = np.argmin(ds, axis=1)
        w1  = ds[np.arange(0,Lax[n],1,int),im1]
        ds[np.arange(0,Lax[n],1,int),im1] = np.inf
        im2 = np.argmin(ds, axis=1)
        w2  = ds[np.arange(0,Lax[n],1,int),im2]
        wnorm = w1+w2
        w1 /= wnorm
        w2 /= wnorm
        fup = (1-w1[:,np.newaxis,np.newaxis]) * fup[im1] + (1-w2[:,np.newaxis,np.newaxis]) * fup[im2]
    fup = np.transpose(fup, (1,2,0))
    return fup

@njit(['(int16[:, :],float32[:,:], float32[:,:], float32[:,:])', 
        '(float32[:, :],float32[:,:], float32[:,:], float32[:,:])'], cache=True)
def map_coordinates(I, yc, xc, Y):
    """ bilinear transform of image with ycoordinates yc and xcoordinates xc to Y 
    
    Parameters
    -------------

    I : int16 or float32, 2D array
        size [Ly x Lx]     

    yc : 2D array
        size [Ly x Lx], new y coordinates

    xc : 2D array
        size [Ly x Lx], new x coordinates

    Returns
    -----------

    Y : float32, 2D array
        size [Ly x Lx], shifted I


    """
    Ly,Lx = I.shape
    yc_floor = yc.copy().astype(np.int32)
    xc_floor = xc.copy().astype(np.int32)
    yc -= yc_floor
    xc -= xc_floor
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

@vectorize([int32(float32)], nopython=True, cache=True)
def nfloor(y):
    return math.floor(y) #np.int32(np.floor(y))

@njit(['int16[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]',
       'float32[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]'], parallel=True, cache=True)
def shift_coordinates(data, yup, xup, mshy, mshx, Y):
    """ shift data into yup and xup coordinates

    Parameters
    -------------

    data : int16 or float32, 3D array
        size [nimg x Ly x Lx]     

    yup : 3D array
        size [nimg x Ly x Lx], y shifts for each coordinate

    xup : 3D array
        size [nimg x Ly x Lx], x shifts for each coordinate

    mshy : 2D array
        size [Ly x Lx], meshgrid in y

    mshx : 2D array
        size [Ly x Lx], meshgrid in x
        
    Returns
    -----------
    Y : float32, 3D array
        size [nimg x Ly x Lx], shifted data

    """
    Ly,Lx = data.shape[1:]
    for t in prange(data.shape[0]):
        map_coordinates(data[t], mshy+yup[t], mshx+xup[t], Y[t])

@njit((float32[:, :,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:], float32[:,:,:]), parallel=True, cache=True)
def block_interp(ymax1, xmax1, mshy, mshx, yup, xup):
    """ interpolate from ymax1 to mshy to create coordinate transforms """
    for t in prange(ymax1.shape[0]):
        # y shifts for blocks to coordinate map
        map_coordinates(ymax1[t], mshy.copy(), mshx.copy(), yup[t])
        # x shifts for blocks to coordinate map
        map_coordinates(xmax1[t], mshy.copy(), mshx.copy(), xup[t])

def upsample_block_shifts(Lx, Ly, nblocks, xblock, yblock, ymax1, xmax1):
    """ upsample blocks of shifts into full pixel-wise maps for shifting

    this function upsamples ymax1, xmax1 so that they are nimg x Ly x Lx
    for later bilinear interpolation
        

    Parameters
    ------------

    ymax1 : 2D array
        size [nimg x nblocks], y shifts of blocks

    xmax1 : 2D array
        size [nimg x nblocks], y shifts of blocks
    
    Returns
    -----------

    yup : 3D array
        size [nimg x Ly x Lx], y shifts for each coordinate

    xup : 3D array
        size [nimg x Ly x Lx], x shifts for each coordinate

    """

    nimg = ymax1.shape[0]
    ymax1 = np.reshape(ymax1, (nimg,nblocks[0], nblocks[1]))
    xmax1 = np.reshape(xmax1, (nimg,nblocks[0], nblocks[1]))
    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1).astype(np.float32)
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1).astype(np.float32)

    iy = np.arange(0, Ly, 1, np.float32)
    ix = np.arange(0, Lx, 1, np.float32)
    iy = np.interp(iy, yb, np.arange(0, yb.size, 1, int)).astype(np.float32)
    ix = np.interp(ix, xb, np.arange(0, xb.size, 1, int)).astype(np.float32)
    mshx, mshy = np.meshgrid(ix, iy)
    # interpolate from block centers to all points Ly x Lx
    #Ly,Lx = mshy.shape
    yup = np.zeros((nimg, Ly, Lx), np.float32)
    xup = np.zeros((nimg, Ly, Lx), np.float32)

    block_interp(ymax1, xmax1, mshy, mshx, yup, xup)
    return yup, xup


def transform_data(data, nblocks, xblock, yblock, ymax1, xmax1):
    """ piecewise affine transformation of data using block shifts ymax1, xmax1 
    
    Parameters
    -------------

    data : int16 or float32, 3D array
        size [nimg x Ly x Lx]

    ymax1 : 2D array
        size [nimg x nblocks], y shifts of blocks

    xmax1 : 2D array
        size [nimg x nblocks], y shifts of blocks

    Returns
    -----------
    Y : float32, 3D array
        size [nimg x Ly x Lx], shifted data

    """
    if data.ndim<3:
        data = data[np.newaxis, :, :]
    nimg, Ly, Lx = data.shape
    # take shifts and make matrices of shifts nimg x Ly x Lx
    yup, xup = upsample_block_shifts(
        Lx=Lx,
        Ly=Ly,
        nblocks=nblocks,
        xblock=xblock,
        yblock=yblock,
        ymax1=ymax1,
        xmax1=xmax1,
    )
    mshx, mshy = np.meshgrid(np.arange(0, Lx, 1, np.float32), np.arange(0, Ly, 1, np.float32))
    Y = np.zeros(data.shape, np.float32)
    # use shifts and do bilinear interpolation
    shift_coordinates(data, yup, xup, mshy, mshx, Y)
    return Y
