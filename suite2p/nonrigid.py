import numpy as np
from numpy import fft
from scipy.fftpack import next_fast_len
from numba import vectorize,float32,int32,int16,jit,njit,prange, complex64
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.transform import warp#, PiecewiseAffineTransform
from suite2p import register
import time
import math
from mkl_fft import fft2, ifft2

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad
subpixel = 10

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

# smoothing kernel
def kernelD(a, b):
    dxs = np.reshape(a[0], (-1,1)) - np.reshape(b[0], (1,-1))
    dys = np.reshape(a[1], (-1,1)) - np.reshape(b[1], (1,-1))
    ds = np.square(dxs) + np.square(dys)
    K = np.exp(-ds/(2*np.square(sigL)))
    return K

def mat_upsample(lpad):
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

def getSNR(cc, Ls, ops):
    ''' compute SNR of phase-correlation - is it an accurate predicted shift? '''
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    X1max  = np.amax(cc0, axis = 1)
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    # set to 0 all pts +-lpad from ymax,xmax
    cc0 = cc[:, (Lyhalf-lcorr-lpad):(Lyhalf+lcorr+1+lpad), (Lxhalf-lcorr-lpad):(Lxhalf+lcorr+1+lpad)]
    for j in range(nimg):
        cc0[j,ymax[j]:ymax[j]+2*lpad, xmax[j]:xmax[j]+2*lpad] = 0
    cc0 = np.reshape(cc0, (nimg, -1))
    Xmax  = np.maximum(0, np.amax(cc0, axis = 1))
    snr = X1max / Xmax # computes snr
    return snr

def getXYup(cc, Ls, ops):
    ''' get subpixel registration shifts from phase-correlation matrix cc '''
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    ix  = np.argmax(cc0, axis = 1)
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    #X1max  = np.amax(cc0, axis = 1)
    # set to 0 all pts +-lpad from ymax,xmax
    cc0 = cc[:, (Lyhalf-lcorr-lpad):(Lyhalf+lcorr+1+lpad), (Lxhalf-lcorr-lpad):(Lxhalf+lcorr+1+lpad)].copy()
    for j in range(nimg):
        cc0[j,ymax[j]:ymax[j]+2*lpad, xmax[j]:xmax[j]+2*lpad] = 0
    cc0 = np.reshape(cc0, (nimg, -1))
    mxpt = [ymax+Lyhalf-lcorr, xmax + Lxhalf-lcorr]
    ccmat = np.zeros((nimg, 2*lpad+1, 2*lpad+1))
    for j in range(0, nimg):
        ccmat[j,:,:] = cc[j, (mxpt[0][j] -lpad):(mxpt[0][j] +lpad+1), (mxpt[1][j] -lpad):(mxpt[1][j] +lpad+1)]
    ccmat = np.reshape(ccmat, (nimg,-1))
    ccb = np.dot(ccmat, Kmat)
    imax = np.argmax(ccb, axis=1)
    cmax = np.amax(ccb, axis=1)
    ymax, xmax = np.unravel_index(imax, (nup,nup))
    mdpt = np.floor(nup/2)
    ymax,xmax = (ymax-mdpt)/subpixel, (xmax-mdpt)/subpixel
    ymax, xmax = ymax + mxpt[0] - Lyhalf, xmax + mxpt[1] - Lxhalf
    return ymax, xmax, cmax

def prepare_masks(refImg1, ops):
    refImg0=refImg1.copy()
    if ops['1Preg']:
        maskSlope    = ops['spatial_taper']
    else:
        maskSlope    = 3 * ops['smooth_sigma'] # slope of taper mask at the edges
    Ly,Lx = refImg0.shape
    maskMul = register.spatial_taper(maskSlope, Ly, Lx)

    if ops['1Preg']:
        refImg0 = register.one_photon_preprocess(refImg0[np.newaxis,:,:], ops).squeeze()

    # split refImg0 into multiple parts
    cfRefImg1 = []
    maskMul1 = []
    maskOffset1 = []
    nb = len(ops['yblock'])

    #patch taper
    Ly = ops['yblock'][0][1] - ops['yblock'][0][0]
    Lx = ops['xblock'][0][1] - ops['xblock'][0][0]
    if ops['pad_fft']:
        cfRefImg1 = np.zeros((nb,1,next_fast_len(Ly), next_fast_len(Lx)),'complex64')
    else:
        cfRefImg1 = np.zeros((nb,1,Ly,Lx),'complex64')
    maskMul1 = np.zeros((nb,1,Ly,Lx),'float32')
    maskOffset1 = np.zeros((nb,1,Ly,Lx),'float32')
    for n in range(nb):
        yind = ops['yblock'][n]
        yind = np.arange(yind[0],yind[-1]).astype('int')
        xind = ops['xblock'][n]
        xind = np.arange(xind[0],xind[-1]).astype('int')

        refImg = refImg0[np.ix_(yind,xind)]
        maskMul2 = register.spatial_taper(2 * ops['smooth_sigma'], Ly, Lx)
        maskMul1[n,0,:,:] = maskMul[np.ix_(yind,xind)].astype('float32')
        maskMul1[n,0,:,:] *= maskMul2.astype('float32')
        maskOffset1[n,0,:,:] = (refImg.mean() * (1. - maskMul1[n,0,:,:])).astype(np.float32)
        cfRefImg   = np.conj(fft.fft2(refImg))
        if ops['do_phasecorr']:
            absRef     = np.absolute(cfRefImg)
            cfRefImg   = cfRefImg / (eps0 + absRef)

        # gaussian filter
        fhg = register.gaussian_fft(ops['smooth_sigma'], cfRefImg.shape[0], cfRefImg.shape[1])
        cfRefImg *= fhg

        cfRefImg1[n,0,:,:] = (cfRefImg.astype('complex64'))
    return maskMul1, maskOffset1, cfRefImg1

@vectorize([float32(float32, float32, float32)], nopython=True, target = 'parallel')
def apply_masks(Y, maskMul, maskOffset):
    return Y*maskMul + maskOffset
@vectorize([complex64(int16, float32, float32)], nopython=True, target = 'parallel')
def addmultiply(x,y,z):
    return np.complex64(x*y + z)

def phasecorr(data, refAndMasks, ops):
    t0=tic()
    ''' loop through blocks and compute phase correlations'''
    nimg, Ly, Lx = data.shape
    maskMul    = refAndMasks[0].squeeze()
    maskOffset = refAndMasks[1].squeeze()
    cfRefImg   = refAndMasks[2].squeeze()

    LyMax = np.diff(np.array(ops['yblock']))
    ly,lx = cfRefImg.shape[-2:]
    lyhalf = int(np.floor(ly/2))
    lxhalf = int(np.floor(lx/2))

    # maximum registration shift allowed
    maxregshift = np.round(ops['maxregshiftNR'])
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(ly,lx)/2.)-lpad))
    nb = len(ops['yblock'])
    nblocks = ops['nblocks']

    # preprocessing for 1P recordings
    if ops['1Preg']:
        X = register.one_photon_preprocess(data.copy().astype(np.float32), ops)

    # shifts and corrmax
    ymax1 = np.zeros((nimg,nb),np.float32)
    cmax1 = np.zeros((nimg,nb),np.float32)
    xmax1 = np.zeros((nimg,nb),np.float32)

    cc0 = np.zeros((nimg, nb, 2*lcorr + 2*lpad + 1, 2*lcorr + 2*lpad + 1), np.float32)
    snr = np.zeros((nimg, nb), np.float32)
    ymax = np.zeros((nb,), np.int32)
    xmax = np.zeros((nb,), np.int32)

    print('fft %2.2f'%toc(t0))
    Y = np.zeros((nimg, nb, ly, lx), 'int16')
    for n in range(nb):
        yind, xind = ops['yblock'][n], ops['xblock'][n]
        Y[:,n] = data[:, yind[0]:yind[-1], xind[0]:xind[-1]]
    Y = addmultiply(Y, maskMul, maskOffset)

    for n in range(nb):
        for t in range(nimg):
            fft2(Y[t,n], overwrite_x=True)
    Y = register.apply_dotnorm(Y, cfRefImg)

    for n in range(nb):
        for t in range(nimg):
            ifft2(Y[t,n], overwrite_x=True)
            output = fft.fftshift(np.real(Y[t,n]), axes=(-2,-1))
            cc0[t,n] = output[(lyhalf-lcorr-lpad):(lyhalf+lcorr+1+lpad), (lxhalf-lcorr-lpad):(lxhalf+lcorr+1+lpad)]
            ix = np.argmax(cc0[t,n][lpad:-lpad, lpad:-lpad], axis=None)
            ym, xm = np.unravel_index(ix, (2*lcorr+1, 2*lcorr+1))
            X1max = cc0[t,n][ym+lpad, xm+lpad]
            cc0[t,n][ym:ym+2*lpad+1, xm:xm+2*lpad+1] = 0 # set to 0 all pts +-lpad from ymax,xmax
            Xmax  = np.maximum(0, np.max(cc0[t,n], axis=None))
            snr[t,n] = X1max / (1e-5 + Xmax)
    print('fft %2.2f'%toc(t0))

    for t in range(nimg):
        ccsm = np.reshape(ops['NRsm'] @ np.reshape(cc0[t], (cc0.shape[1], -1)), cc0.shape[1:])
        cc0[t][snr[t] < ops['snr_thresh']] = ccsm[snr[t] < ops['snr_thresh']]
        del ccsm
        ccmat = np.zeros((nb, 2*lpad+1, 2*lpad+1), np.float32)
        for n in range(nb):
            ix = np.argmax(cc0[t,n][lpad:-lpad, lpad:-lpad], axis=None)
            ym, xm = np.unravel_index(ix, (2*lcorr+1, 2*lcorr+1))
            ccmat[n] = cc0[t,n][ym:ym+2*lpad+1, xm:xm+2*lpad+1]
            ymax[n], xmax[n] = ym, xm
        ccmat = np.reshape(ccmat, (nb,-1))
        ccb = np.dot(ccmat, Kmat)
        imax = np.argmax(ccb, axis=1)
        cmax = np.amax(ccb, axis=1)
        ymax1[t], xmax1[t] = np.unravel_index(imax, (nup,nup))
        mdpt = np.floor(nup/2)
        ymax1[t], xmax1[t] = (ymax1[t] - mdpt)/subpixel, (xmax1[t] - mdpt)/subpixel
        ymax1[t], xmax1[t] = ymax1[t] + ymax - lyhalf, xmax1[t] + xmax - lxhalf
    print('fft %2.2f'%toc(t0))

    sig = 0
    if sig>0:
        cc1 = np.reshape(cc1,(nimg,ly,lx,nblocks[0],nblocks[1]))
        cc1 = gaussian_filter(cc1, [0,0,0,sig,sig])
        cc1 = np.reshape(cc1,(nimg,ly,lx,nb))
        for n in range(nb):
            ymax, xmax, cmax = getXYup(cc1[:,:,:,n], (lcorr,lpad, lyhalf, lxhalf), ops)
            ymax1[:,n] = ymax
            xmax1[:,n] = xmax
            cmax1[:,n] = cmax

    return ymax1, xmax1, cmax1

def linear_interp(iy, ix, yb, xb, f):
    ''' 2d interpolation of f on grid of yb, xb into grid of iy, ix '''
    ''' assumes f is 3D and last two dimensions are yb,xb '''
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

@vectorize([float32(float32,float32,float32,float32,float32,float32)], nopython=True)
def bilinear_interp(d00, d01, d10, d11, yc, xc):
    out = (d00 * (1 - yc) * (1 - xc) +
           d01 * (1 - yc) * xc +
           d10 * yc * (1 - xc) +
           d11 * yc * xc)
    return out

@njit((float32[:, :],int32[:,:],int32[:,:], float32[:,:], float32[:,:], float32[:,:], float32[:,:]))
def index_square(I, yc_floor, xc_floor, d00, d01, d10, d11):
    for i in range(I.shape[-2]):
        for j in range(I.shape[-1]):
            ycf = yc_floor[i,j]
            xcf = xc_floor[i,j]
            d00[i,j] = I[ycf, xcf]
            d01[i,j] = I[ycf, xcf+1]
            d10[i,j] = I[ycf+1, xcf]
            d11[i,j] = I[ycf+1, xcf+1]

@vectorize([int32(float32)], nopython=True)
def nfloor(y):
    return math.floor(y) #np.int32(np.floor(y))

@njit((float32[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]), parallel=True)
def map_coordinates(data, yup, xup, mshy, mshx, Y):
    ''' warp data to y and x coords (data is nimg x Ly x Lx) '''
    d00 = np.zeros_like(data[0])
    d01 = np.zeros_like(data[0])
    d10 = np.zeros_like(data[0])
    d11 = np.zeros_like(data[0])
    #yc_floor = np.zeros(data[0].shape, np.int32)
    #xc_floor = np.zeros(data[0].shape, np.int32)
    for t in prange(data.shape[0]):
        yc = mshy + yup[t]
        xc = mshx + xup[t]
        yc_floor = yc.astype(np.int32)
        xc_floor = xc.astype(np.int32)
        yc -= yc_floor
        xc -= xc_floor
        index_square(data[t], yc_floor, xc_floor, d00, d01, d10, d11)
        Y[t] = bilinear_interp(d00, d01, d10, d11, yc, xc)

@njit((float32[:, :,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:], float32[:,:,:]), parallel=True)
def block_interp(ymax1, xmax1, mshy, mshx, yup, xup):
    d00 = np.zeros_like(mshx)
    d01 = np.zeros_like(mshx)
    d10 = np.zeros_like(mshx)
    d11 = np.zeros_like(mshx)
    my_floor = mshy.astype(np.int32)
    mx_floor = mshy.astype(np.int32)
    mshy -= my_floor
    mshx -= mx_floor
    for t in prange(ymax1.shape[0]):
        index_square(ymax1[t], my_floor, mx_floor, d00, d01, d10, d11)
        yup[t] = bilinear_interp(d00, d01, d10, d11, mshy, mshx)
        index_square(xmax1[t], my_floor, mx_floor, d00, d01, d10, d11)
        xup[t] = bilinear_interp(d00, d01, d10, d11, mshy, mshx)

def upsample_block_shifts(ops, ymax1, xmax1):
    '''
        ymax1,xmax1 are shifts in Y and X of blocks of size nimg x nblocks
        this function upsamples ymax1, xmax1 so that they are nimg x Ly x Lx
        for later bilinear interpolation
        returns yup, xup <- nimg x Ly x Lx
    '''
    Ly,Lx = ops['Ly'],ops['Lx']
    nblocks = ops['nblocks']
    nimg = ymax1.shape[0]
    ymax1 = np.reshape(ymax1, (nimg,nblocks[0], nblocks[1]))
    xmax1 = np.reshape(xmax1, (nimg,nblocks[0], nblocks[1]))
    # replicate first and last row and column for padded interpolation
    ymax1 = np.pad(ymax1, ((0,0), (1,1), (1,1)), 'edge')
    xmax1 = np.pad(xmax1, ((0,0), (1,1), (1,1)), 'edge')
    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    iy = np.arange(0,Ly,1,np.float32)
    ix = np.arange(0,Lx,1,np.float32)
    yb = np.array(ops['yblock'][::ops['nblocks'][1]]).mean(axis=1).astype(np.float32)
    xb = np.array(ops['xblock'][:ops['nblocks'][1]]).mean(axis=1).astype(np.float32)
    yb = np.hstack((0, yb, Ly-1))
    xb = np.hstack((0, xb, Lx-1))

    # normalize distances for interpolation
    iy /= yb.max() * yb.shape[0]
    yb /= yb.max() * yb.shape[0]
    ix /= xb.max() * xb.shape[0]
    xb /= xb.max() * xb.shape[0]
    mshx,mshy = np.meshgrid(iy, ix)

    # interpolate from block centers to all points Ly x Lx
    yup = np.zeros((nimg,Ly,Lx), np.float32)
    xup = np.zeros((nimg,Ly,Lx), np.float32)
    block_interp(ymax1,ymax1,mshy,mshx,yup,xup)
    return yup, xup

def transform_data(data, ops, ymax1, xmax1):
    ''' piecewise affine transformation of data using block shifts ymax1, xmax1 '''
    nblocks = ops['nblocks']
    if data.ndim<3:
        data = data[np.newaxis,:,:]
    nimg,Ly,Lx = data.shape
    # take shifts and make matrices of shifts nimg x Ly x Lx
    yup,xup = upsample_block_shifts(ops, ymax1, xmax1)
    mshx,mshy = np.meshgrid(np.arange(0,Lx,1,np.float32), np.arange(0,Ly,1,np.float32))
    Y = np.zeros_like(data)
    # use shifts and do bilinear interpolation
    map_coordinates(data, yup, xup, mshy, mshx, Y)
    return Y
