import numpy as np
import scipy.fftpack as fft
from scipy.fftpack import next_fast_len
#from numpy import fft
from scipy.ndimage import gaussian_filter
from skimage.transform import warp#, PiecewiseAffineTransform
from suite2p import register
import time
import multiprocessing
from multiprocessing import Pool
N_threads = int(multiprocessing.cpu_count() / 2)
import numexpr3 as ne3
ne3.set_nthreads(N_threads)

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

def phasecorr(data, refAndMasks, ops):
    ''' loop through blocks and compute phase correlations'''
    nimg, Ly, Lx = data.shape
    maskMul    = refAndMasks[0]
    maskOffset = refAndMasks[1]
    cfRefImg   = refAndMasks[2].squeeze()
    t0=tic()
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
        data1 = register.one_photon_preprocess(data.copy().astype(np.float32), ops)
    else:
        data1 = data.astype(np.float32)

    # shifts and corrmax
    ymax1 = np.zeros((nimg,nb),np.float32)
    cmax1 = np.zeros((nimg,nb),np.float32)
    xmax1 = np.zeros((nimg,nb),np.float32)

    # placeholder variables for numpexpr
    xfft = np.empty_like(cfRefImg[0])
    epsm = np.empty_like(cfRefImg[0])
    cfRefImg0 = np.empty_like(cfRefImg[0])
    x = np.empty((ly,lx), np.float32)
    maskMul0 = np.empty((ly,lx), np.float32)
    maskOffset0 = np.empty((ly,lx), np.float32)

    # compute phase-correlation of blocks
    xcorr2 = ne3.NumExpr( 'xfft=xfft*cfRefImg0/(epsm + abs(xfft*cfRefImg0))' )
    # mask for X to set edges to zero (especially useful in 1P)
    xmask = ne3.NumExpr( 'x = x*maskMul0 + maskOffset0' )
    epsm = eps0

    cc0 = np.zeros((nb, 2*lcorr + 2*lpad + 1, 2*lcorr + 2*lpad + 1), np.float32)
    snr = np.zeros((nb,), np.float32)
    ymax = np.zeros((nb,), np.int32)
    xmax = np.zeros((nb,), np.int32)
    for t in range(nimg):
        for n in range(nb):
            yind = ops['yblock'][n]
            yind = np.arange(yind[0],yind[-1]).astype(int)
            xind = ops['xblock'][n]
            xind = np.arange(xind[0],xind[-1]).astype(int)
            x = data1[t][np.ix_(yind, xind)]
            #xmask(x=x, maskMul0=maskMul[n], maskOffset0=maskOffset[n])
            xfft = register.fft2(x)
            xcorr2( xfft=xfft, cfRefImg0=cfRefImg[n], epsm=epsm)
            output = np.real(register.ifft2( xfft ))
            output = fft.fftshift(output, axes=(-2,-1))
            cc = output[np.ix_(np.arange(lyhalf-lcorr-lpad,lyhalf+lcorr+1+lpad,1,int),
                            np.arange(lxhalf-lcorr-lpad,lxhalf+lcorr+1+lpad,1,int))]
            cc0[n] = cc.copy()

            # compute SNR
            ix = np.argmax(cc[np.ix_(np.arange(lpad, cc.shape[-2]-lpad, 1, int),
                                     np.arange(lpad, cc.shape[-1]-lpad, 1, int))], axis=None)
            ym, xm = np.unravel_index(ix, (2*lcorr+1, 2*lcorr+1))
            X1max = cc[ym+lpad, xm+lpad]
            # set to 0 all pts +-lpad from ymax,xmax
            cc[np.ix_(np.arange(ym, ym+2*lpad+1, 1, int), np.arange(xm, xm+2*lpad+1, 1, int))] = 0
            Xmax  = np.maximum(0, np.max(cc, axis=None))
            snr[n] = X1max / Xmax
        ccsm = np.reshape(ops['NRsm'] @ np.reshape(cc0, (cc0.shape[0], -1)), cc0.shape)
        cc0[snr < ops['snr_thresh']] = ccsm[snr < ops['snr_thresh']]
        del ccsm
        ccmat = np.zeros((nb, 2*lpad+1, 2*lpad+1), np.float32)
        for n in range(nb):
            ix = np.argmax(cc0[n][np.ix_(np.arange(lpad, cc.shape[-2]-lpad, 1, int),
                                     np.arange(lpad, cc.shape[-1]-lpad, 1, int))], axis=None)
            ym, xm = np.unravel_index(ix, (2*lcorr+1, 2*lcorr+1))
            ccmat[n] = cc0[n][np.ix_(np.arange(ym, ym+2*lpad+1, 1, int), np.arange(xm, xm+2*lpad+1, 1, int))]
            ymax[n], xmax[n] = ym, xm
        ccmat = np.reshape(ccmat, (nb,-1))
        ccb = np.dot(ccmat, Kmat)
        imax = np.argmax(ccb, axis=1)
        cmax = np.amax(ccb, axis=1)
        ymax1[t], xmax1[t] = np.unravel_index(imax, (nup,nup))
        mdpt = np.floor(nup/2)
        ymax1[t], xmax1[t] = (ymax1[t] - mdpt)/subpixel, (xmax1[t] - mdpt)/subpixel
        ymax1[t], xmax1[t] = ymax1[t] + ymax - lyhalf, xmax1[t] + xmax - lxhalf
    del data1
    print('fft %2.2f'%toc(t0))
    #cc0 = cc0.reshape((cc0.shape[0], -1))
    #cc2 = []
    #R = ops['NRsm']
    #cc2.append(cc0)
    #for j in range(2):
    #    cc2.append(R @ cc2[j])
    #for j in range(len(cc2)):
    #    cc2[j] = cc2[j].reshape((nb, nimg, 2*lcorr+2*lpad+1, 2*lcorr+2*lpad+1))
    #ccsm = cc2[0]
    #for n in range(nb):
    #    snr = np.ones((nimg,), 'float32')
    #    for j in range(len(cc2)):
    #        ism = snr<ops['snr_thresh']
    #        if np.sum(ism)==0:
    #            break
    #        cc = cc2[j][n,ism,:,:]
    #        if j>0:
    #            ccsm[n, ism, :, :] = cc
    #        snr[ism] = getSNR(cc, (lcorr,lpad, lcorr+lpad, lcorr+lpad), ops)
    #print('snr %2.2f'%toc(t0))
    # smooth cc across blocks if sig>0
    # currently not used
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

def shift_data(data, ops, ymax1, xmax1):
    ''' split into workers across frames '''
    ops['num_workers'] = 0
    if ops['num_workers']<0:
        dreg = shift_data_worker((data, ops, ymax1, xmax1))
    else:
        if ops['num_workers']<1:
            ops['num_workers'] = int(multiprocessing.cpu_count()/2)
        num_cores = ops['num_workers']
        nimg = data.shape[0]
        nbatch = int(np.ceil(nimg/float(num_cores)))
        #nbatch = 50
        inputs = np.arange(0, nimg, nbatch)
        irange = []
        dsplit = []
        for i in inputs:
            ilist = i + np.arange(0,np.minimum(nbatch, nimg-i))
            irange.append(i + np.arange(0,np.minimum(nbatch, nimg-i)))
            dsplit.append([data[ilist,:, :], ops, ymax1[ilist,:], xmax1[ilist,:]])
        with Pool(num_cores) as p:
            results = p.map(shift_data_worker, dsplit)
        dreg = np.zeros_like(data)
        for i in range(0,len(results)):
            dreg[irange[i], :, :] = results[i]
    return dreg

def shift_data_worker(inputs):
    ''' piecewise affine transformation of data using shifts from phasecorr_worker '''
    data,ops,ymax1,xmax1 = inputs
    nblocks = ops['nblocks']
    if data.ndim<3:
        data = data[np.newaxis,:,:]
    nimg,Ly,Lx = data.shape
    ymax1 = np.reshape(ymax1, (nimg,nblocks[0], nblocks[1]))
    xmax1 = np.reshape(xmax1, (nimg,nblocks[0], nblocks[1]))
    # replicate first and last row and column for padded interpolation
    ymax1 = np.pad(ymax1, ((0,0), (1,1), (1,1)), 'edge')
    xmax1 = np.pad(xmax1, ((0,0), (1,1), (1,1)), 'edge')

    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    iy = np.arange(0,Ly,1,int)
    ix = np.arange(0,Lx,1,int)
    yb = np.array(ops['yblock'][::ops['nblocks'][1]]).mean(axis=1).astype(np.float32)
    xb = np.array(ops['xblock'][:ops['nblocks'][1]]).mean(axis=1).astype(np.float32)
    yb = np.hstack((0, yb, Ly-1))
    xb = np.hstack((0, xb, Lx-1))
    yup = linear_interp(iy, ix, yb, xb, ymax1)
    xup = linear_interp(iy, ix, yb, xb, xmax1)
    mshx,mshy = np.meshgrid(np.arange(0,Lx), np.arange(0,Ly))
    Y = np.zeros((nimg,Ly,Lx), np.float32)
    for t in range(nimg):
        ycoor = (mshy + yup[t])#.flatten()
        xcoor = (mshx + xup[t])#.flatten()
        coords = np.concatenate((ycoor[np.newaxis,:], xcoor[np.newaxis,:]))
        Y[t] = warp(data[t],coords, order=1, clip=False, preserve_range=True)
    return Y
