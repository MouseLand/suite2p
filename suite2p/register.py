
import glob, h5py, time, os, shutil
import numpy as np
#from numpy import fft
from scipy.fftpack import next_fast_len
import scipy.fftpack as fft
#import mkl_fft as fft
from numpy import random as rnd
import multiprocessing
from multiprocessing import Pool
import math
from scipy.signal import medfilt
from scipy.ndimage import laplace
#try:
#    import pyfftw
#    HAS_FFTW=True
#except ImportError:
#    HAS_FFTW=False
HAS_FFTW=False

try:
    #os.environ["MKL_NUM_THREADS"] = "1"
    from skimage import io
    import mkl_fft
    HAS_MKL=True
except ImportError:
    HAS_MKL=False
#HAS_MKL=False

from suite2p import nonrigid, utils, regmetrics

def fft2(data, s=None):
    if s==None:
        s=(data.shape[-2], data.shape[-1])
    if HAS_FFTW:
        x = pyfftw.empty_aligned(data.shape, dtype=np.float32)
        x[:] = data
        fft_object = pyfftw.builders.fftn(x, s=s, axes=(-2,-1),threads=2)
        data = fft_object()
    elif HAS_MKL:
        data = mkl_fft.fft2(data,shape=s,axes=(-2,-1))
    else:
        data = fft.fft2(data, s, axes=(-2,-1))
    return data

def ifft2(data, s=None):
    if s==None:
        s=(data.shape[-2], data.shape[-1])
    if HAS_FFTW:
        x = pyfftw.empty_aligned(data.shape, dtype=np.complex64)
        x[:] = data
        fft_object = pyfftw.builders.ifftn(data, s=s, axes=(-2,-1),threads=2)
        data = fft_object()
    elif HAS_MKL:
        data = mkl_fft.ifft2(data, shape=s, axes=(-2,-1))
    else:
        data = fft.ifft2(data, s, axes=(-2,-1))
    return data


def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad
hp = 60
subpixel = 10


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

def gaussian_fft(sig, Ly, Lx):
    ''' gaussian filter in the fft domain with std sig and size Ly,Lx '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg

def spatial_taper(sig, Ly, Lx):
    ''' spatial taper  on edges with gaussian of std sig '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*sig
    mX = x.max() - 2*sig
    maskY = 1./(1.+np.exp((yy-mY)/sig))
    maskX = 1./(1.+np.exp((xx-mX)/sig))
    maskMul = maskY * maskX
    return maskMul

def spatial_smooth(data,N):
    ''' spatially smooth data using cumsum over axis=1,2 with window N'''
    pad = np.zeros((data.shape[0], int(N/2), data.shape[2]))
    dsmooth = np.concatenate((pad, data, pad), axis=1)
    pad = np.zeros((dsmooth.shape[0], dsmooth.shape[1], int(N/2)))
    dsmooth = np.concatenate((pad, dsmooth, pad), axis=2)
    # in X
    cumsum = np.cumsum(dsmooth, axis=1)
    dsmooth = (cumsum[:, N:, :] - cumsum[:, :-N, :]) / float(N)
    # in Y
    cumsum = np.cumsum(dsmooth, axis=2)
    dsmooth = (cumsum[:, :, N:] - cumsum[:, :, :-N]) / float(N)
    return dsmooth

def spatial_high_pass(data, N):
    ''' high pass filters data over axis=1,2 with window N'''
    norm = spatial_smooth(np.ones((1, data.shape[1], data.shape[2])), N).squeeze()
    data -= spatial_smooth(data, N) / norm
    return data

def one_photon_preprocess(data, ops):
    ''' pre filtering for one-photon data '''
    if ops['pre_smooth'] > 0:
        ops['pre_smooth'] = int(np.ceil(ops['pre_smooth']/2) * 2)
        data = spatial_smooth(data, ops['pre_smooth'])

    #for n in range(data.shape[0]):
    #    data[n,:,:] = laplace(data[n,:,:])
    ops['spatial_hp'] = int(np.ceil(ops['spatial_hp']/2) * 2)
    data = spatial_high_pass(data, ops['spatial_hp'])
    return data

def prepare_masks(refImg0, ops):
    refImg = refImg0.copy()
    if ops['1Preg']:
        maskSlope    = ops['spatial_taper'] # slope of taper mask at the edges
    else:
        maskSlope    = 3 * ops['smooth_sigma'] # slope of taper mask at the edges
    Ly,Lx = refImg.shape
    maskMul = spatial_taper(maskSlope, Ly, Lx)

    if ops['1Preg']:
        refImg = one_photon_preprocess(refImg[np.newaxis,:,:], ops).squeeze()
    maskOffset = refImg.mean() * (1. - maskMul);

    # reference image in fourier domain
    if ops['pad_fft']:
        cfRefImg   = np.conj(fft.fft2(refImg,
                            [next_fast_len(ops['Ly']), next_fast_len(ops['Lx'])]))
    else:
        cfRefImg   = np.conj(fft.fft2(refImg))

    if ops['do_phasecorr']:
        absRef     = np.absolute(cfRefImg);
        cfRefImg   = cfRefImg / (eps0 + absRef)

    # gaussian filter in space
    fhg = gaussian_fft(ops['smooth_sigma'], cfRefImg.shape[0], cfRefImg.shape[1])
    cfRefImg *= fhg

    maskMul = maskMul.astype('float32')
    maskOffset = maskOffset.astype('float32')
    cfRefImg = cfRefImg.astype('complex64')
    cfRefImg = np.reshape(cfRefImg, (1, cfRefImg.shape[0], cfRefImg.shape[1]))
    return maskMul, maskOffset, cfRefImg

def correlation_map(X, refAndMasks, do_phasecorr):
    maskMul    = refAndMasks[0]
    maskOffset = refAndMasks[1]
    cfRefImg   = refAndMasks[2]
    #nimg, Ly, Lx = X.shape
    X = X * maskMul + maskOffset
    X = fft2(X, [cfRefImg.shape[-2], cfRefImg.shape[-1]])
    if do_phasecorr:
        X = X / (eps0 + np.absolute(X))
    X *= cfRefImg
    cc = np.real(ifft2(X))
    cc = fft.fftshift(cc, axes=(-2,-1))
    return cc

def getSNR(cc, Ls, ops):
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

def shift_data_subpixel(inputs):
    ''' rigid shift of X by ymax and xmax '''
    ''' no longer used '''    
    X, ymax, xmax = inputs
    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]

    nimg, Ly0, Lx0 = X.shape
    X = fft2(X.astype('float32'), [next_fast_len(Ly0), next_fast_len(Lx0)])
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

def shift_data(inputs):
    ''' rigid shift of X by ymax and xmax '''
    X, ymax, xmax, m0 = inputs
    ymax = ymax.flatten()
    xmax = xmax.flatten()
    if X.ndim<3:
        X = X[np.newaxis,:,:]

    nimg, Ly, Lx = X.shape

    for n in range(nimg):
        X[n] = np.roll(X[n], (-ymax[n], -xmax[n]), axis=(0,1))
        yrange = np.arange(0, Ly,1,int) + ymax[n]
        xrange = np.arange(0, Lx,1,int) + xmax[n]
        yrange = yrange[np.logical_or(yrange<0, yrange>Ly-1)] - ymax[n]
        xrange = xrange[np.logical_or(xrange<0, xrange>Lx-1)] - xmax[n]
        X[n][yrange, :] = m0
        X[n][:, xrange] = m0
    return X


def getCCmax(cc, lcorr):
    Lyhalf = int(np.floor(cc.shape[1]/2))
    Lxhalf = int(np.floor(cc.shape[2]/2))
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]
    cc0 = np.reshape(cc0, (nimg, -1))
    ix  = np.argmax(cc0, axis = 1)
    cmax = cc0[np.arange(0,nimg,1,int), ix]
    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    ymax, xmax = ymax-lcorr, xmax-lcorr
    return ymax,xmax,cmax

def phasecorr_worker(inputs):
    ''' compute registration offsets and shift data '''
    data, refAndMasks, ops = inputs
    k=tic()
    if ops['nonrigid'] and len(refAndMasks)>3:
        refAndMasksNR = refAndMasks[3:]
        refAndMasks = refAndMasks[:3]
        nr = True
    else:
        nr = False
    nimg, Ly, Lx = data.shape
    k=tic()
    maxregshift = np.round(ops['maxregshift'] *np.maximum(Ly, Lx))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly,Lx)/2.)-lpad))
    if ops['1Preg']:
        data1 = data.copy().astype(np.float32)
        data1 = one_photon_preprocess(data1, ops)
        cc = correlation_map(data1, refAndMasks, ops['do_phasecorr'])
        del data1
    else:
        cc = correlation_map(data, refAndMasks, ops['do_phasecorr'])
    #print(toc(k))
    # get ymax,xmax, cmax not upsampled
    ymax, xmax, cmax = getCCmax(cc, lcorr)
    #print(toc(k))
    Y = shift_data((data, ymax, xmax, ops['refImg'].mean()))
    #Y = data
    #print(toc(k))
    yxnr = []
    if nr:
        Y, ymax1, xmax1, cmax1 = nonrigid.phasecorr_worker((Y, refAndMasksNR, ops))
        #Y = nonrigid.shift_data((Y,ymax1,xmax1,ops))
        yxnr = [ymax1,xmax1,cmax1]
    #print(toc(k))
    return Y, ymax, xmax, cmax, yxnr

def register_data(data, refAndMasks, ops):
    ''' register data matrix to reference image and shift '''
    ''' need reference image ops['refImg']'''
    ''' run refAndMasks = prepare_refAndMasks(ops) to get fft'ed masks '''
    ''' calls phasecorr_worker '''

    if ops['bidiphase']!=0:
        data = shift_bidiphase(data.copy(), ops['bidiphase'])

    nr=False
    yxnr = []
    if ops['nonrigid'] and len(refAndMasks)>3:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        nr=True
    if ops['num_workers']<0:
        Y, ymax, xmax, cmax, yxnr = phasecorr_worker((data, refAndMasks, ops))
    else:
        # run phasecorr_worker over multiple cores
        nimg = data.shape[0]
        if ops['num_workers']<1:
            ops['num_workers'] = int(multiprocessing.cpu_count()/2)
        num_cores = ops['num_workers']

        nbatch = int(np.ceil(nimg/float(num_cores)))
        #nbatch = 50
        inputs = np.arange(0, nimg, nbatch)
        irange = []
        dsplit = []
        for i in inputs:
            ilist = i + np.arange(0,np.minimum(nbatch, nimg-i))
            irange.append(ilist)
            dsplit.append([data[ilist,:, :], refAndMasks, ops])

        with Pool(num_cores) as p:
            results = p.map(phasecorr_worker, dsplit)
        Y = np.zeros_like(data)
        ymax = np.zeros((nimg,))
        xmax = np.zeros((nimg,))
        cmax = np.zeros((nimg,))
        if nr:
            ymax1 = np.zeros((nimg,nb))
            xmax1 = np.zeros((nimg,nb))
            cmax1 = np.zeros((nimg,nb))
        for i in range(0,len(results)):
            Y[irange[i], :, :] = results[i][0]
            ymax[irange[i]] = results[i][1]
            xmax[irange[i]] = results[i][2]
            cmax[irange[i]] = results[i][3]
            if nr:
                ymax1[irange[i],:] = results[i][4][0]
                xmax1[irange[i],:] = results[i][4][1]
                cmax1[irange[i],:] = results[i][4][2]
        if nr:
            yxnr = [ymax1,xmax1,cmax1]
    # perform nonrigid shift with no pool
    #if nr:
    
    
    return Y, ymax, xmax, cmax, yxnr

def get_nFrames(ops):
    if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
        nbytes = os.path.getsize(ops['raw_file'])
    else:
        nbytes = os.path.getsize(ops['reg_file'])

    nFrames = int(nbytes/(2* ops['Ly'] *  ops['Lx']))
    return nFrames

def register_myshifts(ops, data, ymax, xmax):
    ''' rigid shifting of other channel data by ymax and xmax '''
    if ops['num_workers']<0:
        dreg = shift_data((data, ymax, xmax))
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
            dsplit.append([data[ilist,:, :], ymax[ilist], xmax[ilist]])
        with Pool(num_cores) as p:
            results = p.map(shift_data, dsplit)

        dreg = np.zeros_like(data)
        for i in range(0,len(results)):
            dreg[irange[i], :, :] = results[i]
    return dreg

def subsample_frames(ops, nsamps):
    ''' get nsamps frames from binary file for initial reference image'''
    nFrames = ops['nframes']
    Ly = ops['Ly']
    Lx = ops['Lx']
    frames = np.zeros((nsamps, Ly, Lx), dtype='int16')
    nbytesread = 2 * Ly * Lx
    istart = np.linspace(0, nFrames, 1+nsamps).astype('int64')
    if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                reg_file = open(ops['raw_file'], 'rb')
            else:
                reg_file = open(ops['raw_file_chan2'], 'rb')
        else:
            reg_file = open(ops['raw_file'], 'rb')
    else:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                reg_file = open(ops['reg_file'], 'rb')
            else:
                reg_file = open(ops['reg_file_chan2'], 'rb')
        else:
            reg_file = open(ops['reg_file'], 'rb')
    for j in range(0,nsamps):
        reg_file.seek(nbytesread * istart[j], 0)
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        frames[j,:,:] = np.reshape(data, (Ly, Lx))
    reg_file.close()
    return frames

def get_bidiphase(frames):
    ''' computes the bidirectional phase offset
        sometimes in line scanning there will be offsets between lines
        if ops['do_bidiphase'], then bidiphase is computed and applied
    '''
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

def shift_bidiphase(frames, bidiphase):
    ''' shift frames by bidirectional phase offset, bidiphase '''
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
    return frames


def pick_init_init(ops, frames):
    nimg = frames.shape[0]
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = frames @ np.transpose(frames)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1);
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (ops['Ly'], ops['Lx']))
    return refImg

def refine_init_init(ops, frames, refImg):
    niter = 8
    nmax  = np.minimum(100, int(frames.shape[0]/2))
    for iter in range(0,niter):
        ops['refImg'] = refImg
        maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops)
        
        freg, ymax, xmax, cmax, yxnr = register_data(frames, [maskMul, maskOffset, cfRefImg], ops)
        isort = np.argsort(-cmax)
        nmax = int(frames.shape[0] * (1.+iter)/(2*niter))
        refImg = freg[isort[1:nmax], :, :].mean(axis=0).squeeze()
        dy, dx = -np.mean(ymax[isort[1:nmax]]), -np.mean(xmax[isort[1:nmax]])
        refImg = shift_data((refImg, dy, dx, refImg.mean())).squeeze()
        ymax, xmax = ymax+dy, xmax+dx
    return refImg

def pick_init(ops):
    ''' compute initial reference image from ops['nimg_init'] frames '''
    Ly = ops['Ly']
    Lx = ops['Lx']
    nFrames = ops['nframes']
    nFramesInit = np.minimum(ops['nimg_init'], nFrames)
    frames = subsample_frames(ops, nFramesInit)
    if ops['do_bidiphase'] and ops['bidiphase']==0:
        ops['bidiphase'] = get_bidiphase(frames)
        print('computed bidiphase %d'%ops['bidiphase'])
    if ops['bidiphase'] != 0:
        frames = shift_bidiphase(frames.copy(), ops['bidiphase'])
    refImg = pick_init_init(ops, frames)
    refImg = refine_init_init(ops, frames, refImg)
    return refImg

def prepare_refAndMasks(refImg,ops):
    maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops)
    if ops['nonrigid']:
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]
    return refAndMasks

def init_offsets(ops):
    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)
        offsets = [yoff,xoff,corrXY,yoff1,xoff1,corrXY1]
    else:
        offsets = [yoff,xoff,corrXY]

    return offsets

def compute_crop(ops):
    ''' determines ops['badframes'] (using ops['th_badframes'])
        and excludes these ops['badframes'] when computing valid ranges
        from registration in y and x
    '''
    dx = ops['xoff'] - medfilt(ops['xoff'], 101)
    dy = ops['yoff'] - medfilt(ops['yoff'], 101)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy /= dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = ops['corrXY'] / medfilt(ops['corrXY'], 101)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    ops['badframes'] = px > ops['th_badframes'] * 100
    ymin = np.maximum(0, np.ceil(np.amax(ops['yoff'][np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(ops['yoff'])))
    xmin = np.maximum(0, np.ceil(np.amax(ops['xoff'][np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(ops['xoff'])))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    return ops

def apply_reg_shifts(data, ops, offsets, iframes=None):
    ''' apply shifts from register_data to data matrix '''
    if iframes is None:
        iframes = np.arange(0, data.shape[0], 1, int)
    if ops['bidiphase']!=0:
        data = shift_bidiphase(data.copy(), ops['bidiphase'])
    data = register_myshifts(ops, data, offsets[0][iframes], offsets[1][iframes])
    if ops['nonrigid']==True:
        data = nonrigid.register_myshifts(ops, data, offsets[3][iframes], offsets[4][iframes])
    return data

def write_tiffs(data, ops, k, ichan):
    if k==0:
        if ichan==0:
            if ops['functional_chan']==ops['align_by_chan']:
                tifroot = os.path.join(ops['save_path'], 'reg_tif')
            else:
                tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
        else:
            if ops['functional_chan']==ops['align_by_chan']:
                tifroot = os.path.join(ops['save_path'], 'reg_tif')
            else:
                tifroot = os.path.join(ops['save_path'], 'reg_tif_chan2')
        if not os.path.isdir(tifroot):
            os.makedirs(tifroot)
        print(tifroot)
    fname = 'file_chan%0.3d.tif'%k
    io.imsave(os.path.join(tifroot, fname), data)

def bin_paths(ops, raw):
    raw_file_align = []
    raw_file_alt = []
    reg_file_align = []
    reg_file_alt = []
    if raw:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                raw_file_align = ops['raw_file']
                raw_file_alt = ops['raw_file_chan2']
                reg_file_align = ops['reg_file']
                reg_file_alt = ops['reg_file_chan2']
            else:
                raw_file_align = ops['raw_file_chan2']
                raw_file_alt = ops['raw_file']
                reg_file_align = ops['reg_file_chan2']
                reg_file_alt = ops['reg_file']
        else:
            raw_file_align = ops['raw_file']
            reg_file_align = ops['reg_file']
    else:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                reg_file_align = ops['reg_file']
                reg_file_alt = ops['reg_file_chan2']
            else:
                reg_file_align = ops['reg_file_chan2']
                reg_file_alt = ops['reg_file']
        else:
            reg_file_align = ops['reg_file']
    return reg_file_align, reg_file_alt, raw_file_align, raw_file_alt

def register_binary_to_ref(ops, refImg, reg_file_align, raw_file_align):
    ''' register binary data to reference image refImg '''
    offsets = init_offsets(ops)
    refAndMasks = prepare_refAndMasks(refImg,ops)

    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch
    raw = 'keep_movie_raw' in ops and ops['keep_movie_raw']
    if raw:
        reg_file_align = open(reg_file_align, 'wb')
        raw_file_align = open(raw_file_align, 'rb')
    else:
        reg_file_align = open(reg_file_align, 'r+b')

    meanImg = np.zeros((Ly, Lx))
    k=0
    nfr=0
    k0 = tic()
    while True:
        if raw:
            buff = raw_file_align.read(nbytesread)
        else:
            buff = reg_file_align.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        if data.size==0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        print(data.shape)

        dout = register_data(data, refAndMasks, ops)
        print(toc(k0))
        data = np.minimum(dout[0], 2**15 - 2)
        meanImg += data.sum(axis=0)
        data = data.astype('int16')
        
        # write to reg_file_align
        if not raw:
            reg_file_align.seek(-2*data.size,1)
        reg_file_align.write(bytearray(data))

        # compile offsets (dout[1:])
        for n in range(len(dout)-1):
            if n < 3:
                offsets[n] = np.hstack((offsets[n], dout[n+1]))
            else:
                # add on nonrigid stats
                for m in range(len(dout[-1])):
                    offsets[n+m] = np.vstack((offsets[n+m], dout[-1][m]))

        # write registered tiffs
        if ops['reg_tif']:
            write_tiffs(data, ops, k, 0)

        nfr += data.shape[0]
        k += 1
        #if k%5==0:
        print('registered %d/%d frames in time %4.2f'%(nfr, ops['nframes'], toc(k0)))

    print('registered %d/%d frames in time %4.2f'%(nfr, ops['nframes'], toc(k0)))

    # mean image across all frames
    if ops['nchannels']==1 or ops['functional_chan']==ops['align_by_chan']:
        ops['meanImg'] = meanImg/ops['nframes']
    else:
        ops['meanImg_chan2'] = meanImg/ops['nframes']

    reg_file_align.close()
    if raw:
        raw_file_align.close()
    return ops, offsets

def apply_shifts_to_binary(ops, offsets, reg_file_alt, raw_file_alt):
    ''' apply registration shifts to binary data'''

    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch
    raw = 'keep_movie_raw' in ops and ops['keep_movie_raw']
    ix = 0
    meanImg = np.zeros((Ly, Lx))
    k=0
    k0 = tic()
    if raw:
        reg_file_alt = open(reg_file_alt, 'wb')
        raw_file_alt = open(raw_file_alt, 'rb')
    else:
        reg_file_alt = open(reg_file_alt, 'r+b')
    while True:
        if raw:
            buff = raw_file_alt.read(nbytesread)
        else:
            buff = reg_file_alt.read(nbytesread)

        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        if data.size==0:
            break
        data = np.reshape(data[:int(np.floor(data.shape[0]/Ly/Lx)*Ly*Lx)], (-1, Ly, Lx))
        nframes = data.shape[0]

        # register by pre-determined amount
        iframes = ix + np.arange(0,nframes,1,int)
        data = apply_reg_shifts(data, ops, offsets, iframes)
        meanImg += data.sum(axis=0)

        ix += nframes
        data = data.astype('int16')

        # write to binary
        if not raw:
            reg_file_alt.seek(-2*data.size,1)
        reg_file_alt.write(bytearray(data))

        # write registered tiffs
        if ops['reg_tif_chan2']:
            write_tiffs(data, ops, k, 1)
        k+=1
    if ops['functional_chan']!=ops['align_by_chan']:
        ops['meanImg'] = meanImg/ops['nframes']
    else:
        ops['meanImg_chan2'] = meanImg/ops['nframes']
    print('registered second channel in time %4.2f'%(toc(k0)))

    reg_file_alt.close()
    if raw:
        raw_file_alt.close()

    return ops


def register_binary(ops, refImg=None):
    ''' registration of binary files '''
    # if ops is a list of dictionaries, each will be registered separately
    if (type(ops) is list) or (type(ops) is np.ndarray):
        for op in ops:
            op = register_binary(op)
        return ops

    # make blocks for nonrigid
    if ops['nonrigid']:
        ops = utils.make_blocks(ops)

    ops['nframes'] = get_nFrames(ops)

    # check number of frames and print warnings
    if ops['nframes']<50:
        raise Exception('the total number of frames should be at least 50 ')
    if ops['nframes']<200:
        print('number of frames is below 200, unpredictable behaviors may occur')

    if 'do_regmetrics' in ops:
        do_regmetrics = ops['do_regmetrics']
    else:
        do_regmetrics = True

    k0 = tic()

    # compute reference image
    if refImg is not None:
        print('using reference frame given')
        print('will not compute registration metrics')
        do_regmetrics = False
    else:
        refImg = pick_init(ops)
        print('computed reference frame for registration in time %4.2f'%(toc(k0)))
    ops['refImg'] = refImg

    # get binary file paths
    raw = 'keep_movie_raw' in ops and ops['keep_movie_raw']
    reg_file_align, reg_file_alt, raw_file_align, raw_file_alt = bin_paths(ops, raw)

    k = 0
    nfr = 0

    # register binary to reference image
    ops, offsets = register_binary_to_ref(ops, refImg, reg_file_align, raw_file_align)

    if ops['nchannels']>1:
        ops = apply_shifts_to_binary(ops, offsets, reg_file_alt, raw_file_alt)

    ops['yoff'] = offsets[0]
    ops['xoff'] = offsets[1]
    ops['corrXY'] = offsets[2]
    if ops['nonrigid']:
        ops['yoff1'] = offsets[3]
        ops['xoff1'] = offsets[4]
        ops['corrXY1'] = offsets[5]

    # compute valid region
    # return frames which fall outside range
    ops = compute_crop(ops)

    if 'ops_path' in ops:
        np.save(ops['ops_path'], ops)

    # compute metrics for registration
    if do_regmetrics:
        ops = regmetrics.get_pc_metrics(ops)
        print('computed registration metrics in time %4.2f'%(toc(k0)))

    if 'ops_path' in ops:
        np.save(ops['ops_path'], ops)
    return ops



def register_npy(Z, ops):
    # if ops does not have refImg, get a new refImg
    if 'refImg' not in ops:
        ops['refImg'] = Z.mean(axis=0)
    ops['nframes'], ops['Ly'], ops['Lx'] = Z.shape

    if ops['nonrigid']:
        ops = utils.make_blocks(ops)

    Ly = ops['Ly']
    Lx = ops['Lx']

    nbatch = ops['batch_size']
    meanImg = np.zeros((Ly, Lx)) # mean of this stack

    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)

    maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops) # prepare masks for rigid registration
    if ops['nonrigid']:
        # prepare masks for non- rigid registration
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
        nb = ops['nblocks'][0] * ops['nblocks'][1]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]

    k = 0
    nfr = 0
    Zreg = np.zeros((nframes, Ly, Lx,), 'int16')
    while True:
        irange = np.arange(nfr, nfr+nbatch)
        data = Z[irange, :,:]
        if data.size==0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax, yxnr = phasecorr(data, refAndMasks, ops)
        dwrite = dwrite.astype('int16') # need to hold on to this
        meanImg += dwrite.sum(axis=0)
        yoff = np.hstack((yoff, ymax))
        xoff = np.hstack((xoff, xmax))
        corrXY = np.hstack((corrXY, cmax))
        if ops['nonrigid']:
            yoff1 = np.vstack((yoff1, yxnr[0]))
            xoff1 = np.vstack((xoff1, yxnr[1]))
            corrXY1 = np.vstack((corrXY1, yxnr[2]))
        nfr += dwrite.shape[0]
        Zreg[irange] = dwrite

        k += 1
        if k%5==0:
            print('registered %d/%d frames in time %4.2f'%(nfr, ops['nframes'], toc(k0)))

    # compute some potentially useful info
    ops['th_badframes'] = 100
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    dxy = (dx**2 + dy**2)**.5
    cXY = corrXY / medfilt(corrXY, 101)
    px = dxy/np.mean(dxy) / np.maximum(0, cXY)
    ops['badframes'] = px > ops['th_badframes']
    ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(yoff)))
    xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(xoff)))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    ops['corrXY'] = corrXY

    ops['yoff'] = yoff
    ops['xoff'] = xoff

    if ops['nonrigid']:
        ops['yoff1'] = yoff1
        ops['xoff1'] = xoff1
        ops['corrXY1'] = corrXY1

    ops['meanImg'] = meanImg/ops['nframes']

    return Zreg, ops
