import numpy as np
from numpy import fft
from scipy.ndimage import gaussian_filter
from skimage.transform import warp#, PiecewiseAffineTransform
from scipy.interpolate import interp2d
from suite2p import register
import time
import multiprocessing
from multiprocessing import Pool

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad
smoothSigma = 1.15 # smoothing constant
maskSlope   = 2. # slope of taper mask at the edges

def make_blocks(ops):
    ## split FOV into blocks to register separately
    Ly = ops['Ly']
    Lx = ops['Lx']
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 0.01
    if 'nblocks' in ops:
        nblocks = ops['nblocks']
    else:
        nblocks = [5,2]
        ops['nblocks'] = nblocks
    nblocks = np.array(nblocks)
    if 'block_fraction' in ops:
        bfrac = ops['block_fraction']
    else:
        bfrac = 1.0 / np.maximum(2.0, nblocks-1)
        bfrac[nblocks==1] = 1.0
        ops['block_fraction'] = bfrac
    bpix = bfrac * np.array([Ly,Lx])
    # choose bpix to be the closest power of 2
    bpix = 2**np.round(np.log2(bpix))
    ops['block_overlap'] = np.round((bpix*nblocks - [Ly,Lx]) / (nblocks-1.9))
    # block centers
    yblocks = np.linspace(0, Ly-1, nblocks[0]+1)
    yblocks = np.round((yblocks[:-1] + yblocks[1:]) / 2)
    xblocks = np.linspace(0, Lx-1, nblocks[1]+1)
    xblocks = np.round((xblocks[:-1] + xblocks[1:]) / 2)
    # block ranges
    ib=0
    ops['yblock'] = []
    ops['xblock'] = []
    bhalf = np.floor(bpix / 2)
    for iy in range(nblocks[0]):
        if iy==nblocks[0]-1:
            yind = Ly-1 + np.array([-bhalf[0]*2+1,0])
        elif iy==0:
            yind = np.array([0,bhalf[0]*2-1])
        else:
            yind = yblocks[iy] + np.array([-bhalf[0], bhalf[0]-1])
        for ix in range(nblocks[1]):
            if ix==nblocks[0]-1:
                xind = Lx-1 + np.array([-bhalf[1]*2+1,0])
            elif ix==0:
                xind = np.array([0,bhalf[1]*2-1])
            else:
                xind = xblocks[ix] + np.array([-bhalf[1], bhalf[1]-1])
            ops['yblock'].append(yind)
            ops['xblock'].append(xind)
            ib+=1
    ## smoothing masks
    # gaussian centered on block with width 2/3 the size of block
    # (or user-specified as ops['smooth_blocks'])
    #sigT = [np.diff(yblocks).mean()*2.0/3, np.diff(xblocks).mean()*2.0/3]
    #if 'smooth_blocks' in ops:
    #    sigT = ops['smooth_blocks']
    #sigT = np.maximum(10.0, sigT)
    #ops['smooth_blocks'] = sigT
    return ops

def prepare_masks(refImg0, ops):
    # split refImg0 into multiple parts
    cfRefImg1 = []
    maskMul1 = []
    maskOffset1 = []
    nb = len(ops['yblock'])
    for n in range(nb):
        yind = ops['yblock'][n]
        yind = np.arange(yind[0],yind[-1]+1).astype(int)
        xind = ops['xblock'][n]
        xind = np.arange(xind[0],xind[-1]+1).astype(int)
        refImg = refImg0[np.ix_(yind,xind)]
        Ly,Lx = refImg.shape
        if n==0:
            cfRefImg1 = np.zeros((nb,1,Ly,Lx),'complex64')
            maskMul1 = np.zeros((nb,1,Ly,Lx),'float32')
            maskOffset1 = np.zeros((nb,1,Ly,Lx),'float32')
        x = np.arange(0, Lx)
        y = np.arange(0, Ly)
        x = np.abs(x - x.mean())
        y = np.abs(y - y.mean())
        xx, yy = np.meshgrid(x, y)
        mY = y.max() - 4.
        mX = x.max() - 4.
        maskY = 1./(1.+np.exp((yy-mY)/maskSlope))
        maskX = 1./(1.+np.exp((xx-mX)/maskSlope))
        maskMul = maskY * maskX
        maskOffset = refImg.mean() * (1. - maskMul);
        hgx = np.exp(-np.square(xx/smoothSigma))
        hgy = np.exp(-np.square(yy/smoothSigma))
        hgg = hgy * hgx
        hgg = hgg/hgg.sum()
        fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
        cfRefImg   = np.conj(fft.fft2(refImg));
        absRef     = np.absolute(cfRefImg);
        cfRefImg   = cfRefImg / (eps0 + absRef) * fhg;
        maskMul1[n,0,:,:] = (maskMul.astype('float32'))
        maskOffset1[n,0,:,:] = (maskOffset.astype('float32'))
        cfRefImg1[n,0,:,:] = (cfRefImg.astype('complex64'))
    return maskMul1, maskOffset1, cfRefImg1

def correlation_map(data, refAndMasks):
    maskMul    = refAndMasks[0]
    maskOffset = refAndMasks[1]
    cfRefImg   = refAndMasks[2]
    nb, nimg, Ly, Lx = data.shape
    data = data.astype('float32') * maskMul + maskOffset
    X = fft.fft2(data)
    J = X / (eps0 + np.absolute(X))
    J = J * cfRefImg
    cc = np.real(fft.ifft2(J))
    cc = fft.fftshift(cc, axes=(2,3))
    return cc

def phasecorr_worker(inputs):
    ''' loop through blocks and compute phase correlations'''
    data, refAndMasks, ops = inputs
    maskMul1    = refAndMasks[0]
    maskOffset1 = refAndMasks[1]
    cfRefImg1   = refAndMasks[2]
    nimg, Ly, Lx = data.shape
    maxregshift = np.round(ops['maxregshiftNR'] *np.maximum(Ly, Lx))
    LyMax = np.diff(np.array(ops['yblock']))
    ly = int(np.diff(ops['yblock'][0])+1)
    lx = int(np.diff(ops['xblock'][0])+1)
    lyhalf = int(np.floor(ly/2))
    lxhalf = int(np.floor(lx/2))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(ly,lx)/2.)-lpad))
    nb = len(ops['yblock'])
    nblocks = ops['nblocks']
    ymax1 = np.zeros((nimg,nb),np.float32)
    cmax1 = np.zeros((nimg,nb),np.float32)
    xmax1 = np.zeros((nimg,nb),np.float32)
    data_block = np.zeros((nb,nimg,ly,lx),np.float32)
    # compute phase-correlation of blocks
    for n in range(nb):
        yind = ops['yblock'][n]
        yind = np.arange(yind[0],yind[-1]+1).astype(int)
        xind = ops['xblock'][n]
        xind = np.arange(xind[0],xind[-1]+1).astype(int)
        data_block[n,:,:,:] = data[np.ix_(np.arange(0,nimg).astype(int),yind,xind)]
    cc1 = correlation_map(data_block, refAndMasks)
    for n in range(nb):
        cc = cc1[n,:,:,:]
        ymax, xmax, cmax = register.getXYup(cc, (lcorr,lpad, lyhalf, lxhalf), ops)
        ymax1[:,n] = ymax
        xmax1[:,n] = xmax
        cmax1[:,n] = cmax
    # smooth cc across blocks if sig>0
    sig = 0
    if sig>0:
        cc1 = np.reshape(cc1,(nimg,ly,lx,nblocks[0],nblocks[1]))
        cc1 = gaussian_filter(cc1, [0,0,0,sig,sig])
        cc1 = np.reshape(cc1,(nimg,ly,lx,nb))
        for n in range(nb):
            ymax, xmax, cmax = register.getXYup(cc1[:,:,:,n], (lcorr,lpad, lyhalf, lxhalf), ops)
            ymax1[:,n] = ymax
            xmax1[:,n] = xmax
            cmax1[:,n] = cmax
    Y = shift_data((data, ymax1, xmax1, ops))
    return Y, ymax1, xmax1, cmax1

def shift_data(inputs):
    ''' piecewise affine transformation of data using shifts from phasecorr_worker '''
    data,ymax,xmax,ops = inputs
    nblocks = ops['nblocks']
    if data.ndim<3:
        data = data[np.newaxis,:,:]
    nimg,Ly,Lx = data.shape
    Y = np.zeros(data.shape, np.float32)
    nb = ymax.shape[1]
    ymax = np.reshape(ymax, (nimg,nblocks[0], nblocks[1]))
    xmax = np.reshape(xmax, (nimg,nblocks[0], nblocks[1]))
    # make arrays of control points for piecewise-affine transform
    # includes centers of blocks AND edges of blocks
    # note indices are flipped for control points
    # block centers
    y = np.round(np.unique(np.array(ops['yblock']).mean(axis=1)))
    y = np.hstack((0,y,Ly-1))
    x = np.round(np.unique(np.array(ops['xblock']).mean(axis=1)))
    x = np.hstack((0,x,Lx-1))
    mshx,mshy = np.meshgrid(np.arange(0,Ly),np.arange(0,Lx))
    # loop over frames
    for t in range(nimg):
        I = data[t,:,:]
        ymax0 = np.pad(ymax[t,:,:],((1,),(1,)),mode='edge')
        xmax0 = np.pad(xmax[t,:,:],((1,),(1,)),mode='edge')
        fy = interp2d(y,x,ymax0,kind='linear')
        fx = interp2d(y,x,xmax0,kind='linear')
        # interpolated values on grid with all points
        fyout = fy(np.arange(0,Ly),np.arange(0,Lx)) + mshy
        fxout = fx(np.arange(0,Ly),np.arange(0,Lx)) + mshx
        coords = np.concatenate((fyout[np.newaxis,:],fxout[np.newaxis,:]))
        Iw = warp(I,coords, order=0, clip=False, preserve_range=True)
        Y[t,:,:] = Iw
    return Y

def register_myshifts(ops, data, ymax, xmax):
    if ops['num_workers']<0:
        dreg = shift_data((data, ymax, xmax, ops))
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
            dsplit.append([data[ilist,:, :], ymax[ilist], xmax[ilist], ops])
        with Pool(num_cores) as p:
            results = p.map(shift_data, dsplit)

        dreg = np.zeros_like(data)
        for i in range(0,len(results)):
            dreg[irange[i], :, :] = results[i]
    return dreg
