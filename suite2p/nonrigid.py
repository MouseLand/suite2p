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

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad

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

        # put reference in fft domain
        #if ops['pad_fft']:
        #    cfRefImg   = np.conj(fft.fft2(refImg,
        #                        [next_fast_len(Ly), next_fast_len(Lx)]))
        #else:
        cfRefImg   = np.conj(fft.fft2(refImg))
        if ops['do_phasecorr']:
            absRef     = np.absolute(cfRefImg)
            cfRefImg   = cfRefImg / (eps0 + absRef)

        # gaussian filter
        fhg = register.gaussian_fft(ops['smooth_sigma'], cfRefImg.shape[0], cfRefImg.shape[1])
        cfRefImg *= fhg

        cfRefImg1[n,0,:,:] = (cfRefImg.astype('complex64'))
    return maskMul1, maskOffset1, cfRefImg1

def phasecorr_worker(inputs):
    ''' loop through blocks and compute phase correlations'''
    data, refAndMasks, ops = inputs
    nimg, Ly, Lx = data.shape
    maxregshift = np.round(ops['maxregshiftNR'])
    LyMax = np.diff(np.array(ops['yblock']))
    ly = int(np.diff(ops['yblock'][0]))
    lx = int(np.diff(ops['xblock'][0]))
    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(ly,lx)/2.)-lpad))
    nb = len(ops['yblock'])
    nblocks = ops['nblocks']
    ymax1 = np.zeros((nimg,nb),np.float32)
    cmax1 = np.zeros((nimg,nb),np.float32)
    #snr1  = np.zeros((nimg,nb),np.float32)
    xmax1 = np.zeros((nimg,nb),np.float32)
    data_block = np.zeros((nb,nimg,ly,lx),np.float32)
    if ops['1Preg']:
        data1 = register.one_photon_preprocess(data.copy(), ops)
    else:
        data1 = data
    # compute phase-correlation of blocks
    for n in range(nb):
        yind = ops['yblock'][n]
        yind = np.arange(yind[0],yind[-1]).astype(int)
        xind = ops['xblock'][n]
        xind = np.arange(xind[0],xind[-1]).astype(int)
        data_block[n,:,:,:] = data1[np.ix_(np.arange(0,nimg).astype(int),yind,xind)]
    del data1
    cc1 = register.correlation_map(data_block, refAndMasks, ops['do_phasecorr'])
    lyhalf = int(np.floor(cc1.shape[-2]/2))
    lxhalf = int(np.floor(cc1.shape[-1]/2))
    cc0 = cc1[:, :, (lyhalf-lcorr-lpad):(lyhalf+lcorr+1+lpad), (lxhalf-lcorr-lpad):(lxhalf+lcorr+1+lpad)]
    cc0 = cc0.reshape((cc0.shape[0], -1))
    cc2 = []
    R = ops['NRsm']
    cc2.append(cc0)
    for j in range(2):
        cc2.append(R @ cc2[j])
    for j in range(len(cc2)):
        cc2[j] = cc2[j].reshape((nb, nimg, 2*lcorr+2*lpad+1, 2*lcorr+2*lpad+1))
    ccsm = cc2[0]
    for n in range(nb):
        snr = np.ones((nimg,), 'float32')
        for j in range(len(cc2)):
            ism = snr<ops['snr_thresh']
            if np.sum(ism)==0:
                break
            cc = cc2[j][n,ism,:,:]
            if j>0:
                ccsm[n, ism, :, :] = cc
            snr[ism] = register.getSNR(cc, (lcorr,lpad, lcorr+lpad, lcorr+lpad), ops)

    for n in range(nb):
        cc = ccsm[n,:,:,:]
        ymax, xmax, cmax = register.getXYup(cc, (lcorr,lpad, lcorr+lpad, lcorr+lpad), ops)
        #cc = cc1[n,:,:,:]
        #ymax, xmax, cmax, snr = register.getXYup(cc, (lcorr,lpad, lyhalf, lxhalf), ops)
        # here cc1 should be smooth if the SNR is insufficient
        ymax1[:,n] = ymax
        xmax1[:,n] = xmax
        cmax1[:,n] = cmax
        #snr1[:,n] = snr

    # smooth cc across blocks if sig>0
    # currently not used
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
    Y = shift_data_worker((data, ops, ymax1, xmax1))
    #Y=data
    return Y, ymax1, xmax1, cmax1

def shift_data_worker(inputs):
    ''' piecewise affine transformation of data using shifts from phasecorr_worker '''
    if len(inputs)==4:
        data,ops,ymax1,xmax1 = inputs
    else:
        data,ops,ymax1,xmax1,ymax,xmax = inputs
        data = register.shift_data_worker((data, ymax, xmax, ops['refImg'].mean()))

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

        #xf = xcoor.astype(np.int16)
        #yf = ycoor.astype(np.int16)
        #xc = xf + 1
        #yc = yf + 1

        #dy = ycoor-yf
        #dx = xcoor-xf

        #xf = np.maximum(0, np.minimum(Lx-1, xf))
        #yf = np.maximum(0, np.minimum(Ly-1, yf))
        #yc = np.maximum(0, np.minimum(Ly-1, yc))
        #xc = np.maximum(0, np.minimum(Lx-1, xc))

        #Y[t] += data[t][yf, xf] * (1 - dy) * (1 - dx)
        #Y[t] += data[t][yf, xc] * (1 - dy) * dx
        #Y[t] += data[t][yc, xf] * dy * (1 - dx)
        #Y[t] += data[t][yc, xc] * dy * dx
    #Y = np.reshape(Y, (nimg, Ly, Lx))
    return Y

def shift_data(ops, data, ymax, xmax, ymax1, xmax1):
    ''' first shift rigid by ymax and xmax'''
    ''' then non-rigid shift of other channel data by ymax1 and xmax1 '''
    if ops['num_workers']<0:
        dreg = shift_data_woker((data, ops, ymax1, xmax1, ymax, xmax))
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
            dsplit.append([data[ilist,:, :], ops, ymax1[ilist], xmax1[ilist],
                                                  ymax[ilist], xmax[ilist]])
        with Pool(num_cores) as p:
            results = p.map(shift_data_worker, dsplit)
        dreg = np.zeros_like(data)
        for i in range(0,len(results)):
            dreg[irange[i], :, :] = results[i]
    return dreg
