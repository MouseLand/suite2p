import numpy as np
import time
import matplotlib.pyplot as plt
from numpy import fft
from numpy import random as rnd
from joblib import Parallel, delayed
import multiprocessing

eps0 = 1e-5;
sigL = 0.85 # smoothing width for up-sampling kernels, keep it between 0.5 and 1.0...
lpad = 3   # upsample from a square +/- lpad 
smoothSigma = 1.15 # smoothing constant
maskSlope   = 2. # slope of taper mask at the edges
    
# smoothing kernel
def kernelD(a, b):        
    dxs = np.reshape(a[0], (-1,1)) - np.reshape(b[0], (1,-1))
    dys = np.reshape(a[1], (-1,1)) - np.reshape(b[1], (1,-1))
    
    ds = np.square(dxs) + np.square(dys)
    K = np.exp(-ds/(2*np.square(sigL)))    
    return K
    
def mat_upsample(lpad, ops):    
    lar    = np.arange(-lpad, lpad+1)
    larUP  = np.arange(-lpad, lpad+.001, 1./ops['subpixel'])

    x, y   = np.meshgrid(lar, lar)   
    xU, yU = np.meshgrid(larUP, larUP)   

    Kx = kernelD((x,y),(x,y))
    Kx = np.linalg.inv(Kx)
    Kg = kernelD((x,y),(xU,yU))
    
    Kmat = np.dot(Kx, Kg)
    nup = larUP.shape[0]
    
    return Kmat, nup


def prepareMasks(refImg):    
    i0,Ly,Lx = refImg.shape
    
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

    #plt.imshow(maskMul)
    #plt.show()
    #plt.imshow(maskOffset)
    #plt.show()
    
    hgx = np.exp(-np.square(xx/smoothSigma))
    hgy = np.exp(-np.square(yy/smoothSigma))
    hgg = hgy * hgx
    hgg = hgg/hgg.sum()

    fhg = np.real(fft.fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain

    cfRefImg   = np.conj(fft.fft2(refImg));
    absRef     = np.absolute(cfRefImg);
    cfRefImg   = cfRefImg / (eps0 + absRef) * fhg;
    
    return maskMul, maskOffset, cfRefImg

def correlation_map(data, cfRefImg):
    X = fft.fft2(data)
    J = X / (eps0 + np.absolute(X)) 
    J = J * cfRefImg
    cc = np.real(fft.ifft2(J))
    cc = fft.fftshift(cc, axes=(1,2))
    return cc, X

def getXYup(cc, Ls, ops):
    (lcorr, lpad, Lyhalf, Lxhalf) = Ls
    
    nimg = cc.shape[0]
    cc0 = cc[:, (Lyhalf-lcorr):(Lyhalf+lcorr+1), (Lxhalf-lcorr):(Lxhalf+lcorr+1)]

    cc0 = np.reshape(cc0, (nimg, -1))
    ix  = np.argmax(cc0, axis = 1)

    ymax, xmax = np.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    mxpt = [ymax+Lyhalf-lcorr, xmax + Lxhalf-lcorr]
    
    ccmat = np.zeros((nimg, 2*lpad+1, 2*lpad+1))
    for j in range(0, nimg):
        ccmat[j,:,:] = cc[j, (mxpt[0][j] -lpad):(mxpt[0][j] +lpad+1), (mxpt[1][j] -lpad):(mxpt[1][j] +lpad+1)]

    ccmat = np.reshape(ccmat, (nimg,-1))

    Kmat, nup = mat_upsample(lpad, ops)

    ccb = np.dot(ccmat, Kmat)
    imax = np.argmax(ccb, axis=1)
    cmax = np.amax(ccb, axis=1)

    ymax, xmax = np.unravel_index(imax, (nup,nup))

    mdpt = np.floor(nup/2)
    
    ymax,xmax = (ymax-mdpt)/ops['subpixel'], (xmax-mdpt)/ops['subpixel']
    
    ymax, xmax = ymax + mxpt[0] - Lyhalf, xmax + mxpt[1] - Lxhalf

    return ymax, xmax

def shift_data(X, ymax,xmax):
    nimg, Ly, Lx = X.shape
    
    Ny = fft.ifftshift(np.arange(-np.fix(Ly/2), np.ceil(Ly/2)))
    Nx = fft.ifftshift(np.arange(-np.fix(Lx/2), np.ceil(Lx/2)))
    [Nx,Ny] = np.meshgrid(Nx,Ny)
    Nx = Nx / Lx
    Ny = Ny / Ly
    
    dph = Nx * np.reshape(xmax, (-1,1,1)) + Ny * np.reshape(ymax, (-1,1,1))
    
    Y = np.real(fft.ifft2(X * np.exp((2j * np.pi) * dph)))
    return Y
    
def phasecorr_worker(data, refImg, ops):    
    nimg, Ly, Lx = data.shape
    refImg = np.reshape(refImg, (1, Ly, Lx))
    
    Lyhalf = int(np.floor(Ly/2))
    Lxhalf = int(np.floor(Lx/2))
    
    if ops['maxregshift']>0:
        maxregshift = ops['maxregshift'] 
    else:
        maxregshift = np.round(.1*np.maximum(Ly, Lx))

    lcorr = int(np.minimum(maxregshift, np.floor(np.minimum(Ly,Lx)/2.)-lpad))

    (maskMul, maskOffset, cfRefImg) = prepareMasks(refImg)

    data = data * maskMul + maskOffset

    cc, X = correlation_map(data, cfRefImg)

    ymax, xmax = getXYup(cc, (lcorr,lpad, Lyhalf, Lxhalf), ops)
    
    Y = shift_data(X, ymax,xmax)
    
    return Y, ymax, xmax

def phasecorr(data, refImg, ops):    
    nimg = data.shape[0]
    
    if ops['num_workers']<0:
        Y, ymax, xmax = phasecorr_worker(data, refImg, ops)
    else:
        num_cores = ops['num_workers']
        if num_cores<1:
            num_cores = multiprocessing.cpu_count()
            
        nbatch = int(np.ceil(nimg/float(num_cores)))
    
        inputs = range(0, nimg, nbatch)
        irange = []
        for i in inputs:
            irange.append(i + np.arange(0,np.minimum(nbatch, nimg-i)))

        results = Parallel(n_jobs=num_cores)(delayed(phasecorr_worker)(data[irange[j],:, :], refImg, ops) for j in range(0,len(irange)))
        
        Y = np.zeros_like(data)
        ymax = np.zeros((nimg,))
        xmax = np.zeros((nimg,))
        for i in range(0,len(results)):
            Y[irange[i], :, :] = results[i][0]
            ymax[irange[i]] = results[i][1]
            xmax[irange[i]] = results[i][2]
    
    return Y, ymax, xmax

def registerBinary(ops):
    
    
    
    
    
    
    return 