import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
import math

def getNeuropilBasis(ops, Ly, Lx):
    ''' computes neuropil basis functions
        inputs: 
            ops, Ly, Lx
            from ops: ratio_neuropil, tile_factor, diameter, neuropil_type
        outputs: 
            basis functions (pixels x nbasis functions)
    '''
    ratio_neuropil = ops['ratio_neuropil']
    tile_factor    = ops['tile_factor']
    diameter       = ops['diameter']
    
    ntiles  = int(np.ceil(tile_factor * (Ly+Lx)/2 / (ratio_neuropil * diameter/2)))
    
    yc = np.linspace(1, Ly, ntiles)
    xc = np.linspace(1, Lx, ntiles)
    ys = np.arange(0,Ly)
    xs = np.arange(0,Lx)
        
    sigy = 4*(Ly - 1)/ntiles
    sigx = 4*(Lx - 1)/ntiles
        
    S = np.zeros((Ly, Lx, ntiles, ntiles), np.float32)
    for kx in range(ntiles):        
        for ky in range(ntiles):        
            cosy = 1 + np.cos(2*math.pi*(ys - yc[ky])/sigy)
            cosx = 1 + np.cos(2*math.pi*(xs - xc[kx])/sigx)
            cosy[abs(ys-yc[ky]) > sigy/2] = 0
            cosx[abs(xs-xc[kx]) > sigx/2] = 0
            S[:,:,ky,kx] = np.expand_dims(cosy,axis=1) @ np.expand_dims(cosx,axis=1).transpose()
    
    S = np.reshape(S,(Ly*Lx, ntiles*ntiles))
    S = S / np.expand_dims(np.sum(np.abs(S)**2,axis=-1)**0.5,axis=1)
    S = S.transpose()
    return S

def circleMask(d0):
    ''' creates array with indices which are the radius of that x,y point
        inputs: 
            d0 (patch of (-d0,d0+1) over which radius computed
        outputs: 
            rs: array (2*d0+1,2*d0+1) of radii
            (dx,dy): indices in rs where the radius is less than d0
    '''
    dx  = np.tile(np.arange(-d0,d0+1), (2*d0+1,1))
    dy  = dx.transpose()
    rs  = (dy**2 + dx**2) ** 0.5
    dx  = dx[rs<=d0]
    dy  = dy[rs<=d0]
    return rs, dx, dy

def morphOpen(V, footprint):
    ''' computes the morphological opening of V (correlation map) with (usually circular) footprint'''
    vrem   = filters.minimum_filter(V, footprint=footprint)
    vrem   = -filters.minimum_filter(-vrem, footprint=footprint)
    return vrem
        
def localMax(V, footprint, thres):
    ''' find local maxima of V using a filter with (usually circular) footprint
        inputs:
            V (correlation map), footprint, thres
        outputs:
            indices i,j of local max greater than thres
    '''
    
    maxV = filters.maximum_filter(V, footprint=footprint)
    imax = (V > (maxV - 1e-10)) & (V > thres)
    i,j  = imax.nonzero()
    i    = i.astype(np.int32)
    j    = j.astype(np.int32)
    return i,j
    
def localRegion(i,j,dy,dx,Ly,Lx):
    ''' returns valid indices of local region surrounding (i,j) of size (dy.size, dx.size)'''
    xc = dx + j
    yc = dy + i
    goodi = (xc>=0) & (xc<Lx) & (yc>=0) & (yc<Ly)
    xc = xc[goodi]
    yc = yc[goodi] 
    return yc, xc, goodi
        
def connectedRegion(mLam, rsmall, d0):
    mLam0 = np.zeros(rsmall.shape)
    mLam1 = np.zeros(rsmall.shape)
    # non-zero lam's
    mLam0[rsmall<=d0] = mLam>0
    mLam1[rsmall<=d0] = mLam

    mmax = mLam1.argmax()
    mask = np.zeros(rsmall.size)
    mask[mmax] = 1
    mask = np.resize(mask, (2*d0+1, 2*d0+1))

    for m in range(int(np.ceil(mask.shape[0]/2))):
        mask = filters.maximum_filter(mask, [d0/4,d0/4]) * mLam0   
        
    mLam *= mask[rsmall<=d0]
    return mLam 
