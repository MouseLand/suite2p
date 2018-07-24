import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils

def draw_masks(ops, stat, iscell, ops_plot):
    '''creates RGB masks using stat and puts them in M1 or M2 depending on
    whether or not iscell is True for a given ROI
    args:
        ops: mean_image, Vcorr
        stat: xpix,ypix
        iscell: vector with True if ROI is cell
        ops_plot: plotROI, view, color,
    outputs:
        M1: ROIs that are True in iscell
        M2: ROIs that are False in iscell
    '''
    ncells = iscell.shape[0]
    r=np.random.random((ncells,))
    iclust = -1*np.ones((Ly,Lx),np.int32)
    Lam = np.zeros((Ly,Lx))
    H = np.zeros((Ly,Lx,1))
    for n in range(ncells):
        goodi   = np.array((mPix[:,n]>=0).nonzero()).astype(np.int32)
        goodi   = goodi.flatten()
        n0      = n*np.ones(goodi.shape,np.int32)
        lam     = mLam[goodi,n0]
        ipix    = mPix[mPix[:,n]>=0,n].astype(np.int32)
        if ipix is not None:
            ypix,xpix = np.unravel_index(ipix, (Ly,Lx))
            isingle = Lam[ypix,xpix]+1e-4 < lam
            ypix = ypix[isingle]
            xpix = xpix[isingle]
            Lam[ypix,xpix] = lam[isingle]
            iclust[ypix,xpix] = n*np.ones(ypix.shape)
            H[ypix,xpix,0] = r[n]*np.ones(ypix.shape)

    S  = np.ones((Ly,Lx,1))
    V  = np.maximum(0, np.minimum(1, 0.75 * Lam / Lam[Lam>1e-10].mean()))
    V  = np.expand_dims(V,axis=2)
    hsv = np.concatenate((H,S,V),axis=2)
    rgb = hsv_to_rgb(hsv)

    return rgb
