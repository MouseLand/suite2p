import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils

def draw_masks(ops, stat, ops_plot, iscell, ichosen):
    '''creates RGB masks using stat and puts them in M1 or M2 depending on
    whether or not iscell is True for a given ROI
    args:
        ops: mean_image, Vcorr
        stat: xpix,ypix
        iscell: vector with True if ROI is cell
        ops_plot: plotROI, view, color, randcols
    outputs:
        M1: ROIs that are True in iscell
        M2: ROIs that are False in iscell
    '''
    plotROI = ops_plot[0]
    view    = ops_plot[1]
    color   = ops_plot[2]
    if color == 0:
        cols = ops_plot[3]
    else:
        cols = ops_plot[3]
    ncells = iscell.shape[0]
    iclust = -1*np.ones((2,Ly,Lx),np.int32)
    Lam = np.zeros((2,Ly,Lx))
    H = np.zeros((2,Ly,Lx))
    S  = np.ones((2,Ly,Lx))
    for n in range(0,ncells):
        lam     = stat[n]['lam']
        ipix    = stat[n]['ipix']
        if ipix is not None:
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            wmap = int(iscell[n])*np.ones(ypix.shape)
            Lam[wmap,ypix,xpix]    = lam
            iclust[wmap,ypix,xpix] = n*np.ones(ypix.shape)
            H[wmap,ypix,xpix]      = cols[n]*np.ones(ypix.shape)
            if n==ichosen:
                S[wmap,ypix,xpix] = np.zeros(ypix.shape)

    V  = np.maximum(0, np.minimum(1, 0.75 * Lam / Lam[Lam>1e-10].mean()))
    #V  = np.expand_dims(V,axis=2)
    M = []
    for j in range(0,2):
        hsv = np.concatenate((H[j,:,:],S[j,:,:],V[j,:,:]),axis=2)
        rgb = hsv_to_rgb(hsv)
        M.append(rgb)
    return M
