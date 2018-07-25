import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils
from matplotlib.colors import hsv_to_rgb

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
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = iscell.shape[0]
    Lam = np.zeros((2,Ly,Lx,1))
    H = np.zeros((2,Ly,Lx,1))
    S  = np.ones((2,Ly,Lx,1))
    for n in range(0,ncells):
        lam     = stat[n]['lam']
        ypix    = stat[n]['ypix'].astype(np.int32)
        if ypix is not None:
            xpix = stat[n]['xpix'].astype(np.int32)
            wmap = (1-int(iscell[n]))*np.ones(ypix.shape,dtype=np.int32)
            Lam[wmap,ypix,xpix]    = np.expand_dims(lam,axis=1)
            H[wmap,ypix,xpix]      = cols[n]*np.expand_dims(np.ones(ypix.shape), axis=1)
            if n==ichosen:
                S[wmap,ypix,xpix] = np.expand_dims(np.zeros(ypix.shape), axis=1)

    V  = np.maximum(0, np.minimum(1, 0.75 * Lam / Lam[Lam>1e-10].mean()))
    #V  = np.expand_dims(V,axis=2)
    M = []
    for j in range(0,2):
        hsv = np.concatenate((H[j,:,:],S[j,:,:],V[j,:,:]),axis=2)
        rgb = hsv_to_rgb(hsv)
        M.append(rgb)
    return M

def ROI_index(ops, stat):
    ncells = len(stat)
    Ly = ops['Ly']
    Lx = ops['Lx']
    iROI = -1 * np.ones((Ly,Lx), dtype=np.int32)
    for n in range(ncells):
        ypix = stat[n]['ypix']
        if ypix is not None:
            xpix = stat[n]['xpix']
            iROI[ypix,xpix] = n
    return iROI
