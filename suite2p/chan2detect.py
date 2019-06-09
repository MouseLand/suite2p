import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
from suite2p import utils, roiextract
import time

'''
identify cells with channel 2 brightness (aka red cells)

main function is detect
takes from ops: 'meanImg', 'meanImg_chan2', 'Ly', 'Lx'
takes from stat: 'ypix', 'xpix', 'lam'
'''

def quadrant_mask(Ly,Lx,ny,nx,sT):
    mask = np.zeros((Ly,Lx), np.float32)
    mask[np.ix_(ny,nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask

def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    sT = np.round((Ly + Lx) / (nblks*2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), np.float32)
    weights = np.zeros((nblks, nblks), np.float32)
    yb = np.linspace(0, Ly, nblks+1).astype(int)
    xb = np.linspace(0, Lx, nblks+1).astype(int)
    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy+1]).astype(int)
            nx = np.arange(xb[ix], xb[ix+1]).astype(int)
            mask[:,:,iy,ix] = quadrant_mask(Ly, Lx, ny, nx, sT)
            x  = mimg[np.ix_(ny,nx)].flatten()
            x2  = mimg2[np.ix_(ny,nx)].flatten()
            # predict chan2 from chan1
            a = (x * x2).sum() / (x * x).sum()
            weights[iy,ix] = a
    mask /= mask.sum(axis=-1).sum(axis=-1)[:,:,np.newaxis,np.newaxis]
    mask *= weights
    mask *= mimg[:,:,np.newaxis,np.newaxis]
    mimg2 -= mask.sum(axis=-1).sum(axis=-1)
    mimg2 = np.maximum(0, mimg2)
    return mimg2

def detect(ops, stat):
    #ops2 = ops.copy()
    mimg = ops['meanImg'].copy()
    mimg2 = ops['meanImg_chan2'].copy()

    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    nblks = 3
    Ly = ops['Ly']
    Lx = ops['Lx']
    mimg2_corr = correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2)
    ops['meanImg_chan2_corrected'] = mimg2_corr

    # compute pixels in cell and in area around cell (including overlaps)
    # (exclude pixels from other cells)
    # ops['min_neuropil_pixels'] = 80
    _, cell_pix, cell_masks = roiextract.create_cell_masks(ops, stat)
    neuropil_masks = roiextract.create_neuropil_masks(ops, stat, cell_pix)
    neuropil_masks = np.reshape(neuropil_masks,(-1,Ly*Lx))
    cell_masks     = np.reshape(cell_masks,(-1,Ly*Lx))

    inpix = cell_masks @ mimg2.flatten()
    extpix = neuropil_masks @ mimg2.flatten()
    inpix = np.maximum(1e-3, inpix)
    redprob = inpix / (inpix + extpix)
    redcell = redprob > ops['chan2_thres']

    redcell = np.concatenate((redcell[:,np.newaxis], redprob[:,np.newaxis]), axis=1)

    return ops, redcell
