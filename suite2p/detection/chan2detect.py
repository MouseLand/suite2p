import numpy as np
from scipy.ndimage import gaussian_filter
from ..extraction import masks 
from . import utils

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

def intensity_ratio(ops, stats):
    """ compute pixels in cell and in area around cell (including overlaps)
        (exclude pixels from other cells) """
    Ly, Lx = ops['Ly'], ops['Lx']
    cell_pix = masks.create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'])
    cell_masks0 = [masks.create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stats]
    neuropil_ipix = masks.create_neuropil_masks(
        ypixs=[stat['ypix'] for stat in stats],
        xpixs=[stat['xpix'] for stat in stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=ops['inner_neuropil_radius'],
        min_neuropil_pixels=ops['min_neuropil_pixels'],
    )
    cell_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    neuropil_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    for cell_mask, cell_mask0, neuropil_mask, neuropil_mask0 in zip(cell_masks, cell_masks0, neuropil_masks, neuropil_ipix):
        cell_mask[cell_mask0[0]] = cell_mask0[1]
        neuropil_mask[neuropil_mask0.astype(np.int64)] = 1. / len(neuropil_mask0)

    mimg2 = ops['meanImg_chan2']
    inpix = cell_masks @ mimg2.flatten()
    extpix = neuropil_masks @ mimg2.flatten()
    inpix = np.maximum(1e-3, inpix)
    redprob = inpix / (inpix + extpix)
    redcell = redprob > ops['chan2_thres']
    return np.stack((redcell, redprob), axis=-1)

def cellpose_overlap(stats, mimg2):
    from . import anatomical 
    masks = anatomical.roi_detect(mimg2)[0]
    Ly, Lx = masks.shape
    redstats = np.zeros(len(stats), np.float32)
    for i in range(len(stats)):
        smask = np.zeros((Ly, Lx), np.uint16)
        ypix0, xpix0= stats[i]['ypix'], stats[i]['xpix']
        smask[ypix0, xpix0] = 1
        ious = utils.mask_ious(masks, smask)[0]
        iou = ious.max()
        redstats[i] = np.array([iou>0.5, iou])
    return redstats

def detect(ops, stats):
    mimg = ops['meanImg'].copy()
    mimg2 = ops['meanImg_chan2'].copy()

    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    nblks = 3
    Ly, Lx = ops['Ly'], ops['Lx']
    ops['meanImg_chan2_corrected'] = correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2)

    redstats = None
    if ops.get('anatomical_red', True):
        try:
            print('>>>> CELLPOSE estimating masks in anatomical channel')
            redstats = cellpose_overlap(stats, mimg2)
        except:
            print('ERROR importing or running cellpose, continuing without anatomical estimates')
            
    if redstats is None:
        redstats = intensity_ratio(ops, stats)
        
    return ops, redstats
