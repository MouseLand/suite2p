import time
import numpy as np
from pathlib import Path
from . import sourcery, sparsedetect, masks, chan2detect, utils


def main_detect(ops, stat=None):
    stat = select_rois(ops, stat)
    # extract fluorescence and neuropil
    cell_pix, cell_masks, neuropil_masks = make_masks(ops, stat)
    ic = np.ones(len(stat), np.bool)
    # if second channel, detect bright cells in second channel
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops['save_path']).joinpath('redcell.npy'), redcell[ic])
    return cell_pix, cell_masks, neuropil_masks, stat, ops


def select_rois(ops, stat=None):
    t0 = time.time()
    if stat is None:
        if ops['sparse_mode']:
            ops, stat = sparsedetect.sparsery(ops)
        else:
            ops, stat = sourcery.sourcery(ops)
        print('Found %d ROIs, %0.2f sec' % (len(stat), time.time() - t0))
    stat = roi_stats(ops, stat)

    stat = masks.get_overlaps(stat, ops)
    stat, ix = masks.remove_overlappers(stat, ops, ops['Ly'], ops['Lx'])
    print('After removing overlaps, %d ROIs remain' % (len(stat)))
    return stat


def make_masks(ops, stat):
    t0=time.time()
    cell_pix, cell_masks = masks.create_cell_masks(stat, ops['Ly'], ops['Lx'], ops['allow_overlap'])
    neuropil_masks = masks.create_neuropil_masks(ops, stat, cell_pix)
    Ly=ops['Ly']
    Lx=ops['Lx']
    neuropil_masks = np.reshape(neuropil_masks, (-1,Ly*Lx))
    print('Masks made in %0.2f sec.'%(time.time()-t0))
    return cell_pix, cell_masks, neuropil_masks


def roi_stats(ops, stat):
    """ computes statistics of ROIs

    Parameters
    ----------
    ops : dictionary
        'aspect', 'diameter'

    stat : dictionary
        'ypix', 'xpix', 'lam'

    Returns
    -------
    stat : dictionary
        adds 'npix', 'npix_norm', 'med', 'footprint', 'compact', 'radius', 'aspect_ratio'

    """
    if 'aspect' in ops:
        d0 = np.array([int(ops['aspect']*10), 10])
    else:
        d0 = ops['diameter']
        if isinstance(d0, int):
            d0 = [d0,d0]

    rs = masks.circle_mask(np.array([30, 30]))
    rsort = np.sort(rs.flatten())

    ncells = len(stat)
    mrs = np.zeros((ncells,))
    for k in range(0,ncells):
        stat0 = stat[k]
        ypix = stat0['ypix']
        xpix = stat0['xpix']
        lam = stat0['lam']
        # compute footprint of ROI
        y0 = np.median(ypix)
        x0 = np.median(xpix)

        # compute compactness of ROI
        r2 = ((ypix-y0))**2 + ((xpix-x0))**2
        r2 = r2**.5
        stat0['mrs']  = np.mean(r2)
        mrs[k] = stat0['mrs']
        stat0['mrs0'] = np.mean(rsort[:r2.size])
        stat0['compact'] = stat0['mrs'] / (1e-10+stat0['mrs0'])
        stat0['med']  = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        stat0['npix'] = xpix.size

        if 'footprint' not in stat0:
            stat0['footprint'] = 0
        if 'med' not in stat:
            stat0['med'] = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        if 'radius' not in stat0:
            radius = utils.fitMVGaus(ypix / d0[0], xpix / d0[1], lam, 2)[2]
            stat0['radius'] = radius[0] * d0.mean()
            stat0['aspect_ratio'] = 2 * radius[0]/(.01 + radius[0] + radius[1])

    npix = np.array([stat[n]['npix'] for n in range(len(stat))]).astype('float32')
    npix /= np.mean(npix[:100])

    mmrs = np.nanmedian(mrs[:100])
    for n in range(len(stat)):
        stat[n]['mrs'] = stat[n]['mrs'] / (1e-10+mmrs)
        stat[n]['npix_norm'] = npix[n]
    stat = np.array(stat)

    return stat