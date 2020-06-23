import time
import numpy as np
from pathlib import Path
from . import sourcery, sparsedetect, masks, chan2detect
from .stats import roi_stats


def main_detect(ops, stat=None):
    stat = select_rois(ops, stat)
    # extract fluorescence and neuropil
    t0 = time.time()
    cell_pix, cell_masks, neuropil_masks = make_masks(ops, stat)
    print('Masks made in %0.2f sec.' % (time.time() - t0))

    ic = np.ones(len(stat), np.bool)
    # if second channel, detect bright cells in second channel
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops['save_path']).joinpath('redcell.npy'), redcell[ic])
    return cell_pix, cell_masks, neuropil_masks, stat, ops


def select_rois(ops, stats=None):
    t0 = time.time()
    if stats is None:
        if ops['sparse_mode']:
            ops, stats = sparsedetect.sparsery(ops)
        else:
            ops, stats = sourcery.sourcery(ops)
        print('Found %d ROIs, %0.2f sec' % (len(stats), time.time() - t0))

    if 'aspect' in ops:
        d0 = np.array([int(ops['aspect']*10), 10])
    else:
        d0 = ops['diameter']
        if isinstance(d0, int):
            d0 = [d0,d0]
    stats = roi_stats(d0, stats)

    ypixs = [stat['ypix'] for stat in stats]
    xpixs = [stat['xpix'] for stat in stats]
    overlap_masks = masks.get_overlaps(
        overlaps=masks.count_overlaps(Ly=ops['Ly'], Lx=ops['Lx'], ypixs=ypixs, xpixs=xpixs),
        ypixs=ypixs,
        xpixs=xpixs,
    )
    for stat, overlap_mask in zip(stats, overlap_masks):
        stat['overlap'] = overlap_mask

    stats, ix = masks.remove_overlappers(stats, ops, ops['Ly'], ops['Lx'])
    print('After removing overlaps, %d ROIs remain' % (len(stats)))
    return stats


def make_masks(ops, stat):
    Ly, Lx = ops['Ly'], ops['Lx']
    cell_pix, cell_masks = masks.create_cell_masks(stat, Ly=Ly, Lx=Lx, allow_overlap=ops['allow_overlap'])
    neuropil_masks = masks.create_neuropil_masks(ops, stat, cell_pix)
    neuropil_masks = np.reshape(neuropil_masks, (-1, Ly * Lx))
    return cell_pix, cell_masks, neuropil_masks


