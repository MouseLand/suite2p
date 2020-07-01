import time
import numpy as np
from pathlib import Path
from . import sourcery, sparsedetect, chan2detect
from .stats import roi_stats
from .masks import get_overlaps, count_overlaps, remove_overlappers, make_masks


def main_detect(ops):
    if 'aspect' in ops:
        dy, dx = int(ops['aspect'] * 10), 10
    else:
        d0 = ops['diameter']
        dy, dx = (d0, d0) if isinstance(d0, int) else d0
    stat = select_rois(dy=dy, dx=dx, Ly=ops['Ly'], Lx=ops['Lx'], max_overlap=ops['max_overlap'], sparse_mode=ops['sparse_mode'], ops=ops)
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


def select_rois(dy: int, dx: int, Ly: int, Lx: int, max_overlap: float, sparse_mode: bool, ops):
    t0 = time.time()
    if sparse_mode:
        ops, stats = sparsedetect.sparsery(ops)
    else:
        ops, stats = sourcery.sourcery(ops)
    print('Found %d ROIs, %0.2f sec' % (len(stats), time.time() - t0))

    stats = roi_stats(dy=dy, dx=dx, stats=stats)

    ypixs = [stat['ypix'] for stat in stats]
    xpixs = [stat['xpix'] for stat in stats]
    overlap_masks = get_overlaps(
        overlaps=count_overlaps(Ly=Ly, Lx=Lx, ypixs=ypixs, xpixs=xpixs),
        ypixs=ypixs,
        xpixs=xpixs,
    )
    for stat, overlap_mask in zip(stats, overlap_masks):
        stat['overlap'] = overlap_mask

    ix = remove_overlappers(ypixs=ypixs, xpixs=xpixs, max_overlap=max_overlap, Ly=Ly, Lx=Lx)
    stats = [stats[i] for i in ix]
    print('After removing overlaps, %d ROIs remain' % (len(stats)))
    return stats


