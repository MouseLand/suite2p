import time
import numpy as np
from pathlib import Path
from . import sourcery, sparsedetect, chan2detect
from .stats import ROI
from .masks import count_overlaps, remove_overlappers, create_cell_masks, create_neuropil_masks, create_cell_pix
from .utils import norm_by_average


def main_detect(ops):
    if 'aspect' in ops:
        dy, dx = int(ops['aspect'] * 10), 10
    else:
        d0 = ops['diameter']
        dy, dx = (d0, d0) if isinstance(d0, int) else d0
    stats = select_rois(dy=dy, dx=dx, Ly=ops['Ly'], Lx=ops['Lx'], max_overlap=ops['max_overlap'], sparse_mode=ops['sparse_mode'], ops=ops)
    # extract fluorescence and neuropil
    t0 = time.time()
    cell_pix = create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap'])
    cell_masks = create_cell_masks(stats, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap'])
    neuropil_masks = create_neuropil_masks(
        ypixs=[stat['ypix'] for stat in stats],
        xpixs=[stat['xpix'] for stat in stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=ops['inner_neuropil_radius'],
        min_neuropil_pixels=ops['min_neuropil_pixels'],
    )

    print('Masks made in %0.2f sec.' % (time.time() - t0))

    ic = np.ones(len(stats), np.bool)
    # if second channel, detect bright cells in second channel
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stats)
        np.save(Path(ops['save_path']).joinpath('redcell.npy'), redcell[ic])
    return cell_pix, cell_masks, neuropil_masks, stats, ops


def select_rois(dy: int, dx: int, Ly: int, Lx: int, max_overlap: float, sparse_mode: bool, ops):
    t0 = time.time()
    if sparse_mode:
        ops, stats = sparsedetect.sparsery(ops)
    else:
        ops, stats = sourcery.sourcery(ops)
    print('Found %d ROIs, %0.2f sec' % (len(stats), time.time() - t0))

    rois = [ROI(ypix=stat['ypix'], xpix=stat['xpix'], lam=stat['lam'], dx=dx, dy=dy) for stat in stats]
    mrs_normeds = norm_by_average([roi.mean_r_squared for roi in rois], estimator=np.nanmedian, offset=1e-10, first_n=100)
    npix_normeds = norm_by_average([roi.n_pixels for roi in rois], first_n=100)
    for roi, mrs_normed, npix_normed, stat in zip(rois, mrs_normeds, npix_normeds, stats):
        stat.update({
            'mrs': mrs_normed,
            'mrs0': roi.mean_r_squared0,
            'compact': roi.mean_r_squared_compact,
            'med': list(roi.median_pix),
            'npix': roi.n_pixels,
            'npix_norm': npix_normed,
            'footprint': 0 if 'footprint' not in stat else stat['footprint'],
        })
        if 'radius' not in stat:
            stat.update({
                'radius': roi.radius,
                'aspect_ratio': roi.aspect_ratio,
            })

    stats = np.array(stats)

    ypixs = [stat['ypix'] for stat in stats]
    xpixs = [stat['xpix'] for stat in stats]
    n_overlaps = count_overlaps(Ly=Ly, Lx=Lx, ypixs=ypixs, xpixs=xpixs)
    for stat in stats:
        stat['overlap'] = n_overlaps[stat['ypix'], stat['xpix']] > 1

    ix = remove_overlappers(ypixs=ypixs, xpixs=xpixs, max_overlap=max_overlap, Ly=Ly, Lx=Lx)
    stats = [stats[i] for i in ix]
    print('After removing overlaps, %d ROIs remain' % (len(stats)))
    return stats


