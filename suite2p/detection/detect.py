import time
import numpy as np
from pathlib import Path
from . import sourcery, sparsedetect, chan2detect
from .stats import ROI
from .masks import create_cell_mask, create_neuropil_masks, create_cell_pix
from ..io.binary import bin_movie


def main_detect(ops):
    if 'aspect' in ops:
        dy, dx = int(ops['aspect'] * 10), 10
    else:
        d0 = ops['diameter']
        dy, dx = (d0, d0) if isinstance(d0, int) else d0

    t0 = time.time()
    bin_size = int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
    print('Binning movie in chunks of length %2.2d' % bin_size)
    mov = bin_movie(
        filename=ops['reg_file'],
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        n_frames=ops['nframes'],
        bin_size=bin_size,
        bad_frames=np.where(ops['badframes'])[0] if 'badframes' in ops else (),
        y_range=ops['yrange'],
        x_range=ops['xrange'],
    )
    ops['nbinned'] = mov.shape[0]
    print('Binned movie [%d,%d,%d], %0.2f sec.' % (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))


    stats = select_rois(mov=mov, dy=dy, dx=dx, Ly=ops['Ly'], Lx=ops['Lx'], max_overlap=ops['max_overlap'], sparse_mode=ops['sparse_mode'], ops=ops)
    # extract fluorescence and neuropil
    t0 = time.time()
    cell_pix = create_cell_pix(stats, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap'])
    cell_masks = [create_cell_mask(stat, Ly=ops['Ly'], Lx=ops['Lx'], allow_overlap=ops['allow_overlap']) for stat in stats]
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


def select_rois(mov: np.ndarray, dy: int, dx: int, Ly: int, Lx: int, max_overlap: float, sparse_mode: bool, ops):

    t0 = time.time()
    if sparse_mode:
        ops.update({'Lyc': mov.shape[1], 'Lxc': mov.shape[2]})
        new_ops, stats = sparsedetect.sparsery(
            mov=mov,
            high_pass=int(ops['high_pass']),
            neuropil_high_pass=ops['spatial_hp_detect'],
            batch_size=ops['batch_size'],
            spatial_scale=ops['spatial_scale'],
            threshold_scaling=ops['threshold_scaling'],
            max_iterations=250 * ops['max_iterations'],
            yrange=ops['yrange'],
            xrange=ops['xrange'],
        )
        ops.update(new_ops)
    else:
        ops, stats = sourcery.sourcery(mov=mov, ops=ops)
    print('Found %d ROIs, %0.2f sec' % (len(stats), time.time() - t0))

    rois = [ROI(ypix=stat['ypix'], xpix=stat['xpix'], lam=stat['lam'], dx=dx, dy=dy) for stat in stats]

    mrs_normeds = ROI.get_mean_r_squared_normed_all(rois=rois)
    npix_normeds = ROI.get_n_pixels_normed_all(rois=rois)
    n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
    keep_rois = ROI.filter_overlappers(rois=rois, overlap_image=n_overlaps, max_overlap=max_overlap)

    good_stats = []
    for keep_roi, roi, mrs_normed, npix_normed, stat in zip(keep_rois, rois, mrs_normeds, npix_normeds, stats):
        if keep_roi:
            stat.update({
                'mrs': mrs_normed,
                'mrs0': roi.mean_r_squared0,
                'compact': roi.mean_r_squared_compact,
                'med': list(roi.median_pix),
                'npix': roi.n_pixels,
                'npix_norm': npix_normed,
                'footprint': 0 if 'footprint' not in stat else stat['footprint'],
                'overlap': roi.get_overlap_image(n_overlaps),
            })
            if 'radius' not in stat:
                stat.update({
                    'radius': roi.radius,
                    'aspect_ratio': roi.aspect_ratio,
                })
            good_stats.append(stat)

    print('After removing overlaps, %d ROIs remain' % (len(good_stats)))
    return good_stats


