import time
import numpy as np
from pathlib import Path

from . import sourcery, sparsedetect, chan2detect
from .stats import roi_stats
from .masks import create_cell_mask, create_neuropil_masks, create_cell_pix
from ..io.binary import BinaryFile
from ..classification import classify


def detect(ops, classfile: Path):
    if 'aspect' in ops:
        dy, dx = int(ops['aspect'] * 10), 10
    else:
        d0 = ops['diameter']
        dy, dx = (d0, d0) if isinstance(d0, int) else d0

    t0 = time.time()
    bin_size = int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
    print('Binning movie in chunks of length %2.2d' % bin_size)
    with BinaryFile(read_filename=ops['reg_file'], Ly=ops['Ly'], Lx=ops['Lx']) as f:
        mov = f.bin_movie(
            bin_size=bin_size,
            bad_frames=ops.get('badframes'),
            y_range=ops['yrange'],
            x_range=ops['xrange'],
        )
    print('Binned movie [%d,%d,%d], %0.2f sec.' % (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))


    stats = select_rois(
        mov=mov,
        dy=dy,
        dx=dx,
        Ly=ops['Ly'],
        Lx=ops['Lx'],
        max_overlap=ops['max_overlap'],
        sparse_mode=ops['sparse_mode'],
        classfile=classfile,
        ops=ops
    )
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

def select_rois(mov: np.ndarray, dy: int, dx: int, Ly: int, Lx: int, max_overlap: float, sparse_mode: bool, classfile: Path, ops):

    t0 = time.time()
    if sparse_mode:
        ops.update({'Lyc': mov.shape[1], 'Lxc': mov.shape[2]})
        new_ops, stats = sparsedetect.sparsery(
            mov=mov,
            high_pass=ops['high_pass'],
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
    stats = np.array(stats)

    if ops['preclassify'] > 0:
        stats =  roi_stats(stats, dy, dx, Ly, Lx)
        if len(stats) == 0:
            iscell = np.zeros((0, 2))
        else:
            iscell = classify(stat=stats, classfile=classfile)
        np.save(Path(ops['save_path']).joinpath('iscell.npy'), iscell)
        ic = (iscell[:,0]>ops['preclassify']).flatten().astype(np.bool)
        stats = stats[ic]
        print('Preclassify threshold %0.2f, %d ROIs removed' % (ops['preclassify'], (~ic).sum()))
        
    # add ROI stats to stats
    stats = roi_stats(stats, dy, dx, Ly, Lx, max_overlap=max_overlap)

    print('After removing overlaps, %d ROIs remain' % (len(stats)))
    return stats

