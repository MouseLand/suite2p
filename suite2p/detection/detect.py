import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import sourcery, sparsedetect, chan2detect, utils
from .stats import roi_stats
from .denoise import pca_denoise
from ..io.binary import BinaryFile
from ..classification import classify, user_classfile

try:
    from . import anatomical
    CELLPOSE_INSTALLED = True
except Exception as e:
    print('Warning: cellpose did not import')
    print(e)
    print('cannot use anatomical mode, but otherwise suite2p will run normally')
    CELLPOSE_INSTALLED = False


def detect(ops, classfile=None):
    
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
    print('Binned movie [%d,%d,%d] in %0.2f sec.' % (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    if ops.get('inverted_activity', False):
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()

    if ops.get('denoise', 1):
        mov = pca_denoise(mov, block_size=[ops['block_size'][0]//2, ops['block_size'][1]//2],
                            n_comps_frac = 0.5)

    t0 = time.time()
    if ops.get('anatomical_only', 0) and not CELLPOSE_INSTALLED:
        print('~~~ tried to import cellpose to run anatomical but failed, install with: ~~~')
        print('$ pip install cellpose')

    if ops.get('anatomical_only', 0) > 0 and CELLPOSE_INSTALLED:
        print('>>>> CELLPOSE finding masks in ' + ['max_proj / mean_img', 'mean_img', 'enhanced_mean_img'][int(ops['anatomical_only'])-1])
        stat = anatomical.select_rois(
                    ops=ops,
                    mov=mov,
                    dy=dy,
                    dx=dx,
                    Ly=ops['Ly'],
                    Lx=ops['Lx'],
                    diameter=ops['diameter'])
        
    else:            
        stat = select_rois(
            ops=ops,
            mov=mov,
            dy=dy,
            dx=dx,
            Ly=ops['Ly'],
            Lx=ops['Lx'],
            max_overlap=ops['max_overlap'],
            sparse_mode=ops['sparse_mode'],
            do_crop=ops['soma_crop'],
            classfile=classfile,
        )

    # if second channel, detect bright cells in second channel
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops['save_path']).joinpath('redcell.npy'), redcell)

    return ops, stat

def select_rois(ops: Dict[str, Any], mov: np.ndarray, dy: int, dx: int, Ly: int, Lx: int, 
                max_overlap: float = True, sparse_mode: bool = True, do_crop: bool=True,
                classfile: Path = None):
    
    t0 = time.time()
    if sparse_mode:
        ops.update({'Lyc': mov.shape[1], 'Lxc': mov.shape[2]})
        new_ops, stat = sparsedetect.sparsery(
            mov=mov,
            high_pass=ops['high_pass'],
            neuropil_high_pass=ops['spatial_hp_detect'],
            batch_size=ops['batch_size'],
            spatial_scale=ops['spatial_scale'],
            threshold_scaling=ops['threshold_scaling'],
            max_iterations=250 * ops['max_iterations'],
            yrange=ops['yrange'],
            xrange=ops['xrange'],
            percentile=ops.get('active_percentile', 0.0),
        )
        ops.update(new_ops)
    else:
        ops, stat = sourcery.sourcery(mov=mov, ops=ops)

    print('Detected %d ROIs, %0.2f sec' % (len(stat), time.time() - t0))
    stat = np.array(stat)
    
    if len(stat)==0:
        raise ValueError("no ROIs were found -- check registered binary and maybe change spatial scale")

    if ops['preclassify'] > 0:
        if classfile is None:
            print(f'NOTE: Applying user classifier at {str(user_classfile)}')
            classfile = user_classfile

        stat =  roi_stats(stat, dy, dx, Ly, Lx, do_crop=do_crop)
        if len(stat) == 0:
            iscell = np.zeros((0, 2))
        else:
            iscell = classify(stat=stat, classfile=classfile)
        np.save(Path(ops['save_path']).joinpath('iscell.npy'), iscell)
        ic = (iscell[:,0]>ops['preclassify']).flatten().astype('bool')
        stat = stat[ic]
        print('Preclassify threshold %0.2f, %d ROIs removed' % (ops['preclassify'], (~ic).sum()))
            
    # add ROI stat to stat
    stat = roi_stats(stat, dy, dx, Ly, Lx, max_overlap=max_overlap, do_crop=do_crop)

    print('After removing overlaps, %d ROIs remain' % (len(stat)))
    return stat

