import numpy as np
import time, os
from suite2p import register, dcnv, celldetect
from scipy import stats
from multiprocessing import Pool

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def default_ops():
    ops = {
        'save_path0': [],
        'diameter':12, # this is the main parameter for cell detection
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)                   
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane  
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'look_one_level_down': False,        
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter 
        'prctile_baseline': 8.,# smoothing constant for gaussian filter        
        'neucoeff': .7,  # neuropil coefficient 
        'neumax': 1.,  # maximum neuropil coefficient (not implemented)
        'niterneu': 5, # number of iterations when the neuropil coefficient is estimated (not implemented)
        'maxregshift': 0.,
        'subpixel' : 10,
        'batch_size': 200, # number of frames per batch
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value        
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value        
        'nimg_init': 200, # subsampled frames for finding reference image        
        'navg_frames_svd': 5000,
        'nsvd_for_roi': 1000,
        'ratio_neuropil': 5,
        'tile_factor': 1,        
        'threshold_scaling': 1,
        'Vcorr': [],
        'allow_overlap': False,
        'inner_neuropil_radius': 2, 
        'outer_neuropil_radius': np.inf, 
        'min_neuropil_pixels': 350, 
        'ratio_neuropil_to_cell': 3,     
        'nframes': 1,
        'diameter': 12,
        'reg_tif': False,
        'max_iterations': 10
      }
    return ops

def get_cells(ops):
    i0 = tic()
    ops, stat = celldetect.sourcery(ops)
    print('time %4.4f. Found %d ROIs'%(toc(i0), len(stat)))
    # extract fluorescence and neuropil
    F, Fneu = celldetect.extractF(ops, stat)
    print('time %4.4f. Extracted fluorescence from %d ROIs'%(toc(i0), len(stat)))
    # subtract neuropil
    dF = F - ops['neucoeff'] * Fneu        
    # deconvolve fluorescence
    spks = dcnv.oasis(dF, ops)
    print('time %4.4f. Detected spikes in %d ROIs'%(toc(i0), len(stat)))
    # compute activity statistics for classifier
    sk = stats.skew(dF, axis=1)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]         
    # save results
    np.save(ops['ops_path'], ops)
    fpath = ops['save_path']
    np.save(os.path.join(fpath,'F.npy'), F)
    np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath,'spks.npy'), spks)        
    np.save(os.path.join(fpath,'stat.npy'), stat)            
    print('results saved to %s'%ops['save_path'])
    
    return ops

def run_s2p(ops={},db={}):
    i0 = tic()
    
    ops = {**ops, **db}
    
    if 'save_path0' not in ops or len(ops['save_path0'])==0:
        ops['save_path0'] = ops['data_path'][0]

    # check if there are files already registered
    fpathops1 = os.path.join(ops['save_path0'], 'suite2p', 'ops1.npy')
    files_found_flag = True
    if os.path.isfile(fpathops1): 
        ops1 = np.load(fpathops1)
        files_found_flag = True
        for i,op in enumerate(ops1):
            files_found_flag &= os.path.isfile(op['reg_file']) 
            # use the new options
            ops1[i] = {**op, **ops} 
    else:
        files_found_flag = False
    
    if not files_found_flag:
        # get default options
        ops0 = default_ops()
        # combine with user options
        ops = {**ops0, **ops} 
        # copy tiff to a binary
        ops1 = register.tiff_to_binary(ops)
        print('time %4.4f. Wrote tifs to binaries for %d planes'%(toc(i0), len(ops1)))
        # register tiff
        ops1 = register.register_binary(ops1)
        # save ops1
        np.save(fpathops1, ops1)
        print('time %4.4f. Registration complete'%toc(i0))
    else:
        print('found ops1 and pre-registered binaries')
        print('overwriting ops1 with new ops')
        print('skipping registration...')
    
    if len(ops1)>1 and ops['num_workers_roi']>=0:
        if ops['num_workers_roi']==0:
            ops['num_workers_roi'] = len(ops1)            
        with Pool(ops['num_workers_roi']) as p:
            results = p.map(get_cells, ops1)
        for k in range(len(ops1)):
            ops1[k] = results[k]
    else:
        for k in range(len(ops1)):
            ops1[k] = get_cells(ops1[k])
    
    # save final ops1 with all planes
    np.save(fpathops1, ops1)
    
    print('finished all tasks in %4.4f sec'%toc(i0))
    
    return ops1