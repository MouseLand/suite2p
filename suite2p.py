import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import time, os
import register, dcnv, celldetect


def default_ops():
    ops = {
        'diameter':12, # this is the main parameter for cell detection
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)           
        'data_path': 'H:/DATA/2017-10-13/',
        'subfolders': ('4'),
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
        'nimg_init': 400, # subsampled frames for finding reference image        
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
        'diameter': 12
      }
    return ops

def main(ops):    
    # copy tiff to a binary
    ops1 = register.tiff_to_binary(ops)
    # register tiff
    ops1 = register.register_binary(ops1)

    for ops in ops1:
        # get SVD components
        U,sdmov      = celldetect.getSVDdata(ops)
        # neuropil projections
        S, StU , StS = celldetect.getStU(ops, U)
        # get ROIs
        ops, stat, cell_masks, neuropil_masks, mPix, mLam = celldetect.sourcery(ops, U, S, StU, StS)
        # extract fluorescence and neuropil
        F, Fneu = celldetect.extractF(ops, stat, cell_masks, neuropil_masks, mPix, mLam)
        # deconvolve fluorescence
        spks = dcnv.oasis(F - ops['neucoeff'] * Fneu, ops1[0])
        # save results
        np.save(ops['ops_path'], spks)
        fpath = ops['save_path']
        np.save(os.path.join(fpath,'F.npy'), F)
        np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
        np.save(os.path.join(fpath,'spks.npy'), spks)        
        np.save(os.path.join(fpath,'stat.npy'), stat)        
    
    return ops1