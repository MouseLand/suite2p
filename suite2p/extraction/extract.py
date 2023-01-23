import os
import time

import numpy as np
from numba import prange, njit, jit, int64, float32
from numba.typed import List
from scipy import stats, signal
from .masks import create_masks
from ..io import BinaryRWFile
from .. import default_ops

def extract_traces(f_in, cell_masks, neuropil_masks, batch_size=500):
    """ extracts activity from f_in using masks in stat and neuropil_masks
    
    computes fluorescence F as sum of pixels weighted by 'lam'
    computes neuropil fluorescence Fneu as sum of pixels in neuropil_masks

    data is from reg_file ops['batch_size'] by pixels:
    .. code-block:: python
        F[n] = data[:, stat[n]['ipix']] @ stat[n]['lam']
        Fneu = neuropil_masks @ data.T

    Parameters
    ----------------

    f_in : np.ndarray or io.BinaryRWFile object
        size n_frames, Ly, Lx

        
    cell_masks : list
        each is a tuple where first element are cell pixels (flattened), and
        second element are pixel weights normalized to sum 1 (lam)

    neuropil_masks : list
        each element is neuropil pixels in (Ly*Lx) coordinates
        GOING TO BE DEPRECATED: size [ncells x npixels] where weights of each mask are elements

    batch_size : int
        function will run with at most batch size of 1000
    
    Returns
    ----------------

    F : float, 2D array
        size [ROIs x time]

    Fneu : float, 2D array
        size [ROIs x time]

    ops : dictionaray

    """
    n_frames, Ly, Lx = f_in.shape
    t0=time.time()
    batch_size = min(batch_size, 1000)
    ncells = len(cell_masks)
    
    F    = np.zeros((ncells, n_frames),np.float32)
    Fneu = np.zeros((ncells, n_frames),np.float32)

    batch_size = int(batch_size)
    
    cell_ipix, cell_lam = List(), List()
    [cell_ipix.append(cell_mask[0].astype(np.int64)) for cell_mask in cell_masks]
    [cell_lam.append(cell_mask[1].astype(np.float32)) for cell_mask in cell_masks]

    #cell_ipix = [int64(cell_mask[0]) for cell_mask in cell_masks]
    #cell_lam = [float32(cell_mask[1]) for cell_mask in cell_masks]

    if neuropil_masks is not None:
        neuropil_ipix = List()
        if isinstance(neuropil_masks, np.ndarray) and neuropil_masks.shape[1] == Ly*Lx:
            [neuropil_ipix.append(np.nonzero(neuropil_mask)[0]) for neuropil_mask in neuropil_masks]
        else:
            [neuropil_ipix.append(neuropil_mask.astype(np.int64)) for neuropil_mask in neuropil_masks]
        neuropil_npix = np.array([len(neuropil_ipixi) for neuropil_ipixi in neuropil_ipix]).astype(np.float32)
    else:
        neuropil_ipix = None

    ix = 0
    for k in np.arange(0, n_frames, batch_size):
        data = f_in[k : min(k + batch_size, n_frames)].astype('float32')
        nimg = data.shape[0]
        if nimg == 0:
            break
        inds = ix+np.arange(0,nimg,1,int)
        data = np.reshape(data, (nimg,-1)).astype(np.float32)
        Fi = np.zeros((ncells, data.shape[0]), np.float32)
        
        # extract traces and neuropil
        
        # (WITHOUT NUMBA)
        #for n in range(ncells):
        #    F[n,inds] = np.dot(data[:, cell_masks[n][0]], cell_masks[n][1])
        #Fneu[:,inds] = np.dot(neuropil_masks , data.T)

        # WITH NUMBA
        F[:,inds] = matmul_traces(Fi, data, cell_ipix, cell_lam)
        if neuropil_ipix is not None:
            Fneu[:,inds] = matmul_neuropil(Fi, data, neuropil_ipix, neuropil_npix)

        ix += nimg
    print('Extracted fluorescence from %d ROIs in %d frames, %0.2f sec.'%(ncells, n_frames, time.time()-t0))
    return F, Fneu

@njit(parallel=True)
def matmul_traces(Fi, data, cell_ipix, cell_lam):
    ncells = Fi.shape[0]
    for n in prange(ncells):
        Fi[n] = np.dot(data[:, cell_ipix[n]], cell_lam[n])
    return Fi

@njit(parallel=True)
def matmul_neuropil(Fi, data, neuropil_ipix, neuropil_npix):
    ncells = Fi.shape[0]
    for n in prange(ncells):
        Fi[n] = data[:, neuropil_ipix[n]].sum(axis=1) / neuropil_npix[n]
    return Fi


def extract_traces_from_masks(ops, cell_masks, neuropil_masks):
    """ extract fluorescence from both channels 
    
    also used in drawroi.py
    
    """
    batch_size=ops['batch_size']
    F_chan2, Fneu_chan2 = [], []
    with BinaryRWFile(Ly=ops['Ly'], Lx=ops['Lx'],
                      filename=ops['reg_file']) as f:    
        F, Fneu = extract_traces(f, cell_masks, neuropil_masks, batch_size=batch_size)
    if 'reg_file_chan2' in ops:
        with BinaryRWFile(Ly=ops['Ly'], Lx=ops['Lx'],
                          filename=ops['reg_file_chan2']) as f:    
            F_chan2, Fneu_chan2 = extract_traces(f, cell_masks, neuropil_masks, batch_size=batch_size)
    return F, Fneu, F_chan2, Fneu_chan2

def extraction_wrapper(stat, f_reg, f_reg_chan2=None, cell_masks=None, neuropil_masks=None, ops=default_ops()):
    """ 
    Main extraction function
    creates masks, computes fluorescence

    Parameters
    ----------------

    stat : array of dicts

    f_reg : array of functional frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx

    f_reg_chan2 : array of anatomical frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx


    Returns
    ----------------

    stat : list of dictionaries
        adds keys 'skew' and 'std'

    F : fluorescence of functional channel

    F_neu : neuropil of functional channel

    F_chan2 : fluorescence of anatomical channel

    F_neu_chan2 : neuropil of anatomical channel

    """
    n_frames, Ly, Lx = f_reg.shape
    batch_size=ops['batch_size']
    if cell_masks is None:
        t10 = time.time()
        cell_masks, neuropil_masks0 = create_masks(stat, Ly, Lx, ops)
        if neuropil_masks is None:
            neuropil_masks = neuropil_masks0
        print('Masks created, %0.2f sec.' % (time.time() - t10))    

    F, Fneu = extract_traces(f_reg, cell_masks, neuropil_masks, batch_size=batch_size)
    if f_reg_chan2 is not None:
        F_chan2, Fneu_chan2 = extract_traces(f_reg_chan2, cell_masks, neuropil_masks, batch_size=batch_size)
    else:
        F_chan2, Fneu_chan2 = [], []

    # subtract neuropil
    dF = F - ops['neucoeff'] * Fneu

    # compute activity statistics for classifier
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]
        stat[k]['std'] = sd[k]
        if not neuropil_masks is None:
            stat[k]['neuropil_mask'] = neuropil_masks[k]
    
    return stat, F, Fneu, F_chan2, Fneu_chan2

def create_masks_and_extract(ops, stat, cell_masks=None, neuropil_masks=None):
    """ creates masks, computes fluorescence, and saves stat, F, and Fneu to .npy

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'reg_file', 'neucoeff', 'ops_path', 
        'save_path', 'sparse_mode', 'nframes', 'batch_size'
        (optional 'reg_file_chan2', 'chan2_thres')

    stat : array of dicts

    Returns
    ----------------

    stat : list of dictionaries
        adds keys 'skew' and 'std'

    F : fluorescence of functional channel

    F_neu : neuropil of functional channel

    F_chan2 : fluorescence of anatomical channel

    F_neu_chan2 : neuropil of anatomical channel

    """

    if len(stat) == 0:
        raise ValueError("stat array should not be of length 0 (no ROIs were found)")

    # create cell and neuropil masks
    Ly, Lx = ops['Ly'], ops['Lx']
    reg_file = ops['reg_file']
    reg_file_alt = ops.get('reg_file_chan2', ops['reg_file'])
    with BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file) as f_in,\
         BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file_alt) as f_in_chan2:         
        if ops['nchannels'] == 1:
            f_in_chan2.close() 
            f_in_chan2 = None
        
        stat, F, Fneu, F_chan2, Fneu_chan2 = extraction_wrapper(stat, f_in, 
                                                                f_reg_chan2=f_in_chan2, 
                                                                cell_masks=cell_masks,
                                                                neuropil_masks=neuropil_masks,
                                                                ops=ops)
    
    return stat, F, Fneu, F_chan2, Fneu_chan2
    

def enhanced_mean_image(ops):
    """ computes enhanced mean image and adds it to ops

    Median filters ops['meanImg'] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops['meanImgE']

    Parameters
    ----------
    ops : dictionary
        uses 'meanImg', 'aspect', 'spatscale_pix', 'yrange' and 'xrange'

    Returns
    -------
        ops : dictionary
            'meanImgE' field added

    """

    I = ops['meanImg'].astype(np.float32)
    if 'spatscale_pix' not in ops:
        if isinstance(ops['diameter'], int):
            diameter = np.array([ops['diameter'], ops['diameter']])
        else:
            diameter = np.array(ops['diameter'])
        if diameter[0]==0:
            diameter[:] = 12
        ops['spatscale_pix'] = diameter[1]
        ops['aspect'] = diameter[0]/diameter[1]

    diameter = 4*np.ceil(np.array([ops['spatscale_pix'] * ops['aspect'], ops['spatscale_pix']])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
        ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg
    print('added enhanced mean image')
    return ops
