import os
import time

import numpy as np
from numba import prange, njit, jit, int64, float32
from numba.typed import List
from scipy import stats, signal
from .masks import create_masks
from ..io import BinaryFile

def extract_traces(ops, cell_masks, neuropil_masks, reg_file):
    """ extracts activity from reg_file using masks in stat and neuropil_masks
    
    computes fluorescence F as sum of pixels weighted by 'lam'
    computes neuropil fluorescence Fneu as sum of pixels in neuropil_masks

    data is from reg_file ops['batch_size'] by pixels:
    .. code-block:: python
        F[n] = data[:, stat[n]['ipix']] @ stat[n]['lam']
        Fneu = neuropil_masks @ data.T

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'nframes', 'batch_size'

        
    cell_masks : list
        each is a tuple where first element are cell pixels (flattened), and
        second element are pixel weights normalized to sum 1 (lam)

    neuropil_masks : list
        each element is neuropil pixels in (Ly*Lx) coordinates
        GOING TO BE DEPRECATED: size [ncells x npixels] where weights of each mask are elements

    reg_file : io.BinaryFile object
        io.BinaryFile object that has iter_frames(batch_size=ops['batch_size']) method
        

    Returns
    ----------------

    F : float, 2D array
        size [ROIs x time]

    Fneu : float, 2D array
        size [ROIs x time]

    ops : dictionaray

    """
    t0=time.time()
    nimgbatch = min(ops['batch_size'], 1000)
    nframes = int(ops['nframes'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = len(cell_masks)
    
    F    = np.zeros((ncells, nframes),np.float32)
    Fneu = np.zeros((ncells, nframes),np.float32)

    nimgbatch = int(nimgbatch)
    
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
    for k, (_, data) in enumerate(reg_file.iter_frames(batch_size=ops['batch_size'])):
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
    print('Extracted fluorescence from %d ROIs in %d frames, %0.2f sec.'%(ncells, ops['nframes'], time.time()-t0))
    reg_file.close()
    return F, Fneu, ops

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
    F_chan2, Fneu_chan2 = [], []
    with BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'],
                    read_filename=ops['reg_file']) as f:    
        F, Fneu, ops = extract_traces(ops, cell_masks, neuropil_masks, f)
    if 'reg_file_chan2' in ops:
        with BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'],
                        read_filename=ops['reg_file_chan2']) as f:    
            F_chan2, Fneu_chan2, _ = extract_traces(ops.copy(), cell_masks, neuropil_masks, f)
    return F, Fneu, F_chan2, Fneu_chan2, ops

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

    ops : dictionary

    stat : list of dictionaries
        adds keys 'skew' and 'std'

    """

    if len(stat) == 0:
        raise ValueError("stat array should not be of length 0 (no ROIs were found)")

    # create cell and neuropil masks
    if cell_masks is None:
        t10 = time.time()
        cell_masks, neuropil_masks0 = create_masks(ops, stat)
        if neuropil_masks is None:
            neuropil_masks = neuropil_masks0
        print('Masks created, %0.2f sec.' % (time.time() - t10))    

    F, Fneu, F_chan2, Fneu_chan2, ops = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
    
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
    
    return ops, stat, F, Fneu, F_chan2, Fneu_chan2


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
