import os
import pathlib
import time

import numpy as np
from scipy import stats, signal

import suite2p
from .. import classification
from ..detection import detect
from . import chan2detect



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
        (optional 'reg_file_chan2', 'chan2_thres')
        
    cell_masks : list
        each is a tuple where first element are cell pixels (flattened), and
        second element are pixel weights normalized to sum 1 (lam)

    neuropil_masks : 2D array
        size [ncells x npixels] where weights of each mask are elements

    reg_file : string
        path to registered binary file

    Returns
    ----------------

    F : float, 2D array
        size [ROIs x time]

    Fneu : float, 2D array
        size [ROIs x time]

    ops : dictionaray
        adds 'meanImg'

    """
    t0=time.time()
    nimgbatch = min(ops['batch_size'], 1000)
    nframes = int(ops['nframes'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = neuropil_masks.shape[0]
    
    F    = np.zeros((ncells, nframes),np.float32)
    Fneu = np.zeros((ncells, nframes),np.float32)

    reg_file = open(reg_file, 'rb')
    nimgbatch = int(nimgbatch)
    block_size = Ly*Lx*nimgbatch*2
    ix = 0
    data = 1

    ops['meanImg'] = np.zeros((Ly,Lx))
    k=0
    while data is not None:
        buff = reg_file.read(block_size)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        nimg = int(np.floor(data.size / (Ly*Lx)))
        if nimg == 0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        inds = ix+np.arange(0,nimg,1,int)
        ops['meanImg'] += data.mean(axis=0)
        data = np.reshape(data, (nimg,-1))

        # extract traces and neuropil
        for n in range(ncells):
            F[n,inds] = np.dot(data[:, cell_masks[n][0]], cell_masks[n][1])
            #Fneu[n,inds] = np.mean(data[neuropil_masks[n,:], :], axis=0)
        Fneu[:,inds] = np.dot(neuropil_masks , data.T)
        ix += nimg
        k += 1
    print('Extracted fluorescence from %d ROIs in %d frames, %0.2f sec.'%(ncells, ops['nframes'], time.time()-t0))
    ops['meanImg'] /= k

    reg_file.close()
    return F, Fneu, ops

def extract_traces_from_masks(ops, cell_masks, neuropil_masks):
    """ extracts activity from ops['reg_file'] using masks in stat
    
    computes fluorescence F as sum of pixels weighted by 'lam'

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'reg_file', 'neucoeff', 'ops_path', 
        'save_path', 'sparse_mode', 'nframes', 'batch_size'
        (optional 'reg_file_chan2', 'chan2_thres')


    Returns
    ----------------

    F : float, 2D array
        size [ROIs x time]

    Fneu : float, 2D array
        size [ROIs x time]

    F_chan2 : float, 2D array
        size [ROIs x time]

    Fneu_chan2 : float, 2D array
        size [ROIs x time]

    ops : dictionaray
        adds 'meanImg' (optional 'meanImg_chan2')

    stat : array of dicts
        adds 'skew', 'std'    

    """

    F,Fneu,ops = extract_traces(ops, cell_masks, neuropil_masks, ops['reg_file'])
    if 'reg_file_chan2' in ops:
        F_chan2, Fneu_chan2, ops2 = extract_traces(ops.copy(), cell_masks, neuropil_masks, ops['reg_file_chan2'])
        ops['meanImg_chan2'] = ops2['meanImg_chan2']
    else:
        F_chan2, Fneu_chan2 = [], []

    return F, Fneu, F_chan2, Fneu_chan2, ops

def detect_and_extract(ops, stat=None):
    """ detects ROIs, computes fluorescence, applies default classifier and saves to *.npy

    if stat is None, ROIs are computed from 'reg_file'

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'reg_file', 'neucoeff', 'ops_path', 
        'save_path', 'sparse_mode', 'nframes', 'batch_size'
        (optional 'reg_file_chan2', 'chan2_thres')

    stat : array of dicts (optional, default None)
        'lam' - pixel weights, 'ypix' - pixels in y, 'xpix' - pixels in x
        
    Returns
    ----------------

    ops : dictionaray
        adds 'meanImg' (optional 'meanImg_chan2')

    """
    cell_pix, cell_masks, neuropil_masks, stat =detect.main_detect(ops, stat)
    F, Fneu, F_chan2, Fneu_chan2, ops = extract_traces_from_masks(ops, cell_masks, neuropil_masks)
    # subtract neuropil
    dF = F - ops['neucoeff'] * Fneu

    # compute activity statistics for classifier
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]
        stat[k]['std']  = sd[k]

    # apply default classifier 
    if len(stat) > 0:
        user_dir = pathlib.Path.home().joinpath('.suite2p')
        classfile = user_dir.joinpath('classifiers', 'classifier_user.npy')
        if not os.path.isfile(classfile):
            s2p_dir = pathlib.Path(suite2p.__file__).parent
            classfile = os.fspath(s2p_dir.joinpath('classifiers', 'classifier.npy'))
        print('NOTE: applying classifier %s'%classfile)
        iscell = classification.Classifier(classfile, keys=['npix_norm', 'compact', 'skew']).run(stat)
        if 'preclassify' in ops and ops['preclassify'] > 0.0:
            ic = (iscell[:,0]>ops['preclassify']).flatten().astype(np.bool)
            stat = stat[ic]
            iscell = iscell[ic]
            print('After classification with threshold %0.2f, %d ROIs remain'%(ops['preclassify'], len(stat)))
        else:
            ic = np.ones(len(stat), np.bool)
    else:
        iscell = np.zeros((0,2))
    fpath = ops['save_path']
    np.save(os.path.join(fpath,'iscell.npy'), iscell)
    np.save(os.path.join(fpath,'stat.npy'), stat)

    # if second channel, detect bright cells in second channel
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        #redcell = np.zeros((len(stat),2))
        np.save(os.path.join(fpath, 'redcell.npy'), redcell[ic])
        np.save(os.path.join(fpath, 'F_chan2.npy'), F_chan2[ic])
        np.save(os.path.join(fpath, 'Fneu_chan2.npy'), Fneu_chan2[ic])

    # add enhanced mean image
    ops = enhanced_mean_image(ops)
    # save ops
    np.save(ops['ops_path'], ops)
    # save results
    np.save(os.path.join(fpath,'F.npy'), F[ic])
    np.save(os.path.join(fpath,'Fneu.npy'), Fneu[ic])

    return ops


def enhanced_mean_image(ops):
    """ computes enhanced mean image and adds it to ops

    Median filters ops['meanImg'] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops['meanImgE']

    Parameters
    ----------
    ops : dictionary
        uses 'meanImg', 'aspect', 'diameter', 'yrange' and 'xrange'

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
    return ops
