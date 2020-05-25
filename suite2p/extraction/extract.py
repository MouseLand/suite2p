import os
import pathlib
import time

import numpy as np
from scipy import stats, signal

import suite2p
from . import masks
from .. import classification
from . import sourcery
from . import sparsedetect
from . import chan2detect
from . import utils



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

def compute_masks_and_extract_traces(ops, stat):
    """ extracts activity from ops['reg_file'] using masks in stat
    
    computes fluorescence F as sum of pixels weighted by 'lam'

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'reg_file', 'neucoeff', 'ops_path', 
        'save_path', 'sparse_mode', 'nframes', 'batch_size'
        (optional 'reg_file_chan2', 'chan2_thres')
        
    stat : array of dicts
        'ipix', 'lam'

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
    t0=time.time()
    cell_pix, cell_masks = masks.create_cell_masks(stat, ops['Ly'], ops['Lx'], ops['allow_overlap'])
    neuropil_masks = masks.create_neuropil_masks(ops, stat, cell_pix)
    Ly=ops['Ly']
    Lx=ops['Lx']
    neuropil_masks = np.reshape(neuropil_masks, (-1,Ly*Lx))
    print('Masks made in %0.2f sec.'%(time.time()-t0))

    F,Fneu,ops = extract_traces(ops, cell_masks, neuropil_masks, ops['reg_file'])
    if 'reg_file_chan2' in ops:
        F_chan2, Fneu_chan2, ops2 = extract_traces(ops.copy(), cell_masks, neuropil_masks, ops['reg_file_chan2'])
        ops['meanImg_chan2'] = ops2['meanImg_chan2']
    else:
        F_chan2, Fneu_chan2 = [], []

    return F, Fneu, F_chan2, Fneu_chan2, ops, stat

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
    t0=time.time()
    if stat is None:
        if ops['sparse_mode']:
            ops, stat = sparsedetect.sparsery(ops)
        else:
            ops, stat = sourcery.sourcery(ops)
        print('Found %d ROIs, %0.2f sec'%(len(stat), time.time()-t0))
    stat = roi_stats(ops, stat)
    
    stat = masks.get_overlaps(stat,ops)
    stat, ix = masks.remove_overlappers(stat, ops, ops['Ly'], ops['Lx'])
    print('After removing overlaps, %d ROIs remain'%(len(stat))) 

    # extract fluorescence and neuropil
    F, Fneu, F_chan2, Fneu_chan2, ops, stat = compute_masks_and_extract_traces(ops, stat)
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


def roi_stats(ops, stat):
    """ computes statistics of ROIs

    Parameters
    ----------
    ops : dictionary
        'aspect', 'diameter'

    stat : dictionary
        'ypix', 'xpix', 'lam'

    Returns
    -------
    stat : dictionary
        adds 'npix', 'npix_norm', 'med', 'footprint', 'compact', 'radius', 'aspect_ratio'

    """
    if 'aspect' in ops:
        d0 = np.array([int(ops['aspect']*10), 10])
    else:
        d0 = ops['diameter']
        if isinstance(d0, int):
            d0 = [d0,d0]

    rs = masks.circle_mask(np.array([30, 30]))
    rsort = np.sort(rs.flatten())

    ncells = len(stat)
    mrs = np.zeros((ncells,))
    for k in range(0,ncells):
        stat0 = stat[k]
        ypix = stat0['ypix']
        xpix = stat0['xpix']
        lam = stat0['lam']
        # compute footprint of ROI
        y0 = np.median(ypix)
        x0 = np.median(xpix)

        # compute compactness of ROI
        r2 = ((ypix-y0))**2 + ((xpix-x0))**2
        r2 = r2**.5
        stat0['mrs']  = np.mean(r2)
        mrs[k] = stat0['mrs']
        stat0['mrs0'] = np.mean(rsort[:r2.size])
        stat0['compact'] = stat0['mrs'] / (1e-10+stat0['mrs0'])
        stat0['med']  = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        stat0['npix'] = xpix.size

        if 'footprint' not in stat0:
            stat0['footprint'] = 0
        if 'med' not in stat:
            stat0['med'] = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        if 'radius' not in stat0:
            radius = utils.fitMVGaus(ypix / d0[0], xpix / d0[1], lam, 2)[2]
            stat0['radius'] = radius[0] * d0.mean()
            stat0['aspect_ratio'] = 2 * radius[0]/(.01 + radius[0] + radius[1])

    npix = np.array([stat[n]['npix'] for n in range(len(stat))]).astype('float32')
    npix /= np.mean(npix[:100])

    mmrs = np.nanmedian(mrs[:100])
    for n in range(len(stat)):
        stat[n]['mrs'] = stat[n]['mrs'] / (1e-10+mmrs)
        stat[n]['npix_norm'] = npix[n]
    stat = np.array(stat)

    return stat