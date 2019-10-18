import numpy as np
import time, os, shutil
from scipy import stats
from .. import utils
from ..classification import classifier
from ..detection import sparsedetect, chan2detect, sourcery
from . import masks

def extract_traces(ops, stat, neuropil_masks, reg_file):
    t0=time.time()
    nimgbatch = 1000
    nframes = int(ops['nframes'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = neuropil_masks.shape[0]
    k0 = time.time()

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
            F[n,inds] = np.dot(data[:, stat[n]['ipix']], stat[n]['lam'])
            #Fneu[n,inds] = np.mean(data[neuropil_masks[n,:], :], axis=0)
        Fneu[:,inds] = np.dot(neuropil_masks , data.T)
        ix += nimg
        k += 1
    print('Extracted fluorescence from %d ROIs in %d frames, %0.2f sec.'%(ncells, ops['nframes'], time.time()-t0))
    ops['meanImg'] /= k

    reg_file.close()
    return F, Fneu, ops

def compute_masks_and_extract_traces(ops, stat):
    ''' main extraction function
        inputs: ops and stat
        creates cell and neuropil masks and extracts traces
        returns: F (ROIs x time), Fneu (ROIs x time), F_chan2, Fneu_chan2, ops, stat
        F_chan2 and Fneu_chan2 will be empty if no second channel
    '''
    t0=time.time()
    stat,cell_pix,_ = masks.create_cell_masks(ops, stat)
    neuropil_masks = masks.create_neuropil_masks(ops,stat,cell_pix)
    Ly=ops['Ly']
    Lx=ops['Lx']
    neuropil_masks = np.reshape(neuropil_masks, (-1,Ly*Lx))

    stat0 = []
    for n in range(len(stat)):
        stat0.append({'ipix':stat[n]['ipix'],'lam':stat[n]['lam']/stat[n]['lam'].sum()})
    print('Masks made in %0.2f sec.'%(time.time()-t0))

    F,Fneu,ops = extract_traces(ops, stat0, neuropil_masks, ops['reg_file'])
    if 'reg_file_chan2' in ops:
        F_chan2, Fneu_chan2, ops2 = extract_traces(ops.copy(), stat0, neuropil_masks, ops['reg_file_chan2'])
        ops['meanImg_chan2'] = ops2['meanImg_chan2']
    else:
        F_chan2, Fneu_chan2 = [], []

    return F, Fneu, F_chan2, Fneu_chan2, ops, stat

def detect_and_extract(ops):
    t0=time.time()
    if ops['sparse_mode']:
        ops, stat = sparsedetect.sparsery(ops)
    else:
        ops, stat = sourcery.sourcery(ops)
    print('Found %d ROIs, %0.2f sec'%(len(stat), time.time()-t0))

    ### apply default classifier ###
    if len(stat) > 0:
        classfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "../classifiers/classifier_user.npy",
        )
        if not os.path.isfile(classfile):
            classorig = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                "../classifiers/classifier.npy"
            )
            shutil.copy(classorig, classfile)
        print('NOTE: applying classifier %s'%classfile)
        iscell = classifier.run(classfile, stat, keys=['npix_norm', 'compact', 'skew'])
        if 'preclassify' in ops and ops['preclassify'] > 0.0:
            ic = (iscell[:,0]>ops['preclassify']).flatten().astype(np.bool)
            stat = stat[ic]
            iscell = iscell[ic]
            print('After classification with threshold %0.2f, %d ROIs remain'%(ops['preclassify'], len(stat)))
    else:
        iscell = np.zeros((0,2))

    stat = sparsedetect.get_overlaps(stat,ops)
    stat, ix = sparsedetect.remove_overlaps(stat, ops, ops['Ly'], ops['Lx'])
    iscell = iscell[ix,:]
    print('After removing overlaps, %d ROIs remain'%(len(stat)))

    np.save(os.path.join(ops['save_path'],'iscell.npy'), iscell)

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

    # if second channel, detect bright cells in second channel
    fpath = ops['save_path']
    if 'meanImg_chan2' in ops:
        if 'chan2_thres' not in ops:
            ops['chan2_thres'] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        #redcell = np.zeros((len(stat),2))
        np.save(os.path.join(fpath, 'redcell.npy'), redcell)
        np.save(os.path.join(fpath, 'F_chan2.npy'), F_chan2)
        np.save(os.path.join(fpath, 'Fneu_chan2.npy'), Fneu_chan2)

    # add enhanced mean image
    ops = utils.enhanced_mean_image(ops)
    # save ops
    np.save(ops['ops_path'], ops)
    # save results
    np.save(os.path.join(fpath,'F.npy'), F)
    np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath,'stat.npy'), stat)
    return ops
