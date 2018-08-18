import numpy as np
import time, os
from suite2p import register, dcnv, classifier, utils
from suite2p import celldetect2 as celldetect2
from scipy import stats, io, signal
from multiprocessing import Pool

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def default_ops():
    ops = {
        'reg_tif': False, # whether to save registered tiffs
        'save_mat': False, # whether to save Matlab results
        'fast_disk': [], # used to store temporary binary file, defaults to save_path0
        'delete_bin': False, # whether to delete binary file after processing
        'h5py': [], # take h5py as input (deactivates data_path)
        'h5py_key': 'data', #key in h5py where data array is stored
        'save_path0': [], # stores results, defaults to first item in data_path
        'diameter':12, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'look_one_level_down': False, # whether to look in all subfolders when searching for tiffs
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'neumax': 1.,  # maximum neuropil coefficient (not implemented)
        'niterneu': 5, # number of iterations when the neuropil coefficient is estimated (not implemented)
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        'batch_size': 200, # number of frames per batch
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        'nimg_init': 200, # subsampled frames for finding reference image
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'allow_overlap': False,
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0]),
        'connected': True,
      }
    return ops

def get_cells(ops):
    i0 = tic()
    if (type(ops['diameter']) is int) or len(ops['diameter'])<2:
        ops['diameter'] = int(np.array(ops['diameter']))
        ops['diameter'] = np.array((ops['diameter'], ops['diameter']))
    ops['diameter'] = np.array(ops['diameter']).astype('int32')
    print(ops['diameter'])
    ops, stat = celldetect2.sourcery(ops)
    print('time %4.4f. Found %d ROIs'%(toc(i0), len(stat)))
    # extract fluorescence and neuropil
    F, Fneu, ops = celldetect2.extractF(ops, stat)
    print('time %4.4f. Extracted fluorescence from %d ROIs'%(toc(i0), len(stat)))
    # subtract neuropil
    dF = F - ops['neucoeff'] * Fneu
    # compute activity statistics for classifier
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)
    for k in range(F.shape[0]):
        stat[k]['skew'] = sk[k]
        stat[k]['std']  = sd[k]
    # add enhanced mean image
    ops = utils.enhanced_mean_image(ops)
    # save ops
    np.save(ops['ops_path'], ops)
    # save results
    fpath = ops['save_path']
    np.save(os.path.join(fpath,'F.npy'), F)
    np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath,'stat.npy'), stat)
    iscell = np.ones((len(stat),2))
    np.save(os.path.join(fpath, 'iscell.npy'), iscell)
    print('results saved to %s'%ops['save_path'])
    return ops

def combined(ops1):
    '''
    Combines all the entries in ops1 into a single result file.
    Multi-plane recordings are arranged to best tile a square.
    Multi-roi recordings are arranged by their dx,dy physical localization.
    '''
    ops = ops1[0]
    if ('dx' not in ops) or ('dy' not in ops):
        Lx = ops['Lx']
        Ly = ops['Ly']
        nX = np.ceil(np.sqrt(ops['Ly'] * ops['Lx'] * len(ops1))/ops['Lx'])
        nX = int(nX)
        nY = int(np.ceil(len(ops1)/nX))
        for j in range(len(ops1)):
            ops1[j]['dx'] = (j%nX) * Lx
            ops1[j]['dy'] = int(j/nX) * Ly
    LY = int(np.amax(np.array([ops['Ly']+ops['dy'] for ops in ops1])))
    LX = int(np.amax(np.array([ops['Lx']+ops['dx'] for ops in ops1])))
    meanImg = np.zeros((LY, LX))
    meanImgE = np.zeros((LY, LX))
    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([ops['nframes'] for ops in ops1]))
    for k,ops in enumerate(ops1):
        fpath = ops['save_path']
        stat0 = np.load(os.path.join(fpath,'stat.npy'))
        xrange = np.arange(ops['dx'],ops['dx']+ops['Lx'])
        yrange = np.arange(ops['dy'],ops['dy']+ops['Ly'])
        meanImg[np.ix_(yrange, xrange)] = ops['meanImg']
        meanImgE[np.ix_(yrange, xrange)] = ops['meanImgE']
        xrange = np.arange(ops['dx']+ops['xrange'][0],ops['dx']+ops['xrange'][-1])
        yrange = np.arange(ops['dy']+ops['yrange'][0],ops['dy']+ops['yrange'][-1])
        Vcorr[np.ix_(yrange, xrange)] = ops['Vcorr']
        for j in range(len(stat0)):
            stat0[j]['xpix'] += ops['dx']
            stat0[j]['ypix'] += ops['dy']
            stat0[j]['med'][0] += ops['dy']
            stat0[j]['med'][1] += ops['dx']
        F0    = np.load(os.path.join(fpath,'F.npy'))
        Fneu0 = np.load(os.path.join(fpath,'Fneu.npy'))
        spks0 = np.load(os.path.join(fpath,'spks.npy'))
        iscell0 = np.load(os.path.join(fpath,'iscell.npy'))
        nn,nt = F0.shape
        if nt<Nfr:
            fcat    = np.zeros((nn,Nfr-nt), 'float32')
            print(F0.shape)
            print(fcat.shape)
            F0      = np.concatenate((F0, fcat), axis=1)
            spks0   = np.concatenate((spks0, fcat), axis=1)
            Fneu0   = np.concatenate((Fneu0, fcat), axis=1)
        if k==0:
            F, Fneu, spks,stat,iscell = F0, Fneu0, spks0,stat0, iscell0
        else:
            F    = np.concatenate((F, F0))
            Fneu = np.concatenate((Fneu, Fneu0))
            spks = np.concatenate((spks, spks0))
            stat = np.concatenate((stat,stat0))
            iscell = np.concatenate((iscell,iscell0))
    ops['meanImg']  = meanImg
    ops['meanImgE'] = meanImgE
    ops['Vcorr'] = Vcorr
    ops['Ly'] = LY
    ops['Lx'] = LX
    ops['xrange'] = [0, ops['Lx']]
    ops['yrange'] = [0, ops['Ly']]
    fpath = os.path.join(ops['save_path0'], 'suite2p', 'combined')
    if not os.path.isdir(fpath):
        os.makedirs(fpath)
    ops['save_path'] = fpath
    np.save(os.path.join(fpath, 'F.npy'), F)
    np.save(os.path.join(fpath, 'Fneu.npy'), Fneu)
    np.save(os.path.join(fpath, 'spks.npy'), spks)
    np.save(os.path.join(fpath, 'ops.npy'), ops)
    np.save(os.path.join(fpath, 'stat.npy'), stat)
    np.save(os.path.join(fpath, 'iscell.npy'), iscell)

    # save as matlab file
    if ('save_mat' in ops) and ops['save_mat']:
        matpath = os.path.join(ops['save_path'],'Fall.mat')
        scipy.io.savemat(matpath, {'stat': stat,
                                   'ops': ops,
                                   'F': F,
                                   'Fneu': Fneu,
                                   'spks': spks,
                                   'iscell': iscell})
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
            # default behavior is to look in the ops
            flag_reg = os.path.isfile(op['reg_file'])
            if not flag_reg:
                # otherwise look in the user defined save_path0
                op['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%i)
                op['ops_path'] = os.path.join(op['save_path'],'ops.npy')
                op['reg_file'] = os.path.join(op['save_path'], 'data.bin')
                flag_reg = os.path.isfile(op['reg_file'])
            files_found_flag &= flag_reg
            # use the new options
            ops1[i] = {**op, **ops}
            ops1[i] = ops1[i].copy()
            print(ops1[i]['save_path'])
            # except for registration results
            ops1[i]['xrange'] = op['xrange']
            ops1[i]['yrange'] = op['yrange']
    else:
        files_found_flag = False
    ######### REGISTRATION #########
    if not files_found_flag:
        # get default options
        ops0 = default_ops()
        # combine with user options
        ops = {**ops0, **ops}
        # copy tiff to a binary
        if len(ops['h5py']):
            ops1 = register.h5py_to_binary(ops)
            print('time %4.4f. Wrote h5py to binaries for %d planes'%(toc(i0), len(ops1)))
        else:
            ops1 = register.tiff_to_binary(ops)
            print('time %4.4f. Wrote tifs to binaries for %d planes'%(toc(i0), len(ops1)))
        # register tiff
        ops1 = register.register_binary(ops1)
        # save ops1
        np.save(fpathops1, ops1)
        print('time %4.4f. Registration complete'%toc(i0))
    else:
        print('found ops1 and pre-registered binaries')
        print(ops1[0]['reg_file'])
        print('overwriting ops1 with new ops')
        print('skipping registration...')
    ######### CELL DETECTION #########
    if len(ops1)>1 and ops['num_workers_roi']>=0:
        if ops['num_workers_roi']==0:
            ops['num_workers_roi'] = len(ops1)
        with Pool(ops['num_workers_roi']) as p:
            ops1 = p.map(get_cells, ops1)
    else:
        for k in range(len(ops1)):
            ops1[k] = get_cells(ops1[k])
    ######### SPIKE DECONVOLUTION AND CLASSIFIER #########
    for ops in ops1:
        fpath = ops['save_path']
        F = np.load(os.path.join(fpath,'F.npy'))
        Fneu = np.load(os.path.join(fpath,'Fneu.npy'))
        dF = F - ops['neucoeff']*Fneu
        spks = dcnv.oasis(dF, ops)
        np.save(os.path.join(ops['save_path'],'spks.npy'), spks)
        print('time %4.4f. Detected spikes in %d ROIs'%(toc(i0), F.shape[0]))
        stat = np.load(os.path.join(fpath,'stat.npy'))
        # apply default classifier
        classfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'classifiers/classifier_user.npy')
        print(classfile)
        iscell = classifier.run(classfile, stat)
        np.save(os.path.join(ops['save_path'],'iscell.npy'), iscell)
        # save as matlab file
        if ('save_mat' in ops) and ops['save_mat']:
            matpath = os.path.join(ops['save_path'],'Fall.mat')
            scipy.io.savemat(matpath, {'stat': stat,
                                       'ops': ops,
                                       'F': F,
                                       'Fneu': Fneu,
                                       'spks': spks,
                                       'iscell': iscell})

    # save final ops1 with all planes
    np.save(fpathops1, ops1)

    #### COMBINE PLANES or FIELDS OF VIEW ####
    if len(ops1)>1 and ops1[0]['combined']:
        combined(ops1)

    for ops in ops1:
        if ('delete_bin' in ops) and ops['delete_bin']:
            os.remove(ops['reg_file'])
            if ops['nchannels']>1:
                os.remove(ops['reg_file_chan2'])

    print('finished all tasks in total time %4.4f sec'%toc(i0))
    return ops1
