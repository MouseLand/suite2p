import numpy as np
import time, os
from suite2p import register, dcnv, classifier, utils
from suite2p import celldetect2 as celldetect2
from scipy import stats, io, signal
from multiprocessing import Pool
try:
    from haussmeister import haussio
    HAS_HAUS = True
except ImportError:
    HAS_HAUS = False

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def default_ops():
    ops = {
        # file paths
        'look_one_level_down': False, # whether to look in all subfolders when searching for tiffs
        'fast_disk': [], # used to store temporary binary file, defaults to save_path0
        'delete_bin': False, # whether to delete binary file after processing
        'h5py': [], # take h5py as input (deactivates data_path)
        'h5py_key': 'data', #key in h5py where data array is stored
        'save_path0': [], # stores results, defaults to first item in data_path
        'subfolders': [],
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'diameter':12, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)
        # output settings
        'save_mat': False, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        # parallel settings
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
        # registration settings
        'do_registration': True, # whether to register data
        'nimg_init': 200, # subsampled frames for finding reference image
        'batch_size': 200, # number of frames per batch
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False, # whether to save registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        # cell detection settings
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'navg_frames_svd': 5000, # max number of binned frames for the SVD
        'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
        'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
        'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
        'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'outer_neuropil_radius': np.inf, # maximum neuropil radius
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        # deconvolution settings
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'allow_overlap': False,
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0]),
      }
    return ops

def run_s2p(ops={},db={}):
    i0 = tic()
    ops = {**ops, **db}
    if 'save_path0' not in ops or len(ops['save_path0'])==0:
        if ('h5py' in ops) and len(ops['h5py'])>0:
            ops['save_path0'], tail = os.path.split(ops['h5py'])
        else:
            ops['save_path0'] = ops['data_path'][0]
    # check if there are files already registered
    fpathops1 = os.path.join(ops['save_path0'], 'suite2p', 'ops1.npy')
    if os.path.isfile(fpathops1):
        files_found_flag = True
        flag_binreg = True
        ops1 = np.load(fpathops1)
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
            if 'refImg' not in op:
                flag_binreg = False
            # use the new options
            ops1[i] = {**op, **ops}
            ops1[i] = ops1[i].copy()
            print(ops1[i]['save_path'])
            # except for registration results
            ops1[i]['xrange'] = op['xrange']
            ops1[i]['yrange'] = op['yrange']
    else:
        files_found_flag = False
        flag_binreg = False
    ######### REGISTRATION #########
    if not files_found_flag:
        # get default options
        ops0 = default_ops()
        # combine with user options
        ops = {**ops0, **ops}
        # copy tiff to a binary
        if len(ops['h5py']):
            ops1 = utils.h5py_to_binary(ops)
            print('time %4.4f. Wrote h5py to binaries for %d planes'%(toc(i0), len(ops1)))
        else:
            try:
                ops1 = utils.tiff_to_binary(ops)
                print('time %4.4f. Wrote tifs to binaries for %d planes'%(toc(i0), len(ops1)))
            except Exception as e:
                if HAS_HAUS:
                    dataset = haussio.load_haussio(ops['data_path'][0])
                    ops1 = dataset.tosuite2p(ops)
                    print('time %4.4f. Wrote data to binaries for %d planes'%(toc(i0), len(ops1)))
                else:
                    print('Unsupported file format: ' + str(e))
                    return
        # save ops1
        np.save(fpathops1, ops1)
    if not ops['do_registration']:
        flag_binreg = True
    if not flag_binreg:
        ops1 = register.register_binary(ops1) # register tiff
        np.save(fpathops1, ops1) # save ops1
        print('time %4.4f. Registration complete'%toc(i0))
    elif files_found_flag:
        print('found ops1 and pre-registered binaries')
        print(ops1[0]['reg_file'])
        print('overwriting ops1 with new ops')
        print('skipping registration...')
    # registration is off, but previous binary was not found
    # need to initialize ops
    else:
        print('binary file created, but registration not performed')
    ######### CELL DETECTION #########
    if len(ops1)>1 and ops['num_workers_roi']>=0:
        if ops['num_workers_roi']==0:
            ops['num_workers_roi'] = len(ops1)
        with Pool(ops['num_workers_roi']) as p:
            ops1 = p.map(utils.get_cells, ops1)
    else:
        for k in range(len(ops1)):
            ops1[k] = utils.get_cells(ops1[k])
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
            io.savemat(matpath, {'stat': stat,
                                 'ops': ops,
                                 'F': F,
                                 'Fneu': Fneu,
                                 'spks': spks,
                                 'iscell': iscell})

    # save final ops1 with all planes
    np.save(fpathops1, ops1)

    #### COMBINE PLANES or FIELDS OF VIEW ####
    if len(ops1)>1 and ops1[0]['combined']:
        utils.combined(ops1)

    for ops in ops1:
        if ('delete_bin' in ops) and ops['delete_bin']:
            os.remove(ops['reg_file'])
            if ops['nchannels']>1:
                os.remove(ops['reg_file_chan2'])

    print('finished all tasks in total time %4.4f sec'%toc(i0))
    return ops1
