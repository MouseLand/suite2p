import datetime
import os
import shutil
import time
from natsort import natsorted
from itertools import chain

import numpy as np
from scipy.io import savemat

from . import extraction, io, registration, detection, classification
from . import version

try:
    from haussmeister import haussio
    HAS_HAUS = True
except ImportError:
    HAS_HAUS = False

from functools import partial
from pathlib import Path
print = partial(print,flush=True)


def default_ops():
    """ default options to run pipeline """
    return {
        # Suite2p version
        'suite2p_version': version,

        # file paths
        'look_one_level_down': False,  # whether to look in all subfolders when searching for tiffs
        'fast_disk': [],  # used to store temporary binary file, defaults to save_path0
        'delete_bin': False,  # whether to delete binary file after processing
        'mesoscan': False,  # for reading in scanimage mesoscope files
        'bruker': False,  # whether or not single page BRUKER tiffs!
        'h5py': [],  # take h5py as input (deactivates data_path)
        'h5py_key': 'data',  #key in h5py where data array is stored
        'save_path0': [],  # stores results, defaults to first item in data_path
        'save_folder': [],
        'subfolders': [],
        'move_bin': False,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk

        # main settings
        'nplanes' : 1,  # each tiff has these many planes in sequence
        'nchannels' : 1,  # each tiff has these many channels per plane
        'functional_chan' : 1,  # this channel is used to extract functional ROIs (1-based)
        'tau':  1.,  # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading
        'frames_include': -1,
        'multiplane_parallel': False, # whether or not to run on server

        # output settings
        'preclassify': 0.,  # apply classifier before signal extraction with probability 0.3
        'save_mat': False,  # whether to save output as matlab files
        'save_NWB': False,  # whether to save output as NWB file
        'combined': True,  # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)

        # bidirectional phase offset
        'do_bidiphase': False,
        'bidiphase': 0,
        'bidi_corrected': False,

        # registration settings
        'do_registration': 1,  # whether to register data (2 forces re-registration)
        'two_step_registration': False,
        'keep_movie_raw': False,
        'nimg_init': 300,  # subsampled frames for finding reference image
        'batch_size': 500,  # number of frames per batch
        'maxregshift': 0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1,  # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False,  # whether to save registered tiffs
        'reg_tif_chan2': False,  # whether to save channel 2 registered tiffs
        'subpixel' : 10,  # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma_time': 0,  # gaussian smoothing in time
        'smooth_sigma': 1.15,  # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
        'th_badframes': 1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,

        # non rigid registration settings
        'nonrigid': True,  # whether to use nonrigid registration
        'block_size': [128, 128],  # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5,  # maximum pixel shift allowed for nonrigid, relative to rigid

        # 1P settings
        '1Preg': False,  # whether to perform high-pass filtering and tapering
        'spatial_hp': 42,  # window for spatial high-pass filtering before registration
        'spatial_hp_reg': 42,  # window for spatial high-pass filtering before registration
        'spatial_hp_detect': 25,  # window for spatial high-pass filtering before registration
        'pre_smooth': 0,  # whether to smooth before high-pass filtering before registration
        'spatial_taper': 40,  # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])

        # cell detection settings
        'roidetect': True,  # whether or not to run ROI extraction
        'spikedetect': True,  # whether or not to run spike deconvolution
        'sparse_mode': True,  # whether or not to run sparse_mode
        'diameter': 12,  # if not sparse_mode, use diameter for filtering and extracting
        'spatial_scale': 0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 5000,  # max number of binned frames for cell detection
        'max_iterations': 20,  # maximum number of iterations to do cell detection
        'threshold_scaling': 1.0,  # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75,  # cells with more overlap than this get removed during triage, before refinement
        'high_pass': 100,  # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        'use_builtin_classifier': False,  # whether or not to use built-in classifier for cell detection (overrides
                                         # classifier specified in classifier_path if set to True)

        # ROI extraction parameters
        'inner_neuropil_radius': 2,  # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 350,  # minimum number of pixels in the neuropil
        'allow_overlap': False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)

        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65,  # minimum for detection of brightness on channel 2

        # deconvolution settings
        'baseline': 'maximin',  # baselining mode (can also choose 'prctile')
        'win_baseline': 60.,  # window for maximin
        'sig_baseline': 10.,  # smoothing constant for gaussian filter
        'prctile_baseline': 8.,  # optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
    }


def run_plane(ops, ops_path=None):
    """ run suite2p processing on a single binary file

    Parameters
    -----------
    ops : :obj:`dict` 
        specify 'reg_file', 'nchannels', 'tau', 'fs'

    Returns
    --------
    ops : :obj:`dict` 
    """
    t1 = time.time()
    
    ops = {**default_ops(), **ops}
    ops['date_proc'] = datetime.datetime.now()
    plane_times = {}
    if ops_path is not None:
        ops['save_path'] = os.path.split(ops_path)[0]
        ops['ops_path'] = ops_path 
        if len(ops['fast_disk'])==0:
            ops['reg_file'] = os.path.join(ops['save_path'], 'data.bin')
            if 'reg_file_chan2' in ops:
                ops['reg_file_chan2'] = os.path.join(ops['save_path'], 'data_chan2.bin')    
            if 'raw_file' in ops:
                ops['raw_file'] = os.path.join(ops['save_path'], 'data_raw.bin')
            if 'raw_file_chan2' in ops:
                ops['raw_file_chan2'] = os.path.join(ops['save_path'], 'data_chan2_raw.bin')

    # check if registration should be done
    if ops['do_registration']>0:
        if 'refImg' not in ops or 'yoff' not in ops or ops['do_registration'] > 1:
            print("NOTE: not registered / registration forced with ops['do_registration']>1")
            try:
                del ops['yoff'], ops['xoff'], ops['corrXY']  # delete previous offsets
            except KeyError:
                print('      (no previous offsets to delete)')
            run_registration = True
        else:
            print("NOTE: not running registration, plane already registered")
            run_registration = False
    else:
        print("NOTE: not running registration, ops['do_registration']=0")
        run_registration = False
    
    if run_registration:
        ######### REGISTRATION #########
        t11=time.time()
        print('----------- REGISTRATION')
        ops = registration.register_binary(ops) # register binary
        np.save(ops['ops_path'], ops)
        plane_times['registration'] = time.time()-t11
        print('----------- Total %0.2f sec' % plane_times['registration'])

        if ops['two_step_registration'] and ops['keep_movie_raw']:
            print('----------- REGISTRATION STEP 2')
            print('(making mean image (excluding bad frames)')
            refImg = registration.sampled_mean(ops)
            ops = registration.register_binary(ops, refImg, raw=False)
            np.save(ops['ops_path'], ops)
            plane_times['two_step_registration'] = time.time()-t11
            print('----------- Total %0.2f sec' % plane_times['two_step_registration'])

        # compute metrics for registration
        if ops.get('do_regmetrics', True) and ops['nframes']>=1500:
            t0 = time.time()
            ops = registration.get_pc_metrics(ops)
            plane_times['registration_metrics'] = time.time()-t0
            print('Registration metrics, %0.2f sec.' % plane_times['registration_metrics'])
            np.save(os.path.join(ops['save_path'], 'ops.npy'), ops)

    if ops.get('roidetect', True):

        # Select file for classification
        ops_classfile = ops.get('classifier_path')
        builtin_classfile = classification.builtin_classfile
        user_classfile = classification.user_classfile
        if ops_classfile:
            print(f'NOTE: applying classifier {str(ops_classfile)}')
            classfile = ops_classfile
        elif ops['use_builtin_classifier'] or not user_classfile.is_file():
            print(f'NOTE: Applying builtin classifier at {str(builtin_classfile)}')
            classfile = builtin_classfile
        else:
            print(f'NOTE: applying default {str(user_classfile)}')
            classfile = user_classfile

        ######## CELL DETECTION ##############
        t11=time.time()
        print('----------- ROI DETECTION')
        cell_pix, cell_masks, neuropil_masks, stat, ops = detection.detect(ops=ops, classfile=classfile)
        plane_times['detection'] = time.time()-t11
        print('----------- Total %0.2f sec.' % plane_times['detection'])

        ######## ROI EXTRACTION ##############
        t11=time.time()
        print('----------- EXTRACTION')
        ops, stat = extraction.extract(ops, cell_pix, cell_masks, neuropil_masks, stat)
        plane_times['extraction'] = time.time()-t11
        print('----------- Total %0.2f sec.' % plane_times['extraction'])

        ops['neuropil_masks'] = neuropil_masks.reshape(neuropil_masks.shape[0], ops['Ly'], ops['Lx'])

        ######## ROI CLASSIFICATION ##############
        t11=time.time()
        print('----------- CLASSIFICATION')
        if len(stat):
            iscell = classification.classify(stat=stat, classfile=classfile)
        else:
            iscell = np.zeros((0, 2))
        np.save(Path(ops['save_path']).joinpath('iscell.npy'), iscell)
        plane_times['classification'] = time.time()-t11
        print('----------- Total %0.2f sec.' % plane_times['classification'])

        ######### SPIKE DECONVOLUTION ###############
        fpath = ops['save_path']
        F = np.load(os.path.join(fpath,'F.npy'))
        Fneu = np.load(os.path.join(fpath,'Fneu.npy'))
        if ops.get('spikedetect', True):
            t11=time.time()
            print('----------- SPIKE DECONVOLUTION')
            dF = F - ops['neucoeff']*Fneu
            dF = extraction.preprocess(
                F=dF,
                baseline=ops['baseline'],
                win_baseline=ops['win_baseline'],
                sig_baseline=ops['sig_baseline'],
                fs=ops['fs'],
                prctile_baseline=ops['prctile_baseline']
            )
            spks = extraction.oasis(F=dF, batch_size=ops['batch_size'], tau=ops['tau'], fs=ops['fs'])
            plane_times['deconvolution'] = time.time()-t11
            print('----------- Total %0.2f sec.' % plane_times['deconvolution'])
        else:
            print("WARNING: skipping spike detection (ops['spikedetect']=False)")
            spks = np.zeros_like(F)
        np.save(os.path.join(ops['save_path'], 'spks.npy'), spks)

        # save as matlab file
        if ops.get('save_mat'):
            if 'date_proc' in ops:
                ops['date_proc'] = []
            savemat(
                file_name=os.path.join(ops['save_path'], 'Fall.mat'),
                mdict={
                    'stat': np.load(os.path.join(fpath, 'stat.npy'), allow_pickle=True),
                    'ops': ops,
                    'F': F,
                    'Fneu': Fneu,
                    'spks': spks,
                    'iscell': np.load(os.path.join(fpath, 'iscell.npy'))
                }
            )
    else:
        print("WARNING: skipping cell detection (ops['roidetect']=False)")

    if ops.get('move_bin') and ops['save_path'] != ops['fast_disk']:
        print('moving binary files to save_path')
        shutil.move(ops['reg_file'], os.path.join(ops['save_path'], 'data.bin'))
        if ops['nchannels']>1:
            shutil.move(ops['reg_file_chan2'], os.path.join(ops['save_path'], 'data_chan2.bin'))
        if 'raw_file' in ops:
            shutil.move(ops['raw_file'], os.path.join(ops['save_path'], 'data_raw.bin'))
            if ops['nchannels'] > 1:
                shutil.move(ops['raw_file_chan2'], os.path.join(ops['save_path'], 'data_chan2_raw.bin'))
    elif ops.get('delete_bin'):
        print('deleting binary files')
        os.remove(ops['reg_file'])
        if ops['nchannels'] > 1:
            os.remove(ops['reg_file_chan2'])
        if 'raw_file' in ops:
            os.remove(ops['raw_file'])
            if ops['nchannels'] > 1:
                os.remove(ops['raw_file_chan2'])
    ops['timing'] = plane_times.copy()
    plane_runtime = time.time()-t1
    ops['timing']['total_plane_runtime'] = plane_runtime
    np.save(ops['ops_path'], ops)
    return ops


def run_s2p(ops={}, db={}):
    """ run suite2p pipeline

        need to provide a 'data_path' or 'h5py'+'h5py_key' in db or ops

        Parameters
        ----------
        ops : :obj:`dict`
            specify 'nplanes', 'nchannels', 'tau', 'fs'
        db : :obj:`dict`
            specify 'data_path' or 'h5py'+'h5py_key' here or in ops

        Returns
        -------
            ops : :obj:`dict`
                ops settings used to run suite2p

    """
    t0 = time.time()
    ops = {**default_ops(), **ops, **db}
    if isinstance(ops['diameter'], list) and len(ops['diameter'])>1 and ops['aspect']==1.0:
        ops['aspect'] = ops['diameter'][0] / ops['diameter'][1]
    print(db)
    if 'save_path0' not in ops or len(ops['save_path0'])==0:
        ops['save_path0'] = os.path.split(ops['h5py'])[0] if ops.get('h5py') else ops['data_path'][0]
    
    # check if there are binaries already made
    if 'save_folder' not in ops or len(ops['save_folder'])==0:
        ops['save_folder'] = 'suite2p'
    save_folder = os.path.join(ops['save_path0'], ops['save_folder'])
    os.makedirs(save_folder, exist_ok=True)
    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    if len(plane_folders) > 0:
        ops_paths = [os.path.join(f, 'ops.npy') for f in plane_folders]
        ops_found_flag = all([os.path.isfile(ops_path) for ops_path in ops_paths])
        binaries_found_flag = all([os.path.isfile(os.path.join(f, 'data_raw.bin')) or os.path.isfile(os.path.join(f, 'data.bin')) 
                                    for f in plane_folders])
        files_found_flag = ops_found_flag and binaries_found_flag
    else:
        files_found_flag = False
    
    if files_found_flag:
        print(f'FOUND BINARIES AND OPS IN {ops_paths}')
    # if not set up files and copy tiffs/h5py to binary
    else:
        if len(ops['h5py']):
            ops['input_format'] = 'h5'
        elif ops.get('mesoscan'):
            ops['input_format'] = 'mesoscan'
        elif HAS_HAUS:
            ops['input_format'] = 'haus'
        elif not 'input_format' in ops:
            ops['input_format'] = 'tif'


        # copy file format to a binary file
        convert_funs = {
            'h5': io.h5py_to_binary,
            'sbx': io.sbx_to_binary,
            'mesoscan': io.mesoscan_to_binary,
            'haus': lambda ops: haussio.load_haussio(ops['data_path'][0]).tosuite2p(ops.copy()),
            'bruker': io.ome_to_binary,
        }
        if ops['input_format'] in convert_funs:
            ops0 = convert_funs[ops['input_format']](ops.copy())
            if isinstance(ops, list):
                ops0 = ops0[0]
        else:
            ops0 = io.tiff_to_binary(ops.copy())
        plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
        ops_paths = [os.path.join(f, 'ops.npy') for f in plane_folders]
        print('time {:0.2f} sec. Wrote {} frames per binary for {} planes'.format(
                  time.time() - t0, ops0['nframes'], len(plane_folders)
            ))

    if ops.get('multiplane_parallel'):
        io.server.send_jobs(save_folder)
        return None
    else:
        for ipl, ops_path in enumerate(ops_paths):
            op = np.load(ops_path, allow_pickle=True).item()
            
            # make sure yrange and xrange are not overwritten
            for key in default_ops().keys():
                if key not in ['data_path', 'save_path0', 'fast_disk', 'save_folder', 'subfolders']:
                    if key in op and key in ops:
                        op[key] = ops[key]
            
            print('>>>>>>>>>>>>>>>>>>>>> PLANE %d <<<<<<<<<<<<<<<<<<<<<<'%ipl)
            op = run_plane(op, ops_path=ops_path)
            print('Plane %d processed in %0.2f sec (can open in GUI).' % 
                    (ipl, op['timing']['total_plane_runtime']))  
        run_time = time.time()-t0
        print('total = %0.2f sec.' % run_time)

        #### COMBINE PLANES or FIELDS OF VIEW ####
        if len(ops_paths)>1 and ops['combined'] and ops.get('roidetect', True):
            print('Creating combined view')
            io.combined(save_folder, save=True)
        
        # save to NWB
        if ops.get('save_NWB'):
            print('Saving in nwb format')
            io.save_nwb(save_folder)

        print('TOTAL RUNTIME %0.2f sec' % (time.time()-t0))
        return op
