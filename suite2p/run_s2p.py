import os
import numpy as np
import time, os, shutil
from suite2p import register, dcnv, classifier, utils, roiextract, regmetrics
from scipy import stats, io, signal
try:
    from haussmeister import haussio
    HAS_HAUS = True
except ImportError:
    HAS_HAUS = False

def tic():
    return time.time()
def toc(t0):
    return time.time() - t0

def default_ops():
    ops = {
        # file paths
        'look_one_level_down': False, # whether to look in all subfolders when searching for tiffs
        'fast_disk': [], # used to store temporary binary file, defaults to save_path0
        'delete_bin': False, # whether to delete binary file after processing
        'mesoscan': False, # for reading in scanimage mesoscope files
        'h5py': [], # take h5py as input (deactivates data_path)
        'h5py_key': 'data', #key in h5py where data array is stored
        'save_path0': [], # stores results, defaults to first item in data_path
        'subfolders': [],
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading
        # output settings
        'preclassify': 0., # apply classifier before signal extraction with probability 0.3
        'save_mat': False, # whether to save output as matlab files
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0, # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # bidirectional phase offset
        'do_bidiphase': False,
        'bidiphase': 0,
        # registration settings
        'do_registration': 1, # whether to register data (2 forces re-registration)
        'keep_movie_raw': False,
        'nimg_init': 300, # subsampled frames for finding reference image
        'batch_size': 500, # number of frames per batch
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False, # whether to save registered tiffs
        'reg_tif_chan2': False, # whether to save channel 2 registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma': 1.15, # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'th_badframes': 1.0, # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        # non rigid registration settings
        'nonrigid': True, # whether to use nonrigid registration
        'block_size': [128, 128], # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2, # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5, # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        '1Preg': False, # whether to perform high-pass filtering and tapering
        'spatial_hp': 50, # window for spatial high-pass filtering before registration
        'pre_smooth': 2, # whether to smooth before high-pass filtering before registration
        'spatial_taper': 50, # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])
        # cell detection settings
        'roidetect': True, # whether or not to run ROI extraction
        'sparse_mode': False, # whether or not to run sparse_mode
        'diameter': 12, # if not sparse_mode, use diameter for filtering and extracting
        'spatial_scale': 0, # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 5000, # max number of binned frames for cell detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'threshold_scaling': 1.0, # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'high_pass': 100, # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        # ROI extraction parameters
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        'allow_overlap': False, # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65, # minimum for detection of brightness on channel 2
        # deconvolution settings
        'baseline': 'maximin', # baselining mode (can also choose 'prctile')
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0]),
      }
    return ops

def run_s2p(ops={},db={}):
    t0 = tic()
    ops0 = default_ops()
    ops = {**ops0, **ops}
    ops = {**ops, **db}
    print(db)
    if 'save_path0' not in ops or len(ops['save_path0'])==0:
        if ('h5py' in ops) and len(ops['h5py'])>0:
            ops['save_path0'], tail = os.path.split(ops['h5py'])
        else:
            ops['save_path0'] = ops['data_path'][0]

    # check if there are files already registered!
    fpathops1 = os.path.join(ops['save_path0'], 'suite2p', 'ops1.npy')
    if os.path.isfile(fpathops1):
        files_found_flag = True
        flag_binreg = True
        ops1 = np.load(fpathops1, allow_pickle=True)
        print('FOUND OPS IN %s'%ops1[0]['save_path'])
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
            if 'refImg' not in op or op['do_registration']>1:
                flag_binreg = False
                if i==len(ops1)-1:
                    print("NOTE: not registered / registration forced with ops['do_registration']>1")
            # use the new False
            ops1[i] = {**op, **ops}.copy()
            # for mesoscope tiffs, preserve original lines, etc
            if 'lines' in op:
                ops1[i]['nrois'] = op['nrois']
                ops1[i]['nplanes'] = op['nplanes']
                ops1[i]['lines'] = op['lines']
                ops1[i]['dy'] = op['dy']
                ops1[i]['dx'] = op['dx']
                ops1[i]['iplane'] = op['iplane']

            #ops1[i] = ops1[i].copy()
            # except for registration results
            ops1[i]['xrange'] = op['xrange']
            ops1[i]['yrange'] = op['yrange']
    else:
        files_found_flag = False
        flag_binreg = False

    # if not set up files and copy tiffs/h5py to binary
    if not files_found_flag:
        # get default options
        ops0 = default_ops()
        # combine with user options
        ops = {**ops0, **ops}
        # copy tiff to a binary
        if len(ops['h5py']):
            ops1 = utils.h5py_to_binary(ops)
            print('time %4.2f sec. Wrote h5py to binaries for %d planes'%(toc(t0), len(ops1)))
        else:
            if 'mesoscan' in ops and ops['mesoscan']:
                ops1 = utils.mesoscan_to_binary(ops)
                print('time %4.2f sec. Wrote tifs to binaries for %d planes'%(toc(t0), len(ops1)))
            elif HAS_HAUS:
                print('time %4.2f sec. Using HAUSIO')
                dataset = haussio.load_haussio(ops['data_path'][0])
                ops1 = dataset.tosuite2p(ops)
                print('time %4.2f sec. Wrote data to binaries for %d planes'%(toc(t0), len(ops1)))
            else:
                ops1 = utils.tiff_to_binary(ops)
                print('time %4.2f sec. Wrote tifs to binaries for %d planes'%(toc(t0), len(ops1)))

        np.save(fpathops1, ops1) # save ops1
    else:
        print('FOUND BINARIES: %s'%ops1[0]['reg_file'])

    ops1 = np.array(ops1)
    #ops1 = utils.split_multiops(ops1)
    if not ops['do_registration']:
        flag_binreg = True
    if flag_binreg:
        print('SKIPPING REGISTRATION FOR ALL PLANES...')
    if flag_binreg and not files_found_flag:
        print('NOTE: binary file created, but registration not performed')

    # set up number of CPU workers for registration and cell detection
    ipl = 0

    while ipl<len(ops1):
        print('>>>>>>>>>>>>>>>>>>>>> PLANE %d <<<<<<<<<<<<<<<<<<<<<<'%ipl)
        t1 = tic()
        if not flag_binreg:
            ######### REGISTRATION #########
            t11=tic()
            print('----------- REGISTRATION')
            ops1[ipl] = register.register_binary(ops1[ipl]) # register binary
            np.save(fpathops1, ops1) # save ops1
            print('----------- Total %0.2f sec'%(toc(t11)))
        if not files_found_flag or not flag_binreg:
            # compute metrics for registration
            if 'do_regmetrics' in ops:
                do_regmetrics = ops['do_regmetrics']
            else:
                do_regmetrics = True
            if do_regmetrics and ops1[ipl]['nframes']>=1500:
                t0=tic()
                ops1[ipl] = regmetrics.get_pc_metrics(ops1[ipl])
                print('Registration metrics, %0.2f sec.'%(toc(t0)))
                np.save(os.path.join(ops1[ipl]['save_path'],'ops.npy'), ops1[ipl])
        if 'roidetect' in ops1[ipl]:
            roidetect = ops['roidetect']
        else:
            roidetect = True
        if roidetect:
            ######## CELL DETECTION AND ROI EXTRACTION ##############
            t11=tic()
            print('----------- ROI DETECTION AND EXTRACTION')
            ops1[ipl] = roiextract.roi_detect_and_extract(ops1[ipl])
            ops = ops1[ipl]
            fpath = ops['save_path']
            print('----------- Total %0.2f sec.'%(toc(t11)))

            ######### SPIKE DECONVOLUTION ###############
            t11=tic()
            print('----------- SPIKE DECONVOLUTION')
            F = np.load(os.path.join(fpath,'F.npy'))
            Fneu = np.load(os.path.join(fpath,'Fneu.npy'))
            dF = F - ops['neucoeff']*Fneu
            spks = dcnv.oasis(dF, ops)
            np.save(os.path.join(ops['save_path'],'spks.npy'), spks)
            print('----------- Total %0.2f sec.'%(toc(t11)))

            # save as matlab file
            if ('save_mat' in ops) and ops['save_mat']:
                stat = np.load(os.path.join(fpath,'stat.npy'), allow_pickle=True)
                iscell = np.load(os.path.join(fpath,'iscell.npy'))
                matpath = os.path.join(ops['save_path'],'Fall.mat')
                io.savemat(matpath, {'stat': stat,
                                     'ops': ops,
                                     'F': F,
                                     'Fneu': Fneu,
                                     'spks': spks,
                                     'iscell': iscell})
        else:
            print("WARNING: skipping cell detection (ops['roidetect']=False)")
        print('Plane %d processed in %0.2f sec (can open in GUI).'%(ipl,toc(t1)))
        print('total = %0.2f sec.'%(toc(t0)))
        ipl += 1 #len(ipl)

    # save final ops1 with all planes
    np.save(fpathops1, ops1)

    #### COMBINE PLANES or FIELDS OF VIEW ####
    if len(ops1)>1 and ops1[0]['combined'] and roidetect:
        utils.combined(ops1)

    # running a clean up script
    if 'clean_script' in ops1[0]:
        print('running clean-up script')
        os.system('python '+ ops['clean_script'] + ' ' + fpathops1)

    for ops in ops1:
        if ('delete_bin' in ops) and ops['delete_bin']:
            os.remove(ops['reg_file'])
            if ops['nchannels']>1:
                os.remove(ops['reg_file_chan2'])

    print('TOTAL RUNTIME %0.2f sec'%toc(t0))
    return ops1
