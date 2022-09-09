import os
import shutil
import time
from natsort import natsorted
from datetime import datetime
from getpass import getpass
import pathlib

import numpy as np
#from scipy.io import savemat

from . import extraction, io, registration, detection, classification, default_ops

try:
    from haussmeister import haussio
    HAS_HAUS = True
except ImportError:
    HAS_HAUS = False

from functools import partial
from pathlib import Path
print = partial(print,flush=True)

def pipeline(f_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None, 
             run_registration=True, ops=default_ops(), stat=None):
    """ run suite2p processing on array or BinaryRWFile 
    
    f_reg: required, registered or unregistered frames
        n_frames x Ly x Lx

    f_raw: optional, unregistered frames that will not be overwritten during registration
        n_frames x Ly x Lx

    f_reg_chan2: optional, non-functional frames
        n_frames x Ly x Lx

    f_raw_chan2: optional, non-functional frames that will not be overwritten during registration
        n_frames x Ly x Lx

    run_registration: optional, default True
        run registration

    ops: optional, `dict` of settings

    stat: optional, input predefined masks
    
    """
    
    plane_times = {}
    t1 = time.time()
    
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

    if run_registration:
        raw = f_raw is not None
        # if already shifted by bidiphase, do not shift again
        if not raw and ops['do_bidiphase'] and ops['bidiphase'] != 0:
            ops['bidi_corrected'] = True
        
        ######### REGISTRATION #########
        t11=time.time()
        print('----------- REGISTRATION')
        refImg = ops['refImg'] if 'refImg' in ops and ops.get('force_refImg', False) else None
        
        align_by_chan2 = ops['functional_chan'] != ops['align_by_chan']
        registration_outputs = registration.register.registration_wrapper(f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2, f_raw_chan2=f_raw_chan2, 
                                                                            refImg=refImg, align_by_chan2=align_by_chan2, ops=ops)

        ops = registration.register.save_registration_outputs_to_ops(registration_outputs, ops)
        # add enhanced mean image
        meanImgE = registration.register.compute_enhanced_mean_image(ops['meanImg'].astype(np.float32), ops)
        ops['meanImgE'] = meanImgE

        if ops.get('ops_path'):
            np.save(ops['ops_path'], ops)

        plane_times['registration'] = time.time()-t11
        print('----------- Total %0.2f sec' % plane_times['registration'])
        n_frames, Ly, Lx = f_reg.shape

        if ops['two_step_registration'] and ops['keep_movie_raw']:
            print('----------- REGISTRATION STEP 2')
            print('(making mean image (excluding bad frames)')
            nsamps = min(n_frames, 1000)
            inds = np.linspace(0, n_frames, 1+nsamps).astype(np.int64)[:-1]
            if align_by_chan2:
                refImg = f_reg_chan2[inds].astype(np.float32).mean(axis=0)
            else:
                refImg = f_reg[inds].astype(np.float32).mean(axis=0)
            registration_outputs = registration.register.registration_wrapper(f_reg, f_raw=None, f_reg_chan2=f_reg_chan2, f_raw_chan2=None, 
                                                                                refImg=refImg, align_by_chan2=align_by_chan2, ops=ops)
            if ops.get('ops_path'):
                np.save(ops['ops_path'], ops)
            plane_times['two_step_registration'] = time.time()-t11
            print('----------- Total %0.2f sec' % plane_times['two_step_registration'])

        # compute metrics for registration
        if ops.get('do_regmetrics', True) and n_frames>=1500:
            t0 = time.time()
            # n frames to pick from full movie
            nsamp = min(2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000, n_frames)
            inds = np.linspace(0, n_frames - 1, nsamp).astype('int')
            mov = f_reg[inds]
            mov = mov[:, ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
            ops = registration.get_pc_metrics(mov, ops)
            plane_times['registration_metrics'] = time.time()-t0
            print('Registration metrics, %0.2f sec.' % plane_times['registration_metrics'])
            if ops.get('ops_path'):
                np.save(ops['ops_path'], ops)
    
    if ops.get('roidetect', True):
        n_frames, Ly, Lx = f_reg.shape
        ######## CELL DETECTION ##############
        t11=time.time()
        print('----------- ROI DETECTION')
        if stat is None:
            ops, stat = detection.detection_wrapper(f_reg,
                                                    ops=ops, 
                                                    classfile=classfile)
        plane_times['detection'] = time.time()-t11
        print('----------- Total %0.2f sec.' % plane_times['detection'])

        
        if len(stat) > 0:
            ######## ROI EXTRACTION ##############
            t11=time.time()
            print('----------- EXTRACTION')
            stat, F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(stat, f_reg, f_reg_chan2=f_reg_chan2, ops=ops)
            # save results
            if ops.get('ops_path'):
                np.save(ops['ops_path'], ops)

            plane_times['extraction'] = time.time()-t11
            print('----------- Total %0.2f sec.' % plane_times['extraction'])

            ######## ROI CLASSIFICATION ##############
            t11=time.time()
            print('----------- CLASSIFICATION')
            if len(stat):
                iscell = classification.classify(stat=stat, classfile=classfile)
            else:
                iscell = np.zeros((0, 2))
            plane_times['classification'] = time.time()-t11
            
            ######### SPIKE DECONVOLUTION ###############
            fpath = ops['save_path']
            if ops.get('spikedetect', True):
                t11=time.time()
                print('----------- SPIKE DECONVOLUTION')
                dF = F.copy() - ops['neucoeff']*Fneu
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

            if ops.get('save_path'):
                fpath = ops['save_path']
                np.save(os.path.join(fpath, 'stat.npy'), stat)
                np.save(os.path.join(fpath,'F.npy'), F)
                np.save(os.path.join(fpath,'Fneu.npy'), Fneu)
                np.save(os.path.join(fpath, 'iscell.npy'), iscell)
                np.save(os.path.join(ops['save_path'], 'spks.npy'), spks)
                # if second channel, save F_chan2 and Fneu_chan2
                if 'meanImg_chan2' in ops:
                    np.save(os.path.join(fpath, 'F_chan2.npy'), F_chan2)
                    np.save(os.path.join(fpath, 'Fneu_chan2.npy'), Fneu_chan2)
            
            # save as matlab file
            if ops.get('save_mat'):
                stat = np.load(os.path.join(ops['save_path'], 'stat.npy'), allow_pickle=True)
                iscell = np.load(os.path.join(ops['save_path'], 'iscell.npy'))
                redcell = np.load(os.path.join(ops['save_path'], 'redcell.npy')) if ops['nchannels']==2 else []
                io.save_mat(ops, stat, F, Fneu, spks, iscell, redcell)
        else:
            print('no ROIs found, only ops.npy file saved')
    else:
        print("WARNING: skipping cell detection (ops['roidetect']=False)")
    ops['timing'] = plane_times.copy()
    plane_runtime = time.time()-t1
    ops['timing']['total_plane_runtime'] = plane_runtime
    if ops.get('ops_path'):
        np.save(ops['ops_path'], ops)

    return ops #, stat, F, Fneu, F_chan2, Fneu_chan2, spks, iscell, redcell
    

def run_plane(ops, ops_path=None, stat=None):
    """ run suite2p processing on a single binary file

    Parameters
    -----------
    ops : :obj:`dict` 
        specify 'reg_file', 'nchannels', 'tau', 'fs'

    ops_path: str
        absolute path to ops file (use if files were moved)

    stat: list of `dict`
        ROIs

    Returns
    --------
    ops : :obj:`dict` 
    """
    
    ops = {**default_ops(), **ops}
    ops['date_proc'] = datetime.now().astimezone()
    
    # for running on server or on moved files, specify ops_path
    if ops_path is not None:
        ops['save_path'] = os.path.split(ops_path)[0]
        ops['ops_path'] = ops_path 
        if len(ops['fast_disk'])==0 or ops['save_path']!=ops['fast_disk']:
            if os.path.exists(os.path.join(ops['save_path'], 'data.bin')):
                ops['reg_file'] = os.path.join(ops['save_path'], 'data.bin')
                if 'reg_file_chan2' in ops:
                    ops['reg_file_chan2'] = os.path.join(ops['save_path'], 'data_chan2.bin')    
                if 'raw_file' in ops:
                    ops['raw_file'] = os.path.join(ops['save_path'], 'data_raw.bin')
                if 'raw_file_chan2' in ops:
                    ops['raw_file_chan2'] = os.path.join(ops['save_path'], 'data_chan2_raw.bin')

    # check that there are sufficient numbers of frames
    if ops['nframes'] < 50:
        raise ValueError('the total number of frames should be at least 50.')
    if ops['nframes'] < 200:
        print('WARNING: number of frames is below 200, unpredictable behaviors may occur.')

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
            print('binary path: %s'%ops['reg_file'])
            run_registration = False
    else:
        print("NOTE: not running registration, ops['do_registration']=0")
        print('binary path: %s'%ops['reg_file'])
        run_registration = False

    Ly, Lx = ops['Ly'], ops['Lx']
    
    # get binary file paths
    raw = ops.get('keep_movie_raw') and 'raw_file' in ops and os.path.isfile(ops['raw_file'])
    reg_file = ops['reg_file']
    raw_file = ops.get('raw_file', 0) if raw else reg_file
    if ops['nchannels'] > 1:
        reg_file_chan2 = ops['reg_file_chan2']
        raw_file_chan2 = ops.get('raw_file_chan2', 0) if raw else reg_file_chan2
    else:
        reg_file_chan2 = reg_file 
        raw_file_chan2 = reg_file   
        

    with io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=raw_file if raw else reg_file) as f_raw, \
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file) as f_reg, \
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2 if raw else reg_file_chan2) as f_raw_chan2,\
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2) as f_reg_chan2:         
        if not raw:
            f_raw.close()
            f_raw = None
            f_raw_chan2.close()
            f_raw_chan2 = None
        if ops['nchannels'] == 1:
            f_reg_chan2.close() 
            f_reg_chan2 = None

        ops = pipeline(f_reg, f_raw, f_reg_chan2, f_raw_chan2, run_registration, ops, stat=stat)
        
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
    return ops


def run_s2p(ops={}, db={}, server={}):
    """ run suite2p pipeline

        need to provide a 'data_path' or 'h5py'+'h5py_key' in db or ops

        Parameters
        ----------
        ops : :obj:`dict`
            specify 'nplanes', 'nchannels', 'tau', 'fs'
        db : :obj:`dict`
            specify 'data_path' or 'h5py'+'h5py_key' here or in ops
        server : :obj:`dict`
            specify 'host', 'username', 'password', 'server_root', 'local_root', 'n_cores' ( for multiplane_parallel )


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
        if ops.get('h5py'):
            ops['save_path0'] = os.path.split(ops['h5py'][0])[0] # Use first element in h5py key to find save_path
        elif ops.get('nwb_file'):
            ops['save_path0'] = os.path.split(ops['nwb_file'])[0]
        else:
            ops['save_path0'] = ops['data_path'][0]
    
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
        print('removing previous detection and extraction files, if present')
        files_to_remove = ['stat.npy', 'F.npy', 'Fneu.npy', 'F_chan2.npy', 'Fneu_chan2.npy', 'iscell.npy', 'redcell.npy']
        for p in ops_paths:
            plane_folder = os.path.split(p)[0]
            for f in files_to_remove:
                if os.path.exists(os.path.join(plane_folder, f)):
                    os.remove(os.path.join(plane_folder, f))
    # if not set up files and copy tiffs/h5py to binary
    else:
        if len(ops['h5py']):
            ops['input_format'] = 'h5'
            # Overwrite data_path with path provided by h5py. 
            # Use the directory containing the first h5 file
            ops['data_path'] = [os.path.split(ops['h5py'][0])[0]]
        elif len(ops['nwb_file']):
            ops['input_format'] = 'nwb'
        elif ops.get('mesoscan'):
            ops['input_format'] = 'mesoscan'
        elif HAS_HAUS:
            ops['input_format'] = 'haus'
        elif not 'input_format' in ops:
            ops['input_format'] = 'tif'


        # copy file format to a binary file
        convert_funs = {
            'h5': io.h5py_to_binary,
            'nwb': io.nwb_to_binary,
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
        if server:
            if 'fnc' in server.keys():
                # Call custom function.
                server['fnc'](save_folder, server)
            else:
                # if user puts in server settings
                io.server.send_jobs(save_folder, host=server['host'], username=server['username'],
                                    password=server['password'], server_root=server['server_root'],
                                    local_root=server['local_root'], n_cores=server['n_cores'])
        else:
            # otherwise use settings modified in io/server.py
            io.server.send_jobs(save_folder)
        return None
    else:
        for ipl, ops_path in enumerate(ops_paths):
            if ipl in ops['ignore_flyback']:
                print('>>>> skipping flyback PLANE', ipl)
                continue
            op = np.load(ops_path, allow_pickle=True).item()
            
            # make sure yrange and xrange are not overwritten
            for key in default_ops().keys():
                if key not in ['data_path', 'save_path0', 'fast_disk', 'save_folder', 'subfolders']:
                    if key in ops:
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
