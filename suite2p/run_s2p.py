import numpy as np
import time, os, shutil, datetime
from scipy.io import savemat
from .io import tiff, h5, save, nwb
from .registration import register, metrics, reference
from .extraction import extract, dcnv
from . import utils
try:
    from haussmeister import haussio
    HAS_HAUS = True
except ImportError:
    HAS_HAUS = False

from functools import partial
print = partial(print,flush=True)


def default_ops():
    """ default options to run pipeline (MOVED TO UTILS)"""
    return utils.default_ops()

def run_s2p(ops={},db={}):
    """ run suite2p pipeline

        need to provide a 'data_path' or 'h5py'+'h5py_key' in db or ops

        Parameters
        ----------
        ops : :obj:`dict`, optional
            specify 'nplanes', 'nchannels', 'tau', 'fs'
        db : :obj:`dict`, optional
            specify 'data_path' or 'h5py'+'h5py_key' here or in ops

        Returns
        -------
            ops1 : list
                list of ops for each plane

    """

    t0 = time.time()
    ops0 = default_ops()
    ops = {**ops0, **ops}
    ops = {**ops, **db}
    if isinstance(ops['diameter'], list) and len(ops['diameter'])>1 and ops['aspect']==1.0:
        ops['aspect'] = ops['diameter'][0] / ops['diameter'][1]
    print(db)
    if 'save_path0' not in ops or len(ops['save_path0'])==0:
        if ('h5py' in ops) and len(ops['h5py'])>0:
            ops['save_path0'], tail = os.path.split(ops['h5py'])
        else:
            ops['save_path0'] = ops['data_path'][0]

    # check if there are files already registered!
    if len(ops['save_folder']) > 0:
        fpathops1 = os.path.join(ops['save_path0'], ops['save_folder'], 'ops1.npy')
    else:
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
                    try:
                        # delete previous offsets
                        del op['yoff'], op['xoff'], op['corrXY']
                    except:
                        print('no offsets to delete')
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
        ops['date_proc'] = datetime.datetime.now()
    if not 'input_format' in ops.keys():
        ops['input_format'] = 'tif'
    if len(ops['h5py']):
        ops['input_format'] = 'h5'
    elif 'mesoscan' in ops and ops['mesoscan']:
        ops['input_format'] = 'mesoscan'
    elif HAS_HAUS:
        ops['input_format'] = 'haus'
    # if not set up files and copy tiffs/h5py to binary
    if not files_found_flag:
        # get default options
        ops0 = default_ops()
        # combine with user options
        ops = {**ops0, **ops}
        # copy tiff to a binary
        if ops['input_format'] == 'h5':
            from .io import h5
            ops1 = h5.h5py_to_binary(ops)
            print('time %4.2f sec. Wrote h5py to binaries for %d planes'%(time.time()-(t0), len(ops1)))
        elif ops['input_format'] == 'sbx':
            from .io import sbx
            ops1 = sbx.sbx_to_binary(ops)
            print('time %4.2f sec. Wrote sbx to binaries for %d planes'%(time.time()-(t0), len(ops1)))
        else:
            from .io import tiff
            if ops['input_format'] == 'mesoscan':
                ops1 = tiff.mesoscan_to_binary(ops)
                print('time %4.2f sec. Wrote mesoscope tifs to binaries for %d planes'%(time.time()-(t0), len(ops1)))
            elif ops['input_format'] == 'haus':
                print('time %4.2f sec. Using HAUSIO')
                dataset = haussio.load_haussio(ops['data_path'][0])
                ops1 = dataset.tosuite2p(ops)
                print('time %4.2f sec. Wrote data to binaries for %d planes'%(time.time()-(t0), len(ops1)))
            elif ops['input_format'] == 'bruker':
                ops['bruker'] = True
                ops1 = tiff.ome_to_binary(ops)
                print('time %4.2f sec. Wrote bruker tifs to binaries for %d planes'%(time.time()-(t0), len(ops1)))
            else:
                ops1 = tiff.tiff_to_binary(ops)
                print('time %4.2f sec. Wrote tifs to binaries for %d planes'%(time.time()-(t0), len(ops1)))
        np.save(fpathops1, ops1) # save ops1
    else:
        print('FOUND BINARIES: %s'%ops1[0]['reg_file'])

    ops1 = np.array(ops1)
    #ops1 = utils.split_multiops(ops1)
    if not ops['do_registration']:
        flag_binreg = True

    if ops['do_registration']>1:
        flag_binreg = False
        print('do_registration>1 => forcing registration')

    if flag_binreg:
        print('SKIPPING REGISTRATION FOR ALL PLANES...')
    if flag_binreg and not files_found_flag:
        print('NOTE: binary file created, but registration not performed')

    # set up number of CPU workers for registration and cell detection
    ipl = 0

    while ipl<len(ops1):
        print('>>>>>>>>>>>>>>>>>>>>> PLANE %d <<<<<<<<<<<<<<<<<<<<<<'%ipl)
        t1 = time.time()
        if not flag_binreg:
            ######### REGISTRATION #########
            t11=time.time()
            print('----------- REGISTRATION')
            ops1[ipl] = register.register_binary(ops1[ipl]) # register binary
            np.save(fpathops1, ops1) # save ops1
            print('----------- Total %0.2f sec'%(time.time()-t11))

            if ops['two_step_registration'] and ops['keep_movie_raw']:
                print('----------- REGISTRATION STEP 2')
                print('(making mean image (excluding bad frames)')
                refImg = reference.sampled_mean(ops1[ipl])
                ops1[ipl] = register.register_binary(ops1[ipl], refImg, raw=False)
                np.save(fpathops1, ops1) # save ops1
                print('----------- Total %0.2f sec'%(time.time()-t11))

        if not files_found_flag or not flag_binreg:
            # compute metrics for registration
            if 'do_regmetrics' in ops:
                do_regmetrics = ops['do_regmetrics']
            else:
                do_regmetrics = True
            if do_regmetrics and ops1[ipl]['nframes']>=1500:
                t0=time.time()
                ops1[ipl] = metrics.get_pc_metrics(ops1[ipl])
                print('Registration metrics, %0.2f sec.'%(time.time()-t0))
                np.save(os.path.join(ops1[ipl]['save_path'],'ops.npy'), ops1[ipl])
        roidetect = True
        spikedetect = True
        if 'roidetect' in ops1[ipl]:
            roidetect = ops['roidetect']
        if 'spikedetect' in ops1[ipl]:
            spikedetect = ops['spikedetect']

        if roidetect:
            ######## CELL DETECTION AND ROI EXTRACTION ##############
            t11=time.time()
            print('----------- ROI DETECTION AND EXTRACTION')
            ops1[ipl] = extract.detect_and_extract(ops1[ipl])
            ops = ops1[ipl]
            fpath = ops['save_path']
            print('----------- Total %0.2f sec.'%(time.time()-t11))

            ######### SPIKE DECONVOLUTION ###############
            F = np.load(os.path.join(fpath,'F.npy'))
            Fneu = np.load(os.path.join(fpath,'Fneu.npy'))
            if spikedetect:
                t11=time.time()
                print('----------- SPIKE DECONVOLUTION')
                dF = F - ops['neucoeff']*Fneu
                dF = dcnv.preprocess(dF,ops)
                spks = dcnv.oasis(dF, ops)
                np.save(os.path.join(ops['save_path'],'spks.npy'), spks)
                print('----------- Total %0.2f sec.'%(time.time()-t11))
            else:
                print("WARNING: skipping spike detection (ops['spikedetect']=False)")
                spks = np.zeros_like(F)
                np.save(os.path.join(ops['save_path'],'spks.npy'), spks)

            # save as matlab file
            if 'save_mat' in ops and ops['save_mat']:
                stat = np.load(os.path.join(fpath,'stat.npy'), allow_pickle=True)
                iscell = np.load(os.path.join(fpath,'iscell.npy'))
                matpath = os.path.join(ops['save_path'],'Fall.mat')
                savemat(matpath, {'stat': stat,
                                     'ops': ops,
                                     'F': F,
                                     'Fneu': Fneu,
                                     'spks': spks,
                                     'iscell': iscell})
            
                
        else:
            print("WARNING: skipping cell detection (ops['roidetect']=False)")
        print('Plane %d processed in %0.2f sec (can open in GUI).'%(ipl,time.time()-t1))
        print('total = %0.2f sec.'%(time.time()-t0))
        ipl += 1 #len(ipl)

    # save final ops1 with all planes
    np.save(fpathops1, ops1)

    #### COMBINE PLANES or FIELDS OF VIEW ####
    if len(ops1)>1 and ops1[0]['combined'] and roidetect:
        save.combined(ops1)
    
    # save to NWB
    if 'save_NWB' in ops and ops['save_NWB']:
        nwb.save(ops1)

    # running a clean up script
    if 'clean_script' in ops1[0]:
        print('running clean-up script')
        os.system('python '+ ops['clean_script'] + ' ' + fpathops1)

    i=0
    for ops in ops1:
        if 'move_bin' in ops and ops['move_bin'] and ops['save_path']!=ops['fast_disk']:
            shutil.move(ops['reg_file'], os.path.join(ops['save_path'], 'data.bin'))
            if ops['nchannels']>1:
                shutil.move(ops['reg_file_chan2'], os.path.join(ops['save_path'], 'data_chan2.bin'))
            if 'raw_file' in ops:
                shutil.move(ops['raw_file'], os.path.join(ops['save_path'], 'data_raw.bin'))
                if ops['nchannels']>1:
                    shutil.move(ops['raw_file_chan2'], os.path.join(ops['save_path'], 'data_chan2_raw.bin'))
            if i==0:
                print('moving binary files to save_path')
        elif ('delete_bin' in ops) and ops['delete_bin']:
            os.remove(ops['reg_file'])
            if ops['nchannels']>1:
                os.remove(ops['reg_file_chan2'])
            if i==0:
                print('deleting binary files')
        i+=1

    print('TOTAL RUNTIME %0.2f sec'%(time.time()-t0))
    return ops1
