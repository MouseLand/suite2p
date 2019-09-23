import numpy as np
import os
from natsort import natsorted
import glob

def list_h5(ops):
    froot = os.path.dirname(ops['h5py'])
    lpath = os.path.join(froot, "*.h5")
    fs = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.hdf5")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    return fs

def list_tifs(froot, look_one_level_down):
    """ get list of tiffs in folder froot + one level down maybe
    """
    first_tiffs = []
    lpath = os.path.join(froot, "*.tif")
    fs  = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.tiff")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    if len(fs) > 0:
        first_tiffs.extend(np.zeros((len(fs),), 'bool'))
        first_tiffs[0] = True
    lfs = len(fs)
    if look_one_level_down:
        fdir = glob.glob(os.path.join(froot, "*", ""))
        for folder_down in fdir:
            lpath = os.path.join(folder_down, "*.tif")
            fs3 = natsorted(glob.glob(lpath))
            lpath = os.path.join(folder_down, "*.tiff")
            fs4 = natsorted(glob.glob(lpath))
            fs.extend(fs3)
            fs.extend(fs4)
            if len(fs3)+len(fs4) > 0:
                first_tiffs.extend(np.zeros((len(fs3)+len(fs4),), 'bool'))
                first_tiffs[lfs] = True
                lfs = len(fs)
    return fs, first_tiffs

def get_tif_list(ops):
    """ make list of tiffs to process
    if ops['subfolders'], then all tiffs ops['data_path'][0] / ops['subfolders'] / *.tif
    if ops['look_one_level_down'], then all tiffs in all folders + one level down
    if ops['tiff_list'], then ops['data_path'][0] / ops['tiff_list'] ONLY
    """
    froot = ops['data_path']
    # use a user-specified list of tiffs
    if 'tiff_list' in ops:
        fsall = []
        for tif in ops['tiff_list']:
            fsall.append(os.path.join(froot[0], tif))
        ops['first_tiffs'] = np.zeros((len(fsall),), dtype=np.bool)
        ops['first_tiffs'][0] = True
        print('** Found %d tifs - converting to binary **'%(len(fsall)))
    else:
        if len(froot)==1:
            if 'subfolders' in ops and len(ops['subfolders'])>0:
                fold_list = []
                for folder_down in ops['subfolders']:
                    fold = os.path.join(froot[0], folder_down)
                    fold_list.append(fold)
            else:
                fold_list = ops['data_path']
        else:
            fold_list = froot
        fsall = []
        nfs = 0
        first_tiffs = []
        for k,fld in enumerate(fold_list):
            fs, ftiffs = list_tifs(fld, ops['look_one_level_down'])
            fsall.extend(fs)
            first_tiffs.extend(ftiffs)
        if len(fs)==0:
            print('Could not find any tiffs')
            raise Exception('no tiffs')
        else:
            ops['first_tiffs'] = np.array(first_tiffs)
            print('** Found %d tifs - converting to binary **'%(len(fsall)))
            #print('Found %d tifs'%(len(fsall)))
    return fsall, ops

def find_files_open_binaries(ops1, ish5):
    """  finds tiffs or h5 files and opens binaries for writing

    Parameters
    ----------
    ops1 : list of dictionaries
        'keep_movie_raw', 'data_path', 'look_one_level_down', 'reg_file'...

    Returns
    -------
        ops1 : list of dictionaries
            adds fields 'filelist', 'first_tiffs', opens binaries

    """

    reg_file = []
    reg_file_chan2=[]


    for ops in ops1:
        nchannels = ops['nchannels']
        if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
            reg_file.append(open(ops['raw_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['raw_file_chan2'], 'wb'))
        else:
            reg_file.append(open(ops['reg_file'], 'wb'))
            if nchannels>1:
                reg_file_chan2.append(open(ops['reg_file_chan2'], 'wb'))

    if ish5:
        # find h5's
        if ops1[0]['look_one_level_down']:
            fs = list_h5(ops1[0])
            print('NOTE: using a list of h5 files:')
            print(fs)
        else:
            fs = [ops1[0]['h5py']]
    else:
        # find tiffs
        fs, ops2 = get_tif_list(ops1[0])
        for ops in ops1:
            ops['first_tiffs'] = ops2['first_tiffs']
            ops['frames_per_folder'] = np.zeros((ops2['first_tiffs'].sum(),), np.int32)
            ops['filelist'] = fs
    return ops1, fs, reg_file, reg_file_chan2


def init_ops(ops):
    """ initializes ops files for each plane in recording

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'save_path', 'save_folder', 'fast_disk', 'nchannels', 'keep_movie_raw'
        + (if mesoscope) 'dy', 'dx', 'lines'

    Returns
    -------
        ops1 : list of dictionaries
            adds fields 'save_path0', 'reg_file'
            (depending on ops: 'raw_file', 'reg_file_chan2', 'raw_file_chan2')

    """

    nplanes = ops['nplanes']
    nchannels = ops['nchannels']
    if 'lines' in ops:
        lines = ops['lines']
    if 'iplane' in ops:
        iplane = ops['iplane']
        #ops['nplanes'] = len(ops['lines'])
    ops1 = []
    if ('fast_disk' not in ops) or len(ops['fast_disk'])==0:
        ops['fast_disk'] = ops['save_path0']
    fast_disk = ops['fast_disk']
    # for mesoscope recording FOV locations
    if 'dy' in ops and ops['dy']!='':
        dy = ops['dy']
        dx = ops['dx']
    # compile ops into list across planes
    for j in range(0,nplanes):
        if len(ops['save_folder']) > 0:
            ops['save_path'] = os.path.join(ops['save_path0'], ops['save_folder'], 'plane%d'%j)
        else:
            ops['save_path'] = os.path.join(ops['save_path0'], 'suite2p', 'plane%d'%j)

        if ('fast_disk' not in ops) or len(ops['fast_disk'])==0:
            ops['fast_disk'] = ops['save_path0']
        ops['fast_disk'] = os.path.join(fast_disk, 'suite2p', 'plane%d'%j)
        ops['ops_path'] = os.path.join(ops['save_path'],'ops.npy')
        ops['reg_file'] = os.path.join(ops['fast_disk'], 'data.bin')
        if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
            ops['raw_file'] = os.path.join(ops['fast_disk'], 'data_raw.bin')
        if 'lines' in ops:
            ops['lines'] = lines[j]
        if 'iplane' in ops:
            ops['iplane'] = iplane[j]
        if nchannels>1:
            ops['reg_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2.bin')
            if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
                ops['raw_file_chan2'] = os.path.join(ops['fast_disk'], 'data_chan2_raw.bin')
        if 'dy' in ops and ops['dy']!='':
            ops['dy'] = dy[j]
            ops['dx'] = dx[j]
        if not os.path.isdir(ops['fast_disk']):
            os.makedirs(ops['fast_disk'])
        if not os.path.isdir(ops['save_path']):
            os.makedirs(ops['save_path'])
        ops1.append(ops.copy())
    return ops1
