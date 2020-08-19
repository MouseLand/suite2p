import glob
import os

import numpy as np
from natsort import natsorted


def search_for_ext(rootdir, extension = 'tif', look_one_level_down=False):
    filepaths = []
    if os.path.isdir(rootdir):
        # search root dir
        tmp = glob.glob(os.path.join(rootdir,'*.'+extension))
        if len(tmp):
            filepaths.extend([t for t in natsorted(tmp)])
        # search one level down
        if look_one_level_down:
            dirs = natsorted(os.listdir(rootdir))
            for d in dirs:
                if os.path.isdir(os.path.join(rootdir,d)):
                    tmp = glob.glob(os.path.join(rootdir, d, '*.'+extension))
                    if len(tmp):
                        filepaths.extend([t for t in natsorted(tmp)])
    if len(filepaths):
        return filepaths
    else:
        raise OSError('Could not find files, check path [{0}]'.format(rootdir))

def get_sbx_list(ops):
    """ make list of scanbox files to process
    if ops['subfolders'], then all tiffs ops['data_path'][0] / ops['subfolders'] / *.sbx
    if ops['look_one_level_down'], then all tiffs in all folders + one level down
    TODO: Implement "tiff_list" functionality
    """
    froot = ops['data_path']
    # use a user-specified list of tiffs
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
    for k,fld in enumerate(fold_list):
        fs = search_for_ext(fld,
                            extension = 'sbx',
                            look_one_level_down = ops['look_one_level_down'])
        fsall.extend(fs)
    if len(fsall)==0:
        print(fold_list)
        raise Exception('No files, check path.')
    else:
        print('** Found %d sbx - converting to binary **'%(len(fsall)))
    return fsall, ops

def list_h5(ops):
    froot = os.path.dirname(ops['h5py'])
    lpath = os.path.join(froot, "*.h5")
    fs = natsorted(glob.glob(lpath))
    lpath = os.path.join(froot, "*.hdf5")
    fs2 = natsorted(glob.glob(lpath))
    fs.extend(fs2)
    return fs

def list_files(froot, look_one_level_down, exts):
    """ get list of files with exts in folder froot + one level down maybe
    """
    fs = []
    for e in exts:
        lpath = os.path.join(froot, e)
        fs.extend(glob.glob(lpath))
    fs = natsorted(set(fs))
    if len(fs) > 0:
        first_tiffs = np.zeros((len(fs),), np.bool)
        first_tiffs[0] = True
    else:
        first_tiffs = np.zeros(0, np.bool)
    lfs = len(fs)
    if look_one_level_down:
        fdir = glob.glob(os.path.join(froot, "*/"))
        for folder_down in fdir:
            fsnew = []
            for e in exts:
                lpath = os.path.join(folder_down, e)
                fsnew.extend(glob.glob(lpath))
            fsnew = natsorted(set(fsnew))
            if len(fsnew) > 0:
                fs.extend(fsnew)
                first_tiffs = np.append(first_tiffs, np.zeros((len(fsnew),), np.bool))
                first_tiffs[lfs] = True
                lfs = len(fs)
    return fs, first_tiffs

def get_h5_list(ops):
    """ make list of h5 files to process
    if ops['look_one_level_down'], then all h5's in all folders + one level down
    """
    froot = ops['data_path']
    fold_list = ops['data_path']
    fsall = []
    nfs = 0
    first_tiffs = []
    for k,fld in enumerate(fold_list):
        fs, ftiffs = list_files(fld, ops['look_one_level_down'],
                                ["*.h5", "*.hdf5"])
        fsall.extend(fs)
        first_tiffs.extend(list(ftiffs))
    if len(fs)==0:
        print('Could not find any h5 files')
        raise Exception('no h5s')
    else:
        ops['first_tiffs'] = np.array(first_tiffs).astype(np.bool)
        print('** Found %d h5 files - converting to binary **'%(len(fsall)))
        #print('Found %d tifs'%(len(fsall)))
    return fsall, ops


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
            fs, ftiffs = list_files(fld, ops['look_one_level_down'],
                                    ["*.tif", "*.tiff", "*.TIF", "*.TIFF"])
            fsall.extend(fs)
            first_tiffs.extend(list(ftiffs))
        if len(fsall)==0:
            print('Could not find any tiffs')
            raise Exception('no tiffs')
        else:
            ops['first_tiffs'] = np.array(first_tiffs).astype(np.bool)
            print('** Found %d tifs - converting to binary **'%(len(fsall)))
    return fsall, ops

def find_files_open_binaries(ops1, ish5=False):
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

        if 'input_format' in ops.keys():
            input_format = ops['input_format']
        else:
            input_format = 'tif'
    if ish5:
        input_format = 'h5'
    print(input_format)
    if input_format == 'h5':
        if len(ops1[0]['data_path'])>0:
            fs, ops2 = get_h5_list(ops1[0])
            print('NOTE: using a list of h5 files:')
            print(fs)
        # find h5's
        else:
            if ops1[0]['look_one_level_down']:
                fs = list_h5(ops1[0])
                print('NOTE: using a list of h5 files:')
                print(fs)
            else:
                fs = [ops1[0]['h5py']]
    elif input_format == 'sbx':
        # find sbx
        fs, ops2 = get_sbx_list(ops1[0])
        print('Scanbox files:')
        print('\n'.join(fs))
    else:
        # find tiffs
        fs, ops2 = get_tif_list(ops1[0])
        for ops in ops1:
            ops['first_tiffs'] = ops2['first_tiffs']
            ops['frames_per_folder'] = np.zeros((ops2['first_tiffs'].sum(),), np.int32)
    for ops in ops1:
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
