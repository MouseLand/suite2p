import math

import h5py
import numpy as np
import os

from .utils import init_ops, find_files_open_binaries


def h5py_to_binary(ops):
    """  finds h5 files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        'nplanes', 'h5_path', 'h5_key', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops : dictionary of first plane
            'Ly', 'Lx', ops['reg_file'] or ops['raw_file'] is created binary

    """
    ops1 = init_ops(ops)

    nplanes = ops1[0]['nplanes']
    nchannels = ops1[0]['nchannels']

    # open all binary files for writing
    ops1, h5list, reg_file, reg_file_chan2 = find_files_open_binaries(ops1, True)
    for ops in ops1:
        if not ops.get('data_path'):
            ops['data_path'] = [os.path.dirname(ops['h5py'])]
    ops1[0]['h5list'] = h5list
    keys = ops1[0]['h5py_key']
    if isinstance(keys, str):
        keys = [keys]
    iall = 0
    for j in range(ops['nplanes']):
        ops1[j]['nframes_per_folder'] = np.zeros(len(h5list), np.int32)

    for ih5,h5 in enumerate(h5list):
        with h5py.File(h5, 'r') as f:
            # if h5py data is 4D instead of 3D, assume that
            # data = nframes x nplanes x pixels x pixels
            for key in keys:
                hdims = f[key].ndim
                # keep track of the plane identity of the first frame (channel identity is assumed always 0)
                nbatch = nplanes*nchannels*math.ceil(ops1[0]['batch_size']/(nplanes*nchannels))
                nframes_all = f[key].shape[0] if hdims == 3 else f[key].shape[0] * f[key].shape[1]
                nbatch = min(nbatch, nframes_all)
                nfunc = ops['functional_chan'] - 1 if nchannels > 1 else 0
                # loop over all tiffs
                ik = 0
                while 1:
                    if hdims==3:
                        irange = np.arange(ik, min(ik+nbatch, nframes_all), 1)
                        if irange.size==0:
                            break
                        im = f[key][irange, :, :]
                    else:
                        irange = np.arange(ik/nplanes, min(ik/nplanes+nbatch/nplanes, nframes_all/nplanes), 1)
                        if irange.size==0:
                            break
                        im = f[key][irange,:,:,:]
                        im = np.reshape(im, (im.shape[0]*nplanes,im.shape[2],im.shape[3]))
                    nframes = im.shape[0]
                    if type(im[0,0,0]) == np.uint16:
                        im = im / 2
                    for j in range(0,nplanes):
                        if iall==0:
                            ops1[j]['meanImg'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                            if nchannels>1:
                                ops1[j]['meanImg_chan2'] = np.zeros((im.shape[1],im.shape[2]),np.float32)
                            ops1[j]['nframes'] = 0
                        i0 = nchannels * ((j)%nplanes)
                        im2write = im[np.arange(int(i0)+nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                        reg_file[j].write(bytearray(im2write))
                        ops1[j]['meanImg'] += im2write.astype(np.float32).sum(axis=0)
                        if nchannels>1:
                            im2write = im[np.arange(int(i0)+1-nfunc, nframes, nplanes*nchannels),:,:].astype(np.int16)
                            reg_file_chan2[j].write(bytearray(im2write))
                            ops1[j]['meanImg_chan2'] += im2write.astype(np.float32).sum(axis=0)
                        ops1[j]['nframes'] += im2write.shape[0]
                        ops1[j]['nframes_per_folder'][ih5] += im2write.shape[0]
                    ik += nframes
                    iall += nframes

    # write ops files
    do_registration = ops1[0]['do_registration']
    for ops in ops1:
        ops['Ly'] = im2write.shape[1]
        ops['Lx'] = im2write.shape[2]
        if not do_registration:
            ops['yrange'] = np.array([0,ops['Ly']])
            ops['xrange'] = np.array([0,ops['Lx']])
        ops['meanImg'] /= ops['nframes']
        if nchannels > 1:
            ops['meanImg_chan2'] /= ops['nframes']
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return ops1[0]
