import os
import gc
import math
import time
import numpy as np
from . import utils
import nd2


def nd2_to_binary(ops):
    """  finds nd2 files and writes them to binaries

    Parameters
    ----------
    ops: dictionary
        'nplanes', 'data_path', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops : dictionary of first plane
            ops['reg_file'] or ops['raw_file'] is created binary
            assigns keys 'Ly', 'Lx', 'tiffreader', 'first_tiffs',
            'nframes', 'meanImg', 'meanImg_chan2'
    """

    t0 = time.time()
    # copy ops to list where each element is ops for each plane
    ops1 = utils.init_ops(ops)

    # open all binary files for writing
    # look for nd2s in all requested folders
    ops1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1, False)
    ops = ops1[0]

    # loop over all nd2 files
    iall = 0
    ik = 0
    for file_name in fs:
        # open nd2
        nd2_file = nd2.ND2File(file_name)
        im = nd2_file.asarray()

        # expand dimensions to have [Time (T), Depth (Z), Channel (C), Height (Y), Width (X)].
        if 'T' not in nd2_file.sizes:
            im = np.expand_dims(im, 0)
        if 'C' not in nd2_file.sizes:
            im = np.expand_dims(im, -3)
        if 'Z' not in nd2_file.sizes:
            im = np.expand_dims(im, 1)

        nplanes = nd2_file.sizes['Z'] if 'Z' in nd2_file.sizes else 1
        nchannels = nd2_file.sizes['C'] if 'C' in nd2_file.sizes else 1
        nframes = nd2_file.sizes['T'] if 'T' in nd2_file.sizes else 1

        iblocks = np.arange(0, nframes, ops1[0]['batch_size'])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        if nchannels > 1:
            nfunc = ops1[0]['functional_chan'] - 1
        else:
            nfunc = 0

        if im.dtype.type == np.uint16:
            im = (im // 2).astype(np.int16)
        elif im.dtype.type == np.int32:
            im = (im // 2).astype(np.int16)
        elif im.dtype.type != np.int16:
            im = im.astype(np.int16)

        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im_p = np.array(im[onset:offset, :, :, :, :])
            im2mean = im_p.mean(axis = 0).astype(np.float32) / len(iblocks)
            for ichan in range(nchannels):
                nframes = im_p.shape[0]
                im2write = im_p[:, :, ichan, :, :]
                for j in range(0, nplanes):
                    if iall == 0:
                        ops1[j]['meanImg'] = np.zeros((im_p.shape[3], im_p.shape[4]), np.float32)
                        if nchannels>1:
                            ops1[j]['meanImg_chan2'] = np.zeros((im_p.shape[3], im_p.shape[4]), np.float32)
                        ops1[j]['nframes'] = 0
                    if ichan == nfunc:
                        ops1[j]['meanImg'] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file[j].write(bytearray(im2write[:, j, :, :].astype('int16')))
                    else:
                        ops1[j]['meanImg_chan2'] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file_chan2[j].write(bytearray(im2write[:, j, :, :].astype('int16')))
                        
                    ops1[j]['nframes'] += im2write.shape[0]
            ik += nframes
            iall += nframes

    # write ops files
    do_registration = ops1[0]['do_registration']
    for ops in ops1:
        ops['Ly'] = im.shape[3]
        ops['Lx'] = im.shape[4]
        if not do_registration:
            ops['yrange'] = np.array([0, ops['Ly']])
            ops['xrange'] = np.array([0, ops['Lx']])
        np.save(ops['ops_path'], ops)
    # close all binary files and write ops files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return ops1[0]
