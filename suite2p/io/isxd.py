"""
Copyright © 2024 Inscopix, Inc., a Bruker company. Authored by Ludovic Bellier.
"""
# import os
import numpy as np
from . import utils

try:
    import isx
    HAS_ISX = True
except (ModuleNotFoundError, ImportError):
    HAS_ISX = False


def isxd_to_binary(ops):
    """  finds Inscopix isxd files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        "nplanes", "data_path", "save_path", "save_folder", "fast_disk",
        "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        ops : dictionary of first plane
            "Ly", "Lx", ops["reg_file"] or ops["raw_file"] is created binary

    """
    if not HAS_ISX:
        raise ImportError("Inscopix isx is required for this file type, please 'pip install isx'")

    ops1 = utils.init_ops(ops)
    # the following should be taken from the metadata and not needed but the files are initialized before...
    nplanes = ops1[0]["nplanes"]
    nchannels = ops1[0]["nchannels"]
    # open all binary files for writing
    ops1, file_list, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1)
    iall = 0
    for j in range(ops1[0]["nplanes"]):
        ops1[j]["nframes_per_folder"] = np.zeros(len(file_list), np.int32)
    ik = 0

    for ifile, fname in enumerate(file_list):
        f = isx.Movie.read(fname)
        nplanes = 1  #f.shape[1]
        nchannels = 1  #f.shape[2]
        nframes = f.timing.num_samples
        iblocks = np.arange(0, nframes, ops1[0]["batch_size"])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data = nframes x nplanes x nchannels x pixels x pixels
        if nchannels > 1:
            nfunc = ops1[0]["functional_chan"] - 1
        else:
            nfunc = 0
        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array([f.get_frame_data(x) for x in np.arange(onset, offset)])
            im2mean = im.mean(axis=0).astype(np.float32) / len(iblocks)
            for ichan in range(nchannels):
                nframes = im.shape[0]
                im2write = im[:]
                for j in range(0, nplanes):
                    if iall == 0:
                        ops1[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                      np.float32)
                        if nchannels > 1:
                            ops1[j]["meanImg_chan2"] = np.zeros(
                                (im.shape[1], im.shape[2]), np.float32)
                        ops1[j]["nframes"] = 0
                    if ichan == nfunc:
                        ops1[j]["meanImg"] += np.squeeze(im2mean)
                        reg_file[j].write(
                            bytearray(im2write[:].astype("int16")))
                    else:
                        ops1[j]["meanImg_chan2"] += np.squeeze(im2mean)
                        reg_file_chan2[j].write(
                            bytearray(im2write[:].astype("int16")))

                    ops1[j]["nframes"] += im2write.shape[0]
                    ops1[j]["nframes_per_folder"][ifile] += im2write.shape[0]
            ik += nframes
            iall += nframes

    # write ops files
    do_registration = ops1[0]["do_registration"]
    do_nonrigid = ops1[0]["nonrigid"]
    for ops in ops1:
        ops["Ly"] = im.shape[1]
        ops["Lx"] = im.shape[2]
        if not do_registration:
            ops["yrange"] = np.array([0, ops["Ly"]])
            ops["xrange"] = np.array([0, ops["Lx"]])
        #ops["meanImg"] /= ops["nframes"]
        #if nchannels>1:
        #    ops["meanImg_chan2"] /= ops["nframes"]
        np.save(ops["ops_path"], ops)
    # close all binary files and write ops files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return ops1[0]
