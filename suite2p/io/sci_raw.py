# ---------------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------------#
# Added by Ahmed Jamali to handle SciScan .raw files - date 10.04.2024

import gc
import json
import math
import os
import time

import numpy as np
from . import utils


try:
    import sci_raw
    HAS_sci_raw = True
except ImportError:
    HAS_sci_raw  = False


def AJ_sci_raw_to_binary(ops):
    """
    Convert raw data to binary format.

    Args:
        ops (dict): Dictionary containing the parameters for conversion.

    Returns:
        dict: Dictionary containing the converted binary data.
    """

    t0 = time.time()
    ops1 = utils.init_ops(ops)
    nplanes = ops1[0]["nplanes"]
    nchannels = ops1[0]["nchannels"]

    ops1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1, False)
    ops = ops1[0]

    batch_size = ops["batch_size"]
    batch_size = nplanes * nchannels * math.ceil(batch_size / (nplanes * nchannels))

    which_folder = -1
    ntotal = 0
    for ik, file in enumerate(fs):
        raw_data = np.memmap(file, dtype=np.uint16, mode='r')  # Memory map the raw file
        Lraw = raw_data.size // (ops['Ly'] * ops['Lx']) # removed (* nchannels) from the end of the line

        
        if ops["first_raws"][ik]:
            which_folder += 1
            iplane = 0
        ix = 0

        while 1:
            if ix >= Lraw:
                break
            nfr = min(Lraw - ix, batch_size)
            im = raw_data[ix * ops['Ly'] * ops['Lx'] : (ix + nfr) * ops['Ly'] * ops['Lx']] # Removed: * nchannels from the end of the line and before colon
            im = im.reshape((nfr, ops['Ly'], ops['Lx']))
            im = im.byteswap().newbyteorder('<')  # Corrects the byteorder from big-endian to little-endian

            if im.dtype.type == np.uint16:
                im = (im // 2).astype(np.int16)

            if im.shape[0] > nfr:
                im = im[:nfr, :, :]
            nframes = im.shape[0]

            for j in range(0,nplanes):
                if ik == 0 and ix == 0:
                    ops1[j]["nframes"] = 0
                    ops1[j]["frames_per_file"] = np.zeros((len(fs),), dtype=int)
                    ops1[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                  np.float32)
                    if nchannels > 1:
                        ops1[j]["meanImg_chan2"] = np.zeros((im.shape[1], im.shape[2]), 
                                                            np.float32)

                i0 = nchannels * ((iplane + j) % nplanes)
                if nchannels > 1:
                    nfunc = ops["functional_chan"] - 1
                else:
                    nfunc = 0                
                im2write = im[int(i0) + nfunc:nframes:nplanes * nchannels]
                
                reg_file[j].write(bytearray(im2write))

                ops1[j]["meanImg"] += im2write.astype(np.float32).sum(axis=(0)) # .squeeze() Sum over all frames to get the mean image
                ops1[j]["nframes"] += im2write.shape[0]
                ops1[j]["frames_per_file"][ik] += im2write.shape[0]
                ops1[j]["frames_per_folder"][which_folder] += im2write.shape[0]

                if nchannels > 1:
                    im2write = im[int(i0) + 1 - nfunc:nframes:nplanes * nchannels]
                    reg_file_chan2[j].write(bytearray(im2write))
                    ops1[j]["meanImg_chan2"] += im2write.mean(axis=0)

            iplane = (iplane - nframes / nchannels) % nplanes
            ix += nframes
            ntotal += nframes
            if ntotal % (batch_size * 4) == 0:
                print("%d frames of binary, time %0.2f sec." % (ntotal, time.time() - t0))
        gc.collect()

    do_registration = ops["do_registration"]
    for ops in ops1:
        ops["Ly"], ops["Lx"] = ops["meanImg"].shape
        ops["yrange"] = np.array([0, ops["Ly"]])
        ops["xrange"] = np.array([0, ops["Lx"]])
        ops["meanImg"] /= ops["nframes"]
        if nchannels > 1:
            ops["meanImg_chan2"] /= ops["nframes"]
        np.save(ops["ops_path"], ops)

    for j in range(0,nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()

    return ops1[0]
