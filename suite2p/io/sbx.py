"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os

import numpy as np
import logging 
logger = logging.getLogger(__name__)



try:
    from sbxreader import sbx_memmap
    HAS_SBX = True
except:
    HAS_SBX = False
    

def sbx_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """  finds scanbox files and writes them to binaries

    Parameters
    ----------
    dbs : list of dict
        Per-plane database dictionaries. Must contain "file_list", "first_files",
        "nplanes", "nchannels", "batch_size", "functional_chan". Updated in-place
        with "Ly", "Lx", "nframes", "frames_per_file", "frames_per_folder",
        "meanImg", and "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries.
    """
    if not HAS_SBX:
        raise ImportError("sbxreader is required for this file type, please 'pip install sbxreader'")

    sbxlist = dbs[0]["file_list"]
    first_files = dbs[0]["first_files"]
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]

    ndeadcols = int(dbs[0].get("sbx_ndeadcols", -1))
    ndeadrows = int(dbs[0].get("sbx_ndeadrows", 0))

    if ndeadcols == -1 or ndeadrows == -1:
        # compute dead rows and cols from the first file
        tmpsbx = sbx_memmap(sbxlist[0])
        # do not remove dead rows in non-multiplane mode
        # This number should be different for each plane since the artifact is larger
        # for larger ETL jumps.
        if nplanes > 1 and ndeadrows == -1:
            colprofile = np.array(np.mean(tmpsbx[0][0][0], axis=1))
            ndeadrows = np.argmax(np.diff(colprofile)) + 1
        else:
            ndeadrows = 0
        # do not remove dead columns in unidirectional scanning mode
        # do this only if ndeadcols is -1
        if tmpsbx.metadata["scanning_mode"] == "bidirectional" and ndeadcols == -1:
            ndeadcols = tmpsbx.ndeadcols
        else:
            ndeadcols = 0
        del tmpsbx
        logger.info("Removing {0} dead columns while loading sbx data.".format(ndeadcols))
        logger.info("Removing {0} dead rows while loading sbx data.".format(ndeadrows))

    for j in range(nplanes):
        dbs[j]["sbx_ndeadcols"] = ndeadcols
        dbs[j]["sbx_ndeadrows"] = ndeadrows

    iall = 0
    for ifile, sbxfname in enumerate(sbxlist):
        f = sbx_memmap(sbxfname)
        nplanes = f.shape[1]
        nchannels = f.shape[2]
        nframes = f.shape[0]
        iblocks = np.arange(0, nframes, dbs[0]["batch_size"])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data = nframes x nplanes x nchannels x pixels x pixels
        if nchannels > 1:
            nfunc = dbs[0]["functional_chan"] - 1
        else:
            nfunc = 0
        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array(f[onset:offset, :, :, ndeadrows:, ndeadcols:]) // 2
            im = im.astype(np.int16)
            im2mean = im.mean(axis=0).astype(np.float32) / len(iblocks)
            for ichan in range(nchannels):
                nframes_chunk = im.shape[0]
                im2write = im[:, :, ichan, :, :]
                for j in range(0, nplanes):
                    if iall == 0:
                        dbs[j]["Ly"] = im.shape[3]
                        dbs[j]["Lx"] = im.shape[4]
                        dbs[j]["meanImg"] = np.zeros((im.shape[3], im.shape[4]), np.float32)
                        if nchannels > 1:
                            dbs[j]["meanImg_chan2"] = np.zeros(
                                (im.shape[3], im.shape[4]), np.float32)
                        dbs[j]["nframes"] = 0
                        dbs[j]["frames_per_file"] = np.zeros(len(sbxlist), np.int32)
                        dbs[j]["frames_per_folder"] = np.zeros(first_files.sum(), np.int32)
                    if ichan == nfunc:
                        dbs[j]["meanImg"] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file[j].write(
                            bytearray(im2write[:, j, :, :].astype("int16")))
                    else:
                        dbs[j]["meanImg_chan2"] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file_chan2[j].write(
                            bytearray(im2write[:, j, :, :].astype("int16")))

                    dbs[j]["nframes"] += nframes_chunk
                    dbs[j]["frames_per_file"][ifile] += nframes_chunk
            iall += im.shape[0]

    return dbs
