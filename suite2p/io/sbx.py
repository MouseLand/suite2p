"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os

import numpy as np
import logging 
logger = logging.getLogger(__name__)


from .utils import init_settings, find_files_open_binaries

try:
    from sbxreader import sbx_memmap
    HAS_SBX = True
except:
    HAS_SBX = False
    

def sbx_to_binary(settings, ndeadcols=-1, ndeadrows=0):
    """  finds scanbox files and writes them to binaries

    Parameters
    ----------
    settings : dictionary
        "nplanes", "data_path", "save_path", "save_folder", "fast_disk",
        "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        settings : dictionary of first plane
            "Ly", "Lx", settings["reg_file"] or settings["raw_file"] is created binary

    """
    if not HAS_SBX:
        raise ImportError("sbxreader is required for this file type, please 'pip install sbxreader'")

    settings1 = init_settings(settings)
    # the following should be taken from the metadata and not needed but the files are initialized before...
    nplanes = settings1[0]["nplanes"]
    nchannels = settings1[0]["nchannels"]
    # open all binary files for writing
    settings1, sbxlist, reg_file, reg_file_chan2 = find_files_open_binaries(settings1)
    iall = 0
    for j in range(settings1[0]["nplanes"]):
        settings1[j]["nframes_per_folder"] = np.zeros(len(sbxlist), np.int32)
    ik = 0
    if "sbx_ndeadcols" in settings1[0].keys():
        ndeadcols = int(settings1[0]["sbx_ndeadcols"])
    if "sbx_ndeadrows" in settings1[0].keys():
        ndeadrows = int(settings1[0]["sbx_ndeadrows"])

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

    settings1[0]["sbx_ndeadcols"] = ndeadcols
    settings1[0]["sbx_ndeadrows"] = ndeadrows

    for ifile, sbxfname in enumerate(sbxlist):
        f = sbx_memmap(sbxfname)
        nplanes = f.shape[1]
        nchannels = f.shape[2]
        nframes = f.shape[0]
        iblocks = np.arange(0, nframes, settings1[0]["batch_size"])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data = nframes x nplanes x nchannels x pixels x pixels
        if nchannels > 1:
            nfunc = settings1[0]["functional_chan"] - 1
        else:
            nfunc = 0
        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array(f[onset:offset, :, :, ndeadrows:, ndeadcols:]) // 2
            im = im.astype(np.int16)
            im2mean = im.mean(axis=0).astype(np.float32) / len(iblocks)
            for ichan in range(nchannels):
                nframes = im.shape[0]
                im2write = im[:, :, ichan, :, :]
                for j in range(0, nplanes):
                    if iall == 0:
                        settings1[j]["meanImg"] = np.zeros((im.shape[3], im.shape[4]),
                                                      np.float32)
                        if nchannels > 1:
                            settings1[j]["meanImg_chan2"] = np.zeros(
                                (im.shape[3], im.shape[4]), np.float32)
                        settings1[j]["nframes"] = 0
                    if ichan == nfunc:
                        settings1[j]["meanImg"] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file[j].write(
                            bytearray(im2write[:, j, :, :].astype("int16")))
                    else:
                        settings1[j]["meanImg_chan2"] += np.squeeze(im2mean[j, ichan, :, :])
                        reg_file_chan2[j].write(
                            bytearray(im2write[:, j, :, :].astype("int16")))

                    settings1[j]["nframes"] += im2write.shape[0]
                    settings1[j]["nframes_per_folder"][ifile] += im2write.shape[0]
            ik += nframes
            iall += nframes

    # write settings files
    do_registration = settings1[0]["do_registration"]
    do_nonrigid = settings1[0]["nonrigid"]
    for settings in settings1:
        settings["Ly"] = im.shape[3]
        settings["Lx"] = im.shape[4]
        if not do_registration:
            settings["yrange"] = np.array([0, settings["Ly"]])
            settings["xrange"] = np.array([0, settings["Lx"]])
        #settings["meanImg"] /= settings["nframes"]
        #if nchannels>1:
        #    settings["meanImg_chan2"] /= settings["nframes"]
        np.save(settings["settings_path"], settings)
    # close all binary files and write settings files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return settings1[0]
