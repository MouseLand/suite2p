"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sbxreader import sbx_memmap

    HAS_SBX = True
except:
    HAS_SBX = False


def sbx_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """finds scanbox files and writes them to binaries

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane. Must contain keys "file_list",
        "nplanes", "nchannels", "batch_size", "functional_chan", "sbx_ndeadcols",
        and "sbx_ndeadrows". Updated in-place with "Ly", "Lx", "nframes",
        "nframes_per_folder", "meanImg", and "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries with image dimensions, frame counts, and
        mean images populated.
    """
    if not HAS_SBX:
        raise ImportError("sbxreader is required for this file type, please 'pip install sbxreader'")

    sbxlist = dbs[0]["file_list"]
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    batch_size = dbs[0].get("batch_size", 500)
    functional_chan = dbs[0].get("functional_chan", 1)
    ndeadcols = int(dbs[0].get("sbx_ndeadcols", -1))
    ndeadrows = int(dbs[0].get("sbx_ndeadrows", 0))

    iall = 0
    for j in range(nplanes):
        dbs[j]["nframes_per_folder"] = np.zeros(len(sbxlist), np.int32)

    if ndeadcols == -1 or ndeadrows == -1:
        tmpsbx = sbx_memmap(sbxlist[0])
        # do not remove dead rows in non-multiplane mode; artifact is larger for bigger ETL jumps
        if ndeadrows == -1:
            if nplanes > 1:
                colprofile = np.array(np.mean(tmpsbx[0][0][0], axis=1))
                ndeadrows = np.argmax(np.diff(colprofile)) + 1
            else:
                ndeadrows = 0
        # do not remove dead columns in unidirectional scanning mode
        if ndeadcols == -1:
            if tmpsbx.metadata["scanning_mode"] == "bidirectional":
                ndeadcols = tmpsbx.ndeadcols
            else:
                ndeadcols = 0
        del tmpsbx
        logger.info("Removing {0} dead columns while loading sbx data.".format(ndeadcols))
        logger.info("Removing {0} dead rows while loading sbx data.".format(ndeadrows))

    for db in dbs:
        db["sbx_ndeadcols"] = ndeadcols
        db["sbx_ndeadrows"] = ndeadrows

    for ifile, sbxfname in enumerate(sbxlist):
        f = sbx_memmap(sbxfname)
        nplanes_f = f.shape[1]
        nchannels_f = f.shape[2]
        nframes = f.shape[0]
        iblocks = np.arange(0, nframes, batch_size)
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data shape: nframes x nplanes x nchannels x Ly x Lx
        nfunc = functional_chan - 1 if nchannels_f > 1 else 0
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array(f[onset:offset, :, :, ndeadrows:, ndeadcols:]) // 2
            im = im.astype(np.int16)
            nframes_batch = im.shape[0]
            if iall == 0:
                for j in range(nplanes_f):
                    dbs[j]["meanImg"] = np.zeros((im.shape[3], im.shape[4]), np.float32)
                    if nchannels_f > 1:
                        dbs[j]["meanImg_chan2"] = np.zeros(
                            (im.shape[3], im.shape[4]), np.float32
                        )
                    dbs[j]["nframes"] = 0
            for ichan in range(nchannels_f):
                im2write = im[:, :, ichan, :, :]
                for j in range(nplanes_f):
                    plane_frames = im2write[:, j, :, :].astype(np.int16)
                    if ichan == nfunc:
                        dbs[j]["meanImg"] += plane_frames.astype(np.float32).sum(axis=0)
                        reg_file[j].write(bytearray(plane_frames))
                        dbs[j]["nframes"] += nframes_batch
                        dbs[j]["nframes_per_folder"][ifile] += nframes_batch
                    else:
                        dbs[j]["meanImg_chan2"] += plane_frames.astype(np.float32).sum(axis=0)
                        reg_file_chan2[j].write(bytearray(plane_frames))
            iall += nframes_batch

    # update dbs with image dimensions and mean images
    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = im.shape[3]
        db["Lx"] = im.shape[4]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs
