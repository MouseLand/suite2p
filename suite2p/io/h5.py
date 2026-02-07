"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import math

try:
    import h5py
    HAS_H5PY=True
except:
    HAS_H5PY=False
import numpy as np
import os



def h5py_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """
    Read HDF5 files and write interleaved plane/channel data to binary files.

    Iterates over all HDF5 files listed in `dbs[0]["file_list"]`, de-interleaves
    planes and channels, and writes each plane's frames to the corresponding binary
    file. Supports 3D, 4D, and 5D HDF5 datasets (higher-dimensional data is
    flattened to frames x Ly x Lx before de-interleaving).

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane. Must contain keys "file_list",
        "nplanes", "nchannels", "batch_size", "h5py_key", and "functional_chan".
        Updated in-place with "Ly", "Lx", "nframes", "nframes_per_folder",
        "meanImg", and "meanImg_chan2".
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
    if not HAS_H5PY:
        raise ImportError("h5py is required for this file type, please 'pip install h5py'")

    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    h5list = dbs[0]["file_list"]

    keys = dbs[0]["h5py_key"]
    if isinstance(keys, str):
        keys = [keys]
    iall = 0
    for j in range(nplanes):
        dbs[j]["nframes_per_folder"] = np.zeros(len(h5list), np.int32)

    for ih5, h5 in enumerate(h5list):
        with h5py.File(h5, "r") as f:
            # if h5py data is 5D or 4D instead of 3D, assume that
            # data = (nchan x) (nframes x) nplanes x pixels x pixels
            # 5D/4D data is flattened to process the same way as interleaved data
            for key in keys:
                hdims = f[key].ndim
                # keep track of the plane identity of the first frame (channel identity is assumed always 0)
                ncp = nplanes * nchannels
                nbatch = ncp * math.ceil(dbs[0]["batch_size"] / ncp)
                nframes_all = f[key].shape[
                    0] if hdims == 3 else f[key].shape[0] * f[key].shape[1]
                nbatch = min(nbatch, nframes_all)
                nfunc = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0
                # loop over all tiffs
                ik = 0
                while 1:
                    if hdims == 3:
                        irange = np.arange(ik, min(ik + nbatch, nframes_all), 1)
                        if irange.size == 0:
                            break
                        im = f[key][irange, :, :]
                    else:
                        irange = np.arange(
                            ik / ncp, min(ik / ncp + nbatch / ncp, nframes_all / ncp),
                            1)
                        if irange.size == 0:
                            break
                        im = f[key][irange, ...]
                        if im.ndim == 5 and im.shape[0] == nchannels:
                            im = im.transpose((1, 0, 2, 3, 4))
                        # flatten to frames x pixels x pixels
                        im = np.reshape(im, (-1, im.shape[-2], im.shape[-1]))
                    nframes = im.shape[0]
                    if type(im[0, 0, 0]) == np.uint16:
                        im = im / 2
                    for j in range(0, nplanes):
                        if iall == 0:
                            dbs[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                          np.float32)
                            if nchannels > 1:
                                dbs[j]["meanImg_chan2"] = np.zeros(
                                    (im.shape[1], im.shape[2]), np.float32)
                            dbs[j]["nframes"] = 0
                        i0 = nchannels * ((j) % nplanes)
                        im2write = im[np.arange(int(i0) +
                                                nfunc, nframes, ncp), :, :].astype(
                                                    np.int16)
                        reg_file[j].write(bytearray(im2write))
                        dbs[j]["meanImg"] += im2write.astype(np.float32).sum(axis=0)
                        if nchannels > 1:
                            im2write = im[np.arange(int(i0) + 1 -
                                                    nfunc, nframes, ncp), :, :].astype(
                                                        np.int16)
                            reg_file_chan2[j].write(bytearray(im2write))
                            dbs[j]["meanImg_chan2"] += im2write.astype(
                                np.float32).sum(axis=0)
                        dbs[j]["nframes"] += im2write.shape[0]
                        dbs[j]["nframes_per_folder"][ih5] += im2write.shape[0]
                    ik += nframes
                    iall += nframes

    # update dbs with image dimensions and mean images
    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = im2write.shape[1]
        db["Lx"] = im2write.shape[2]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        # Save db and settings to each plane folder
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs
