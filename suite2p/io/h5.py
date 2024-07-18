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

from .utils import init_settings, find_files_open_binaries


def h5py_to_binary(settings):
    """  finds h5 files and writes them to binaries

    Parameters
    ----------
    settings : dictionary
        "nplanes", "h5_path", "h5_key", "save_path", "save_folder", "fast_disk",
        "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        settings : dictionary of first plane
            "Ly", "Lx", settings["reg_file"] or settings["raw_file"] is created binary

    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for this file type, please 'pip install h5py'")

    settings1 = init_settings(settings)

    nplanes = settings1[0]["nplanes"]
    nchannels = settings1[0]["nchannels"]

    # open all binary files for writing
    settings1, h5list, reg_file, reg_file_chan2 = find_files_open_binaries(settings1, True)
    for settings in settings1:
        if not settings.get("data_path"):
            settings["data_path"] = [os.path.dirname(settings["h5py"])]
    settings1[0]["h5list"] = h5list
    keys = settings1[0]["h5py_key"]
    if isinstance(keys, str):
        keys = [keys]
    iall = 0
    for j in range(settings["nplanes"]):
        settings1[j]["nframes_per_folder"] = np.zeros(len(h5list), np.int32)

    for ih5, h5 in enumerate(h5list):
        with h5py.File(h5, "r") as f:
            # if h5py data is 5D or 4D instead of 3D, assume that
            # data = (nchan x) (nframes x) nplanes x pixels x pixels
            # 5D/4D data is flattened to process the same way as interleaved data
            for key in keys:
                hdims = f[key].ndim
                # keep track of the plane identity of the first frame (channel identity is assumed always 0)
                ncp = nplanes * nchannels
                nbatch = ncp * math.ceil(settings1[0]["batch_size"] / ncp)
                nframes_all = f[key].shape[
                    0] if hdims == 3 else f[key].shape[0] * f[key].shape[1]
                nbatch = min(nbatch, nframes_all)
                nfunc = settings["functional_chan"] - 1 if nchannels > 1 else 0
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
                            settings1[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                          np.float32)
                            if nchannels > 1:
                                settings1[j]["meanImg_chan2"] = np.zeros(
                                    (im.shape[1], im.shape[2]), np.float32)
                            settings1[j]["nframes"] = 0
                        i0 = nchannels * ((j) % nplanes)
                        im2write = im[np.arange(int(i0) +
                                                nfunc, nframes, ncp), :, :].astype(
                                                    np.int16)
                        reg_file[j].write(bytearray(im2write))
                        settings1[j]["meanImg"] += im2write.astype(np.float32).sum(axis=0)
                        if nchannels > 1:
                            im2write = im[np.arange(int(i0) + 1 -
                                                    nfunc, nframes, ncp), :, :].astype(
                                                        np.int16)
                            reg_file_chan2[j].write(bytearray(im2write))
                            settings1[j]["meanImg_chan2"] += im2write.astype(
                                np.float32).sum(axis=0)
                        settings1[j]["nframes"] += im2write.shape[0]
                        settings1[j]["nframes_per_folder"][ih5] += im2write.shape[0]
                    ik += nframes
                    iall += nframes

    # write settings files
    do_registration = settings1[0]["do_registration"]
    for settings in settings1:
        settings["Ly"] = im2write.shape[1]
        settings["Lx"] = im2write.shape[2]
        if not do_registration:
            settings["yrange"] = np.array([0, settings["Ly"]])
            settings["xrange"] = np.array([0, settings["Lx"]])
        settings["meanImg"] /= settings["nframes"]
        if nchannels > 1:
            settings["meanImg_chan2"] /= settings["nframes"]
        np.save(settings["settings_path"], settings)
    # close all binary files and write settings files
    for j in range(nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return settings1[0]
