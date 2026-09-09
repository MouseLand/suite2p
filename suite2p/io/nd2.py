"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import gc
import math
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import nd2
    HAS_ND2 = True
except ImportError:
    HAS_ND2 = False


def nd2_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """finds nd2 files and writes them to binaries

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
    if not HAS_ND2:
        raise ImportError("nd2 is required for this file type, please 'pip install nd2'")

    fs = dbs[0]["file_list"]
    first_files = dbs[0]["first_files"]
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    if nchannels > 2:
        raise ValueError(f"nd2 input supports at most 2 channels, got nchannels={nchannels}.")
    # batch_size counts interleaved frames like the tiff/h5 readers, so one chunk
    # of timepoints holds batch_size / (nplanes * nchannels) volumes
    tbatch = max(1, math.ceil(dbs[0]["batch_size"] / (nplanes * nchannels)))
    nfunc = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0

    t0 = time.time()
    iall = 0
    which_folder = -1
    for ifile, file_name in enumerate(fs):
        if first_files[ifile]:
            which_folder += 1
        with nd2.ND2File(file_name) as nd2_file:
            sizes = nd2_file.sizes
            valid_dimensions = "TZCYX"
            unknown = set(sizes) - set(valid_dimensions)
            if unknown:
                raise ValueError(f"Unknown dimensions {unknown} in file {file_name}.")
            nplanes_file = sizes.get("Z", 1)
            nchannels_file = sizes.get("C", 1)
            nframes = sizes.get("T", 1)
            if nplanes_file != nplanes or nchannels_file != nchannels:
                raise ValueError(
                    f"{file_name} has {nplanes_file} planes and {nchannels_file} channels, "
                    f"but db has nplanes={nplanes} and nchannels={nchannels}.")

            # lazy view reordered to T x Z x C x Y x X, inserting missing axes
            data = nd2_file.to_dask()
            data = data.transpose([list(sizes).index(d) for d in valid_dimensions if d in sizes])
            data = data[tuple(slice(None) if d in sizes else None for d in "TZC") + (Ellipsis,)]

            iblocks = np.arange(0, nframes, tbatch)
            if iblocks[-1] < nframes:
                iblocks = np.append(iblocks, nframes)
            for ichunk, onset in enumerate(iblocks[:-1]):
                offset = iblocks[ichunk + 1]
                im = np.asarray(data[onset:offset])
                if im.dtype.type == np.uint16:
                    im = (im // 2).astype(np.int16)
                elif im.dtype.type == np.int32:
                    if im.min() < -65536 or im.max() > 65535:
                        raise ValueError(
                            f"int32 values in {file_name} do not fit int16 after halving.")
                    im = (im // 2).astype(np.int16)
                elif im.dtype.type in (np.uint8, np.int16):
                    im = im.astype(np.int16)
                else:
                    raise ValueError(f"unsupported nd2 pixel dtype {im.dtype} in {file_name}.")
                nframes_chunk, _, _, Ly, Lx = im.shape
                for j in range(nplanes):
                    if iall == 0:
                        dbs[j]["Ly"] = Ly
                        dbs[j]["Lx"] = Lx
                        dbs[j]["meanImg"] = np.zeros((Ly, Lx), np.float32)
                        if nchannels > 1:
                            dbs[j]["meanImg_chan2"] = np.zeros((Ly, Lx), np.float32)
                        dbs[j]["nframes"] = 0
                        dbs[j]["frames_per_file"] = np.zeros(len(fs), np.int32)
                        dbs[j]["frames_per_folder"] = np.zeros(first_files.sum(), np.int32)
                    im2write = np.ascontiguousarray(im[:, j, nfunc])
                    reg_file[j].write(bytearray(im2write))
                    dbs[j]["meanImg"] += im2write.astype(np.float32).sum(axis=0)
                    if nchannels > 1:
                        im2write = np.ascontiguousarray(im[:, j, 1 - nfunc])
                        reg_file_chan2[j].write(bytearray(im2write))
                        dbs[j]["meanImg_chan2"] += im2write.astype(np.float32).sum(axis=0)
                    dbs[j]["nframes"] += nframes_chunk
                    dbs[j]["frames_per_file"][ifile] += nframes_chunk
                    dbs[j]["frames_per_folder"][which_folder] += nframes_chunk
                iall += nframes_chunk
                if (ichunk + 1) % 4 == 0 or offset == nframes:
                    logger.info("%d frames of binary, time %0.2f sec." % (iall, time.time() - t0))
        gc.collect()

    for db in dbs:
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs
