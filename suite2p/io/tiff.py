"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import gc
import glob
import json
import math
import os
import time
from typing import Union, Tuple, Optional
import logging 
logger = logging.getLogger(__name__)

import numpy as np
from tifffile import imread, TiffFile, TiffWriter

from . import utils

try:
    from ScanImageTiffReader import ScanImageTiffReader
    HAS_SCANIMAGE = True
except ImportError:
    ScanImageTiffReader = None
    HAS_SCANIMAGE = False

def open_tiff(file: str,
              sktiff: bool) -> Tuple[Union[TiffFile, ScanImageTiffReader], int]:
    """ Returns image and its length from tiff file with either ScanImageTiffReader or tifffile, based on "sktiff" """
    if sktiff:
        tif = TiffFile(file)
        Ltif = len(tif.pages)
    else:
        tif = ScanImageTiffReader(file)
        Ltif = 1 if len(tif.shape()) < 3 else tif.shape()[0]  # single page tiffs
    return tif, Ltif


def use_sktiff_reader(tiff_filename, batch_size: Optional[int] = None) -> bool:
    """Returns False if ScanImageTiffReader works on the tiff file, else True (in which case use tifffile)."""
    if HAS_SCANIMAGE:
        try:
            with ScanImageTiffReader(tiff_filename) as tif:
                tif.data() if len(tif.shape()) < 3 else tif.data(
                    beg=0, end=np.minimum(batch_size,
                                          tif.shape()[0] - 1))
            return False
        except:
            logger.info(
                "NOTE: ScanImageTiffReader not working for this tiff type, using tifffile"
            )
            return True
    else:
        logger.info("NOTE: ScanImageTiffReader not installed, using tifffile")
        return True

def read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff):
    # tiff reading
    if ix >= Ltif:
        return None
    nfr = min(Ltif - ix, batch_size)
    if use_sktiff:
        im = imread(file, key=range(ix, ix + nfr))
    elif Ltif == 1:
        im = tif.data()
    else:
        im = tif.data(beg=ix, end=ix + nfr)
    # for single-page tiffs, add 1st dim
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=0)

    # check if uint16
    if im.dtype.type == np.uint16:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type == np.int32:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type != np.int16:
        im = im.astype(np.int16)

    if im.shape[0] > nfr:
        im = im[:nfr, :, :]

    return im

def tiff_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """  finds tiff files and writes them to binaries

    Parameters
    ----------
    settings : dictionary
        "nplanes", "data_path", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        settings : dictionary of first plane
            settings["reg_file"] or settings["raw_file"] is created binary
            assigns keys "Ly", "Lx", "tiffreader", "first_tiffs",
            "frames_per_folder", "nframes", "meanImg", "meanImg_chan2"

    """

    t0 = time.time()
        
    fs = dbs[0]["file_list"]
    first_files = dbs[0]["first_files"]

    # try tiff readers
    batch_size = dbs[0]["batch_size"]
    use_sktiff = True if dbs[0]["force_sktiff"] else use_sktiff_reader(fs[0], batch_size=batch_size)

    nplanes, nchannels = dbs[0]["nplanes"], dbs[0]["nchannels"]
    # processing for multiple ROIs (nrois > 1 if mesoscope recording)
    nrois = dbs[0].get("nrois", 1)
    batch_size = nplanes * nchannels * math.ceil(batch_size / (nplanes * nchannels))

    # loop over all tiffs
    which_folder = -1
    ntotal = 0
    for ifile, file in enumerate(fs):
        # open tiff
        tif, Ltif = open_tiff(file, use_sktiff)
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        if first_files[ifile]:
            which_folder += 1
            iplane = 0
        ix = 0
        while 1:
            im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
            if im is None:
                break          
            nframes = im.shape[0]
            for j in range(0, nplanes):
                if ifile == 0 and ix == 0:
                    Ly, Lx = im.shape[1], im.shape[2]
                    for k in range(nrois):
                        jk = j*nrois + k
                        Ly = (dbs[jk]["lines"][-1] + 1 - dbs[jk]["lines"][0] 
                              if nrois > 1 else Ly)
                        dbs[jk]["Ly"], dbs[jk]["Lx"] = Ly, Lx
                        dbs[jk]["nframes"] = 0
                        dbs[jk]["frames_per_file"] = np.zeros(len(fs), "int")
                        dbs[jk]["frames_per_folder"] = np.zeros(first_files.sum(), "int")
                        dbs[jk]["meanImg"] = np.zeros((Ly, Lx), "float64")
                        if nchannels > 1:
                            dbs[jk]["meanImg_chan2"] = np.zeros((Ly, Lx), "float64")
                i0 = nchannels * ((iplane + j) % nplanes)
                if nchannels > 1:
                    nfunc = dbs[jk]["functional_chan"] - 1
                else:
                    nfunc = 0
                im2write = im[int(i0) + nfunc:nframes:nplanes * nchannels]

                for k in range(nrois):
                    jk = j*nrois + k
                    if nrois > 1:
                        imk = im2write[:, dbs[jk]["lines"][0] : dbs[jk]["lines"][-1] + 1]
                    else:
                        imk = im2write
                    reg_file[jk].write(bytearray(imk))
                    dbs[jk]["meanImg"] += imk.sum(axis=0).astype("float64")
                    dbs[jk]["nframes"] += imk.shape[0]
                    dbs[jk]["frames_per_file"][ifile] += imk.shape[0]
                    dbs[jk]["frames_per_folder"][which_folder] += imk.shape[0]
                    
                if nchannels > 1:
                    im2write = im[int(i0) + 1 - nfunc:nframes:nplanes * nchannels]
                    for k in range(nrois):
                        jk = j*nrois + k
                        if nrois > 1:
                            imk = im2write[:, dbs[jk]["lines"][0] : dbs[jk]["lines"][-1] + 1]
                        else:
                            imk = im2write
                        reg_file_chan2[jk].write(bytearray(imk))
                        dbs[jk]["meanImg_chan2"] += imk.sum(axis=0).astype("float64")
            iplane = (iplane - nframes / nchannels) % nplanes
            ix += nframes
            ntotal += nframes
            if ntotal % (batch_size * 4) == 0:
                logger.info("%d frames of binary, time %0.2f sec." %
                      (ntotal, time.time() - t0))
        gc.collect()
    # write settings files
    for db in dbs:
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
    
    # close all binary files and write settings files
    for jk in range(0, nplanes * nrois):
        reg_file[jk].close()
        if nchannels > 1:
            reg_file_chan2[jk].close()
    return dbs

def ome_to_binary(settings):
    """
    converts ome.tiff to *.bin file for non-interleaved red channel recordings
    assumes SINGLE-PAGE tiffs where first channel has string "Ch1"
    and also SINGLE FOLDER

    Parameters
    ----------
    settings : dictionary
        keys nplanes, nchannels, data_path, look_one_level_down, reg_file

    Returns
    -------
    settings : dictionary of first plane
        creates binaries settings["reg_file"]
        assigns keys: tiffreader, first_tiffs, frames_per_folder, nframes, meanImg, meanImg_chan2
    """
    t0 = time.time()

    # copy settings to list where each element is settings for each plane
    settings1 = utils.init_settings(settings)
    nplanes = settings1[0]["nplanes"]

    # open all binary files for writing and look for tiffs in all requested folders
    settings1, fs, reg_file, reg_file_chan2 = utils.find_files_open_binaries(settings1, False)
    settings = settings1[0]
    batch_size = settings["batch_size"]
    use_sktiff = not HAS_SCANIMAGE

    fs_Ch1, fs_Ch2 = [], []
    for f in fs:
        if f.find("Ch1") > -1:
            if settings["functional_chan"] == 1:
                fs_Ch1.append(f)
            else:
                fs_Ch2.append(f)
        else:
            if settings["functional_chan"] == 1:
                fs_Ch2.append(f)
            else:
                fs_Ch1.append(f)

    if len(fs_Ch2) == 0:
        settings1[0]["nchannels"] = 1
    nchannels = settings1[0]["nchannels"]
    logger.info(f"nchannels = {nchannels}")
    
    # loop over all tiffs
    TiffReader = ScanImageTiffReader if HAS_SCANIMAGE else TiffFile
    with TiffReader(fs_Ch1[0]) as tif:
        if HAS_SCANIMAGE:
            n_pages = tif.shape()[0] if len(tif.shape()) > 2 else 1
            shape = tif.shape()[-2:]
        else:
            n_pages = len(tif.pages)
            im0 = tif.pages[0].asarray()
            shape = im0.shape
    
    for settings1_0 in settings1:
        settings1_0["nframes"] = 0
        settings1_0["frames_per_folder"][0] = 0
        settings1_0["frames_per_file"] = np.ones(len(fs_Ch1), "int") if n_pages==1 else np.zeros(len(fs_Ch1), "int")
        settings1_0["meanImg"] = np.zeros(shape, np.float32)
        if nchannels > 1:
            settings1_0["meanImg_chan2"] = np.zeros(shape, np.float32)

    bruker_bidirectional = settings.get("bruker_bidirectional", False)
    iplanes = np.arange(0, nplanes)
    if not bruker_bidirectional:
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(fs_Ch1) / nplanes))).flatten()
        iplanes = iplanes[:len(fs_Ch1)]
    else:
        iplanes = np.hstack((iplanes, iplanes[::-1]))
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(fs_Ch1) / (2 * nplanes)))).flatten()
        iplanes = iplanes[:len(fs_Ch1)]

    itot = 0
    for ik, file in enumerate(fs_Ch1):
        ip = iplanes[ik]
        # read tiff
        if n_pages==1:    
            with TiffReader(file) as tif:
                im = tif.data()  if HAS_SCANIMAGE else tif.pages[0].asarray()
            if im.dtype.type == np.uint16:
                im = (im // 2)
            im = im.astype(np.int16)

            # write to binary
            settings1[ip]["nframes"] += 1
            settings1[ip]["frames_per_folder"][0] += 1
            settings1[ip]["meanImg"] += im.astype(np.float32)
            reg_file[ip].write(bytearray(im))
            #gc.collect()
        else:
            tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
            # keep track of the plane identity of the first frame (channel identity is assumed always 0)
            ix = 0
            while 1:
                im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                if im is None:
                    break          
                nframes = im.shape[0]
                ix += nframes
                itot += nframes
                reg_file[ip].write(bytearray(im))
                settings1[ip]["meanImg"] += im.astype(np.float32).sum(axis=0)
                settings1[ip]["nframes"] += im.shape[0]
                settings1[ip]["frames_per_file"][ik] += nframes
                settings1[ip]["frames_per_folder"][0] += nframes
                if itot % 1000 == 0:
                    logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - t0))
                gc.collect()            

    if nchannels > 1:
        itot = 0
        for ik, file in enumerate(fs_Ch2):
            ip = iplanes[ik]
            if n_pages==1:
                with TiffReader(file) as tif:
                    im = tif.data() if HAS_SCANIMAGE else tif.pages[0].asarray()
                if im.dtype.type == np.uint16:
                    im = (im // 2)
                im = im.astype(np.int16)
                settings1[ip]["meanImg_chan2"] += im.astype(np.float32)
                reg_file_chan2[ip].write(bytearray(im))
            else:
                tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
                ix = 0
                while 1:
                    im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                    if im is None:
                        break          
                    nframes = im.shape[0]
                    ix += nframes
                    itot += nframes
                    settings1[ip]["meanImg_chan2"] += im.astype(np.float32).sum(axis=0)
                    reg_file_chan2[ip].write(bytearray(im))
                    if itot % 1000 == 0:
                        logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - t0))
                    gc.collect()

    # write settings files
    do_registration = settings["do_registration"]
    for settings in settings1:
        settings["Ly"], settings["Lx"] = shape
        if not do_registration:
            settings["yrange"] = np.array([0, settings["Ly"]])
            settings["xrange"] = np.array([0, settings["Lx"]])
        settings["meanImg"] /= settings["nframes"]
        if nchannels > 1:
            settings["meanImg_chan2"] /= settings["nframes"]
        np.save(settings["settings_path"], settings)
    # close all binary files and write settings files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return settings1[0]
