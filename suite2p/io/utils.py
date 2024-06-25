"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import glob
import os
import copy
from pathlib import Path

import numpy as np
from natsort import natsorted

from .. import default_ops

EXTS = {"tif": ["*.tif", "*.tiff", "*.TIF", "*.TIFF"],
        "h5": ["*.h5", "*.hdf5", "*.mesc"],
        "sbx": ["*.sbx"],
        "nd2": ["*.nd2"],
        "dcimg": ["*.dcimg"],
        "movie": ["*.mp4", "*.avi"]}

def find_files_open_binaries(ops):
    return None

def list_files(froot, look_one_level_down, exts):
    """ get list of files with exts in folder froot + one level down maybe
    """
    fs = []
    for e in exts:
        lpath = os.path.join(froot, e)
        fs.extend(glob.glob(lpath))
    fs = natsorted(set(fs))
    if len(fs) > 0:
        first_files = np.zeros((len(fs),), "bool")
        first_files[0] = True
    else:
        first_files = np.zeros(0, "bool")
    lfs = len(fs)
    if look_one_level_down:
        fdir = natsorted(glob.glob(os.path.join(froot, "*/")))
        for folder_down in fdir:
            fsnew = []
            for e in exts:
                lpath = os.path.join(folder_down, e)
                fsnew.extend(glob.glob(lpath))
            fsnew = natsorted(set(fsnew))
            if len(fsnew) > 0:
                fs.extend(fsnew)
                first_files = np.append(first_files, np.zeros((len(fsnew),), "bool"))
                first_files[lfs] = True
                lfs = len(fs)
    return fs, first_files

def get_file_list(db):
    """ make list of files to process
    if db["subfolders"], then all files db["data_path"][0] / db["subfolders"] / *.tif
    if db["look_one_level_down"], then all files in all folders + one level down
    if db["file_list"], then db["data_path"][0] / db["filst_list"] ONLY
    """
    data_path = db["data_path"]
    input_format = db.get("input_format", "tif")
    # use a user-specified list of tiffs
    if db.get("file_list", None) is not None:
        fsall = []
        for f in db["file_list"]:
            fsall.append(os.path.join(data_path[0], f))
        db["first_files"] = np.zeros((len(fsall),), dtype="bool")
        db["first_files"][0] = True
        print(f"** Found {len(fsall)} files - converting to binary **")
    else:
        if len(data_path) == 1 and db.get("subfolders", None) is not None:
                fold_list = []
                for folder_down in db["subfolders"]:
                    fold = os.path.join(data_path[0], folder_down)
                    fold_list.append(fold)
        else:
            fold_list = data_path
        fsall = []
        first_files = []
        for k, fld in enumerate(fold_list):
            fs, firsts = list_files(fld, db["look_one_level_down"],
                                    EXTS[input_format])
            fsall.extend(fs)
            first_files.extend(list(firsts))
        if len(fsall) == 0:
            print(f"Could not find any {EXTS[input_format]} files in {data_path}")
            raise Exception("no files found")
        else:
            first_files = np.array(first_files).astype("bool")
            print(f"** Found {len(fsall)} files - converting to binary **")
    return fsall, first_files

def open_binaries(dbs):
    """  finds imaging files and opens binaries for writing

    Parameters
    ----------
    dbs : list of dictionaries
        w/ keys "reg_file", "nchannels", "raw_file", "raw_file_chan2", "reg_file_chan2"

    Returns
    -------
        ops1 : list of dictionaries
            adds fields "filelist", "first_tiffs", opens binaries

    """
    reg_file = []
    reg_file_chan2 = []

    for db in dbs:
        nchannels = db["nchannels"]
        if db["keep_movie_raw"]:
            reg_file.append(open(db["raw_file"], "wb"))
            if nchannels > 1:
                reg_file_chan2.append(open(db["raw_file_chan2"], "wb"))
        else:
            reg_file.append(open(db["reg_file"], "wb"))
            if nchannels > 1:
                reg_file_chan2.append(open(db["reg_file_chan2"], "wb"))
    return reg_file, reg_file_chan2


def init_dbs(db0):
    """ initializes db files for each plane in recording

    Parameters
    ----------
    ops : dictionary
        "nplanes", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw"
        + (if mesoscope) "dy", "dx", "lines"

    Returns
    -------
        db1 : list of dictionaries
            adds fields "save_path0", "reg_file"
            (depending on ops: "raw_file", "reg_file_chan2", "raw_file_chan2")

    """
    nplanes = db0["nplanes"]
    nchannels = db0["nchannels"]
    keep_movie_raw = db0["keep_movie_raw"]
    if "lines" in db0:
        lines = db0["lines"]
    if "iplane" in db0:
        iplane = db0["iplane"]
        #ops["nplanes"] = len(ops["lines"])
    dbs = []
    if db0.get("fast_disk", None) is None or len(db0["fast_disk"]) == 0:
        db0["fast_disk"] = db0["save_path0"]
    fast_disk = db0["fast_disk"]
    # for mesoscope recording FOV locations
    if "dy" in db0 and db0["dy"] != "":
        dy = db0["dy"]
        dx = db0["dx"]
    # compile dbs into list across planes
    for j in range(0, nplanes):
        db = copy.deepcopy(db0)
        db["save_path"] = os.path.join(db["save_path0"], db["save_folder"], f"plane{j}")
        fast_disk = os.path.join(db["fast_disk"], "suite2p", f"plane{j}")
        db["fast_disk"] = fast_disk
        db["ops_path"] = os.path.join(db["save_path"], "ops.npy")
        db["db_path"] = os.path.join(db["save_path"], "db.npy")
        db["reg_file"] = os.path.join(fast_disk, "data.bin")
        if keep_movie_raw:
            db["raw_file"] = os.path.join(fast_disk, "data_raw.bin")
        if "lines" in db:
            db["lines"] = lines[j]
        if "iplane" in db:
            db["iplane"] = iplane[j]
        if nchannels > 1:
            db["reg_file_chan2"] = os.path.join(fast_disk, "data_chan2.bin")
            if keep_movie_raw:
                db["raw_file_chan2"] = os.path.join(fast_disk, "data_chan2_raw.bin")
        if "dy" in db and db["dy"] != "":
            db["dy"] = dy[j]
            db["dx"] = dx[j]
        os.makedirs(db["fast_disk"], exist_ok=True)
        os.makedirs(db["save_path"], exist_ok=True)
        dbs.append(db)
    return dbs

def init_ops(db, ops):
    """ initializes ops files for each plane in recording

    Parameters
    ----------
    ops : dictionary
        "nplanes", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw"
        + (if mesoscope) "dy", "dx", "lines"

    Returns
    -------
        ops1 : list of dictionaries
            adds fields "save_path0", "reg_file"
            (depending on ops: "raw_file", "reg_file_chan2", "raw_file_chan2")

    """

    nplanes = ops["nplanes"]
    nchannels = ops["nchannels"]
    if "lines" in ops:
        lines = ops["lines"]
    if "iplane" in ops:
        iplane = ops["iplane"]
        #ops["nplanes"] = len(ops["lines"])
    ops1 = []
    if ("fast_disk" not in db) or len(db["fast_disk"]) == 0:
        db["fast_disk"] = db["save_path0"]
    fast_disk = ops["fast_disk"]
    # for mesoscope recording FOV locations
    if "dy" in ops and ops["dy"] != "":
        dy = ops["dy"]
        dx = ops["dx"]
    # compile ops into list across planes
    for j in range(0, nplanes):
        if len(ops["save_folder"]) > 0:
            ops["save_path"] = os.path.join(ops["save_path0"], ops["save_folder"],
                                            "plane%d" % j)
        else:
            ops["save_path"] = os.path.join(ops["save_path0"], "suite2p", "plane%d" % j)

        fast_disk = os.path.join(ops["fast_disk"], "suite2p", "plane%d" % j)
        ops["ops_path"] = os.path.join(ops["save_path"], "ops.npy")
        ops["reg_file"] = os.path.join(fast_disk, "data.bin")
        if "keep_movie_raw" in ops and ops["keep_movie_raw"]:
            ops["raw_file"] = os.path.join(fast_disk, "data_raw.bin")
        if "lines" in ops:
            ops["lines"] = lines[j]
        if "iplane" in ops:
            ops["iplane"] = iplane[j]
        if nchannels > 1:
            ops["reg_file_chan2"] = os.path.join(fast_disk, "data_chan2.bin")
            if "keep_movie_raw" in ops and ops["keep_movie_raw"]:
                ops["raw_file_chan2"] = os.path.join(fast_disk, "data_chan2_raw.bin")
        if "dy" in ops and ops["dy"] != "":
            ops["dy"] = dy[j]
            ops["dx"] = dx[j]
        if not os.path.isdir(fast_disk):
            os.makedirs(fast_disk)
        if not os.path.isdir(ops["save_path"]):
            os.makedirs(ops["save_path"])
        ops1.append(ops.copy())
    return ops1


def get_suite2p_path(path: Path) -> Path:
    """Find the root `suite2p` folder in the `path` variable"""

    path = Path(path)  # In case `path` is a string

    # Cheap sanity check
    if "suite2p" in str(path):
        # Walk the folders in path backwards
        for path_idx in range(len(path.parts) - 1, 0, -1):
            if path.parts[path_idx] == "suite2p":
                new_path = Path(path.parts[0])
                for path_part in path.parts[1:path_idx + 1]:
                    new_path = new_path.joinpath(path_part)
                break
    else:
        raise FileNotFoundError("The `suite2p` folder was not found in path")
    return new_path
