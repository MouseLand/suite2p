"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import glob
import os
import copy
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
import numpy as np
from natsort import natsorted



EXTS = {"tif": ["*.tif", "*.tiff", "*.TIF", "*.TIFF"],
        "h5": ["*.h5", "*.hdf5", "*.mesc"],
        "sbx": ["*.sbx"],
        "nd2": ["*.nd2"],
        "dcimg": ["*.dcimg"],
        "movie": ["*.mp4", "*.avi"]}


def find_files_open_binaries(settings):
    return None

def list_files(froot, look_one_level_down, exts):
    """ get list of files with exts in folder froot + one level down maybe
    """
    fs = []
    first_files = np.zeros(0, "bool")
    for e in exts:
        lpath = os.path.join(froot, e)
        fs.extend(glob.glob(lpath))
    fs = natsorted(set(fs))
    if len(fs) > 0:
        first_files = np.zeros(len(fs), "bool")
        first_files[0] = True
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
        first_files = np.zeros(len(fsall), dtype="bool")
        first_files[0] = True
        logger.info(f"** Found {len(fsall)} files - converting to binary **")
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
            logger.info(f"Could not find any {EXTS[input_format]} files in {data_path}")
            raise Exception("no files found")
        else:
            first_files = np.array(first_files).astype("bool")
            logger.info(f"** Found {len(fsall)} files - converting to binary **")
    return fsall, first_files

def init_dbs(db0):
    """ initializes db files for each plane in recording

    Parameters
    ----------
    settings : dictionary
        "nplanes", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw"
        + (if mesoscope) "dy", "dx", "lines"

    Returns
    -------
        db1 : list of dictionaries
            adds fields "save_path0", "reg_file"
            (depending on settings: "raw_file", "reg_file_chan2", "raw_file_chan2")

    """
    nplanes = db0["nplanes"]
    nchannels = db0["nchannels"]
    keep_movie_raw = db0["keep_movie_raw"]
    nfolders = nplanes
    iplane = db0.get("iplane", np.arange(0, nplanes))
    if "lines" in db0 and db0["lines"] is not None and len(db0["lines"]) > 0:
        nrois = len(db0["lines"])
        db0["nrois"] = nrois
        nfolders *= nrois
        logger.info(f"NOTE: nplanes={nplanes}, nrois={nrois} => nfolders = {nfolders}")
        # replicate lines across planes if nplanes > 1
        if nplanes > 1:
            lines0, dy0, dx0 = db0["lines"].copy(), db0["dy"].copy(), db0["dx"].copy()
            dy0, dx0 = np.array(dy0), np.array(dx0)
            dy = np.tile(dy0[np.newaxis, :], (nplanes, 1)).flatten()
            dx = np.tile(dx0[np.newaxis, :], (nplanes, 1)).flatten()
            lines = []
            [lines.extend(lines0) for _ in range(nplanes)]
            iroi = np.tile(np.arange(nrois)[np.newaxis,:], (nplanes, 1)).flatten()
            iplane = np.tile(np.arange(nplanes)[:, np.newaxis], (1, nrois)).flatten()
        else:
            lines, dy, dx = db0["lines"].copy(), db0["dy"].copy(), db0["dx"].copy()
            iroi = np.arange(nrois)
            iplane = np.zeros(nrois, "int")
    
    dbs = []
    if db0.get("fast_disk", None) is None or len(db0["fast_disk"]) == 0:
        db0["fast_disk"] = db0["save_path0"]
    fast_disk = db0["fast_disk"]
    
    # compile dbs into list across planes
    for j in range(0, nfolders):
        db = copy.deepcopy(db0)
        db["save_path"] = os.path.join(db["save_path0"], db["save_folder"], f"plane{j}")
        fast_disk = os.path.join(db["fast_disk"], "suite2p", f"plane{j}")
        db["fast_disk"] = fast_disk
        db["settings_path"] = os.path.join(db["save_path"], "settings.npy")
        db["db_path"] = os.path.join(db["save_path"], "db.npy")
        db["reg_file"] = os.path.join(fast_disk, "data.bin")
        if keep_movie_raw:
            db["raw_file"] = os.path.join(fast_disk, "data_raw.bin")
        if "lines" in db and db["lines"] is not None and len(db["lines"]) > 0:
            db["lines"], db["dy"], db["dx"] = lines[j], dy[j], dx[j]
            db["iroi"] = iroi[j]
        db["iplane"] = iplane[j]
        if nchannels > 1:
            db["reg_file_chan2"] = os.path.join(fast_disk, "data_chan2.bin")
            if keep_movie_raw:
                db["raw_file_chan2"] = os.path.join(fast_disk, "data_chan2_raw.bin")
        
        os.makedirs(db["fast_disk"], exist_ok=True)
        os.makedirs(db["save_path"], exist_ok=True)
        dbs.append(db)
    return dbs

def init_settings(db, settings):
    """ initializes settings files for each plane in recording

    Parameters
    ----------
    settings : dictionary
        "nplanes", "save_path", "save_folder", "fast_disk", "nchannels", "keep_movie_raw"
        + (if mesoscope) "dy", "dx", "lines"

    Returns
    -------
        settings1 : list of dictionaries
            adds fields "save_path0", "reg_file"
            (depending on settings: "raw_file", "reg_file_chan2", "raw_file_chan2")

    """

    nplanes = settings["nplanes"]
    nchannels = settings["nchannels"]
    if "lines" in settings:
        lines = settings["lines"]
    if "iplane" in settings:
        iplane = settings["iplane"]
        #settings["nplanes"] = len(settings["lines"])
    settings1 = []
    if ("fast_disk" not in db) or len(db["fast_disk"]) == 0:
        db["fast_disk"] = db["save_path0"]
    fast_disk = settings["fast_disk"]
    # for mesoscope recording FOV locations
    if "dy" in settings and settings["dy"] != "":
        dy = settings["dy"]
        dx = settings["dx"]
    # compile settings into list across planes
    for j in range(0, nplanes):
        if len(settings["save_folder"]) > 0:
            settings["save_path"] = os.path.join(settings["save_path0"], settings["save_folder"],
                                            "plane%d" % j)
        else:
            settings["save_path"] = os.path.join(settings["save_path0"], "suite2p", "plane%d" % j)

        fast_disk = os.path.join(settings["fast_disk"], "suite2p", "plane%d" % j)
        settings["settings_path"] = os.path.join(settings["save_path"], "settings.npy")
        settings["reg_file"] = os.path.join(fast_disk, "data.bin")
        if "keep_movie_raw" in settings and settings["keep_movie_raw"]:
            settings["raw_file"] = os.path.join(fast_disk, "data_raw.bin")
        if "lines" in settings:
            settings["lines"] = lines[j]
        if "iplane" in settings:
            settings["iplane"] = iplane[j]
        if nchannels > 1:
            settings["reg_file_chan2"] = os.path.join(fast_disk, "data_chan2.bin")
            if "keep_movie_raw" in settings and settings["keep_movie_raw"]:
                settings["raw_file_chan2"] = os.path.join(fast_disk, "data_chan2_raw.bin")
        if "dy" in settings and settings["dy"] != "":
            settings["dy"] = dy[j]
            settings["dx"] = dx[j]
        if not os.path.isdir(fast_disk):
            os.makedirs(fast_disk)
        if not os.path.isdir(settings["save_path"]):
            os.makedirs(settings["save_path"])
        settings1.append(settings.copy())
    return settings1


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
