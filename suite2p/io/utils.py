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
        "bruker": ["*.ome.tif", "*.ome.TIF"],
        "movie": ["*.mp4", "*.avi"]}


def find_files_open_binaries(settings):
    """
    Find input files and open binary files for writing.

    Parameters
    ----------
    settings : dict
        Suite2p settings dictionary.

    Returns
    -------
    None
        No longer used!!
    """
    return None

def init_settings(settings):
    return None

def list_files(froot, look_one_level_down, exts):
    """
    Collect files matching the given extensions from a folder, optionally including subfolders.

    Parameters
    ----------
    froot : str
        Root directory to search for files.
    look_one_level_down : bool
        If True, also search immediate subdirectories of `froot`.
    exts : list of str
        Glob patterns to match (e.g. ["*.tif", "*.tiff"]).

    Returns
    -------
    fs : list of str
        Naturally sorted list of matching file paths.
    first_files : numpy.ndarray
        Boolean array of length len(fs), where True marks the first file from each
        folder (used to track folder boundaries).
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
    """
    Build the list of input files to process from the database configuration.

    Supports three modes: an explicit file list (db["file_list"]), subfolder-based
    discovery (db["subfolders"]), or recursive search with optional one-level-down
    lookup (db["look_one_level_down"]).

    Parameters
    ----------
    db : dict
        Database dictionary. Must contain "data_path" (list of str). Optionally
        contains "file_list" (list of str), "subfolders" (list of str),
        "look_one_level_down" (bool), and "input_format" (str, default "tif").

    Returns
    -------
    fsall : list of str
        List of all file paths to process.
    first_files : numpy.ndarray
        Boolean array of length len(fsall), where True marks the first file from
        each folder.
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
    """
    Initialize per-plane database dictionaries and create output directories.

    Creates a deep copy of `db0` for each plane (and each ROI for mesoscope recordings),
    setting up save paths, binary file paths, and fast-disk directories.

    Parameters
    ----------
    db0 : dict
        Base database dictionary. Must contain "nplanes", "nchannels",
        "keep_movie_raw", "save_path0", and "save_folder". Optionally contains
        "fast_disk", "iplane", "lines", "dy", and "dx" (for mesoscope recordings
        with multiple ROIs).

    Returns
    -------
    dbs : list of dict
        List of per-plane database dictionaries, each with added keys "save_path",
        "fast_disk", "settings_path", "db_path", "reg_file", and optionally
        "raw_file", "reg_file_chan2", "raw_file_chan2", "lines", "dy", "dx",
        "iroi", "iplane".
    """
    nplanes = db0["nplanes"]
    nchannels = db0["nchannels"]
    keep_movie_raw = db0["keep_movie_raw"]
    nfolders = nplanes
    iplane = db0.get("iplane", np.arange(0, nplanes))
    has_lines = False
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
        has_lines = True 

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
        if has_lines:
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


def get_suite2p_path(path: Path) -> Path:
    """
    Find the root `suite2p` folder within a given path.

    Walks the path components backwards to locate the last occurrence of a folder
    named "suite2p" and returns the path up to and including that folder.

    Parameters
    ----------
    path : str or pathlib.Path
        File or directory path that should contain a "suite2p" folder component.

    Returns
    -------
    new_path : pathlib.Path
        Path truncated at the "suite2p" folder.

    Raises
    ------
    FileNotFoundError
        If "suite2p" is not found in any component of `path`.
    """

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
