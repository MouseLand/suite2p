"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os, sys
import shutil
import time
from natsort import natsorted
from datetime import datetime
from getpass import getpass
import pathlib
import contextlib
import numpy as np
from scipy.stats import skew
from scipy.io import loadmat
import torch
import logging 

logger = logging.getLogger(__name__)

from . import io, default_settings, default_db, pipeline, version_str

from functools import partial
from pathlib import Path

# copy file format to a binary file
files_to_binary = {
    "tif":
        io.tiff_to_binary,
    "h5":
        io.h5py_to_binary,
    "nwb":
        io.nwb_to_binary,
    "sbx":
        io.sbx_to_binary,
    "nd2":
        io.nd2_to_binary,
    "bruker":
        io.ome_to_binary,
    "movie":
        io.movie_to_binary,
    "dcimg":
        io.dcimg_to_binary,
}

def get_save_folder(db):
    """
    Get the save folder path from the database dictionary.

    Parameters
    ----------
    db : dict
        Database dictionary containing "save_path0", "save_folder", and
        "data_path".

    Returns
    -------
    save_folder : str
        Full path to the suite2p save folder.
    """
    if db["save_path0"] is None or len(db["save_path0"])==0:
        db["save_path0"] = db["data_path"][0]

    if db["save_folder"] is None or len(db["save_folder"]) == 0:
        db["save_folder"] = "suite2p"
    save_folder = os.path.join(db["save_path0"], db["save_folder"])
    return save_folder

def logger_setup(save_path=None):
    """
    Configure logging for the suite2p package.

    Sets up console and file logging handlers for the suite2p logger.

    Parameters
    ----------
    save_path : str, optional (default None)
        Directory to write the log file. If None, only console logging is
        configured.
    """
    if save_path is not None and not pathlib.Path(save_path).exists():
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    s2p_logger = logging.getLogger('suite2p')
    s2p_logger.setLevel(logging.DEBUG)
    
    # Skip this if the handlers were already added, like when running multiple
    # times in a single session.
    if not s2p_logger.handlers:
        # Add console handler at info level with shorter messages,
        # unless verbose is requested.
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        console.setFormatter(file_formatter)
        s2p_logger.addHandler(console)


    if save_path is not None:
        log_file = pathlib.Path(save_path) / "run.log"
        try:
            log_file.unlink()
        except:
            pass
        print(f"creating log file {log_file}")
        file = logging.FileHandler(log_file, mode='w')
        file.setLevel(logging.DEBUG)
        log_file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file.setFormatter(log_file_formatter)
        s2p_logger.addHandler(file)


def _assign_torch_device(str_device):
    """
    Validate and return a torch device, falling back to CPU if unavailable.

    Parameters
    ----------
    str_device : str
        Device string, e.g. "cuda", "cpu", or "mps".

    Returns
    -------
    device : torch.device
        Validated torch device. Falls back to CPU if the requested device
        is not available.
    """
    if str_device == "cpu":
        logger.info("** using CPU **")
        return torch.device(str_device)
    else:
        try:
            device = torch.device(str_device)
            _ = torch.zeros([1, 2, 3]).to(device)
            logger.info(f"** torch.device('{str_device}') installed and working. **")
            return torch.device(str_device)
        except:
            logger.info(f"torch.device('{str_device}') not working, using cpu.")
            return torch.device("cpu")

def _check_run_registration(settings, db):
    if settings["run"]["do_registration"] > 0:
        reg_outputs_path = os.path.join(db["save_path"], "reg_outputs.npy")
        reg_outputs = (np.load(reg_outputs_path, allow_pickle=True).item() 
                       if os.path.exists(reg_outputs_path) else {})
        if "yoff" in reg_outputs and settings["run"]["do_registration"] > 1:
            logger.info("Forced re-run of registration with settings['run']['do_registration']>1")
            logger.info("(NOTE: final offsets will be relative to previous registration if keep_movie_raw is False)")
            # delete reg_outputs.npy
            os.remove(os.path.join(db["save_path"], "reg_outputs.npy"))
            run_registration = True
        elif "yoff" not in reg_outputs and settings["run"]["do_registration"] > 0:
            logger.info("Running registration")
            run_registration = True 
        else:
            logger.info("NOTE: not running registration, plane already registered")
            logger.info("binary path: %s" % db["reg_file"])
            run_registration = False
    else:
        logger.info("NOTE: not running registration, settings['run']['do_registration']=0")
        logger.info("binary path: %s" % db["reg_file"])
        run_registration = False
    return run_registration

def _find_existing_binaries(plane_folders):
    db_paths = [os.path.join(f, "db.npy") for f in plane_folders]
    settings_paths = [os.path.join(f, "settings.npy") for f in plane_folders]
    db_found_flag = all([os.path.isfile(db_path) for db_path in db_paths])
    settings_found_flag = all([os.path.isfile(settings_path) for settings_path in settings_paths])
    binaries_found_flag = all([
        os.path.isfile(os.path.join(f, "data_raw.bin")) or
        os.path.isfile(os.path.join(f, "data.bin")) for f in plane_folders
    ])
    files_found_flag = db_found_flag and settings_found_flag and binaries_found_flag
    return files_found_flag, db_paths, settings_paths

def run_plane(db, settings, db_path=None, stat=None):
    """
    Run suite2p processing on a single plane/ROI.

    Opens binary files, runs the pipeline, and saves outputs including
    optional .mat and ops.npy files.

    Parameters
    ----------
    db : dict
        Database dictionary for this plane, containing "reg_file",
        "save_path", "nframes", "Ly", "Lx", and other plane-specific keys.
    settings : dict
        Pipeline settings dictionary.
    db_path : str, optional (default None)
        Absolute path to db.npy file. If provided and binary files have
        been moved, paths are updated accordingly.
    stat : numpy.ndarray, optional (default None)
        Pre-defined ROI masks. If provided, detection is skipped.

    Returns
    -------
    outputs : tuple
        Pipeline outputs from pipeline(), see pipeline_s2p.pipeline for
        details.
    """

    settings = {**default_settings(), **settings}
    settings["date_proc"] = datetime.now().astimezone()
    
    # torch device
    device = _assign_torch_device(settings["torch_device"])

    # for running on server or on moved files, specify db_path and paths are renamed
    if (db_path is not None and os.path.exists(db_path) and not
        (os.path.exists(db["reg_file"]) or (db.get("raw_file", None) is not None and 
                                            os.path.exists(db.get("raw_file", None))))):
        db["save_path"] = os.path.split(db_path)[0]
        db["db_path"] = db_path
        db["settings_path"] = os.path.join(db["save_path"], "settings.npy")
        if (os.path.exists(os.path.join(db["save_path"], "data.bin")) or 
            os.path.exists(os.path.join(db["save_path"], "data_raw.bin"))):
            for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
                if key in db:
                    db[key] = os.path.join(db["save_path"], os.path.split(db[key])[-1])
        else:
            raise FileNotFoundError("binary file data.bin or data_raw.bin not found in db_path")
        # re-save db and settings in new path
        db["save_path0"] = os.path.split(os.path.split(db["save_path"])[0])[0]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    # check that there are sufficient numbers of frames
    if db["nframes"] < 10: raise ValueError("number of frames should be at least 50")
    elif db["nframes"] < 200:
        logger.warn("WARNING: number of frames < 200, unpredictable behaviors may occur")

    # check if registration should be done
    run_registration = _check_run_registration(settings, db)
    
    # get binary file paths
    reg_file = db["reg_file"]
    raw_file = db.get("raw_file", None)
    raw = db["keep_movie_raw"] and os.path.isfile(raw_file)
    twoc = db["nchannels"] > 1
    reg_file_chan2 = db["reg_file_chan2"] if twoc else None
    raw_file_chan2 = db.get("raw_file_chan2", None) if twoc else None

    # shape of binary files
    n_frames, Ly, Lx = db["nframes"], db["Ly"], db["Lx"]
    
    # get frames to exclude from registration and detection (e.g. photostim frames)
    badframes_path = os.path.join(db["data_path"][0], "bad_frames.npy")
    if not os.path.exists(badframes_path):
        badframes_path = os.path.join(db["save_path0"], "bad_frames.npy")
    badframes_path = badframes_path if os.path.exists(badframes_path) else None
    # badframes from file (optional)
    badframes0 = np.zeros(db["nframes"], "bool")
    if badframes_path is not None and os.path.exists(badframes_path):
        bf_indices = np.load(badframes_path).flatten().astype("int")
        badframes0[bf_indices] = True
        logger.info(f"badframes file: {badframes_path};\n # of badframes: {badframes0.sum()}")

    # check for zstack file to align to
    Zstack = None
    if os.path.exists(os.path.join(db["save_path"], "zcorr.npy")):
        logger.info("z-correlation already computed")
    else:
        zstack_path = os.path.join(db["data_path"][0], "zstack.npy")
        if not os.path.exists(zstack_path):
            zstack_path = os.path.join(db["save_path0"], 'zstack.mat')
        zstack_path = zstack_path if os.path.exists(zstack_path) else None
        if zstack_path is not None:
            logger.info(f"zstack file: {zstack_path}")
            data = loadmat(zstack_path)
            iplane = db.get("iplane", 0)
            iroi = db.get("iroi", 0)
            if len(data['Z']) == 0:
                Zstack = None
                logger.info("zstack file is empty")
            elif iroi > 0:
                if iroi < len(data['Z']):
                    Zstack = data['Z'][iroi][0].squeeze()
                else:
                    logger.info(f"plane {iplane} roi {iroi} not in zstack file")
            else:
                if iplane < len(data['Z'][0]):
                    Zstack = data['Z'][0][iplane].squeeze()
                else:
                    logger.info(f"plane {iplane} not in zstack file")
        if Zstack is not None:
            if Zstack.ndim > 3:
                Zstack = Zstack[0]
            Zstack = Zstack.transpose(2, 1, 0)
            logger.info(f"zstack shape: {Zstack.shape}")

    logger.info(f"binary output path: {reg_file}")
    if raw_file is not None:
        logger.info(f"raw binary path: {raw_file}")
    null = contextlib.nullcontext()
    with io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file, n_frames=n_frames, write=False) \
            if raw else null as f_raw, \
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames, write=True) as f_reg, \
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2, n_frames=n_frames, write=False) \
            if raw and twoc else null as f_raw_chan2,\
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2, n_frames=n_frames, write=True) \
            if twoc else null as f_reg_chan2:

        outputs = pipeline(db["save_path"], f_reg, f_raw, f_reg_chan2, f_raw_chan2, 
                   run_registration, settings, badframes=badframes0, stat=stat,
                   device=device, Zstack=Zstack)
        (reg_outputs, detect_outputs, stat, F, Fneu, F_chan2, Fneu_chan2, spks, iscell, redcell, zcorr, plane_times) = outputs

    # save as matlab file
    if settings["io"]["save_mat"]:
        if reg_outputs is None:
            logger.info("No registration outputs to save (registration was skipped)")
        if detect_outputs is None:
            logger.info("No detection outputs to save (detection was skipped)")
        ops = {**db, **settings, **(reg_outputs or {}), **(detect_outputs or {}), **plane_times}
        io.save_mat(ops, stat, F, Fneu, spks, iscell, redcell, F_chan2, Fneu_chan2)

    # save ops orig
    if settings["io"]["save_ops_orig"]:
        if reg_outputs is None:
            logger.info("No registration outputs in ops.npy (registration was skipped)")
        if detect_outputs is None:
            logger.info("No detection outputs in ops.npy (detection was skipped)")
        ops = {**db, **settings, **(reg_outputs or {}), **(detect_outputs or {})}
        ops["plane_times"] = plane_times
        np.save(os.path.join(db["save_path"], "ops.npy"), ops)
    
    if settings["io"]["move_bin"] and db["save_path"] != db["fast_disk"]:
        logger.info("moving binary files to save_path")
        for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
            if key in db:
                shutil.move(db[key],
                        os.path.join(db["save_path"], os.path.split(db[key])[1]))
                
    elif settings["io"]["delete_bin"]:
        logger.info("deleting binary files")
        for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
            if key in db:
                os.remove(db[key])

    return outputs


def run_s2p(db={}, settings=default_settings(), server={}):
    """
    Run the full suite2p pipeline across all planes.

    Converts input files to binary format (if needed), then runs
    registration, detection, extraction, deconvolution, and classification
    on each plane sequentially (or dispatches to a server for parallel
    processing).

    Parameters
    ----------
    db : dict
        Database dictionary specifying "data_path", "nplanes", "nchannels",
        and other input/output configuration.
    settings : dict
        Pipeline settings dictionary, e.g. "fs", "tau", "diameter".
    server : dict
        Server configuration for multiplane_parallel mode. Specify "host",
        "username", "password", "server_root", "local_root", "n_cores".

    Returns
    -------
    db_paths : list of str or None
        Paths to the per-plane db.npy files. None if running in
        multiplane_parallel server mode.
    """

    t0 = time.time()
            
    settings = {**default_settings(), **settings}
    db = {**default_db(), **db}

    save_folder = get_save_folder(db)
    
    if os.path.exists(save_folder):
        plane_folders = natsorted([
            f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
        ])
    else:
        plane_folders = []

    logger.info(version_str)
    logger.info(f"data_path: {db['data_path']}")
    
    if len(plane_folders) > 0:
        files_found_flag, db_paths, settings_paths = _find_existing_binaries(plane_folders)
    else:
        files_found_flag = False

    if files_found_flag:
        logger.info(f"FOUND BINARIES AND DBS IN {db_paths}")
        logger.info("removing previous detection and extraction files, if present")
        logger.info(f"will update settings.npy but not db.npy")
        files_to_remove = [
            "detect_outputs.npy", "stat.npy", 
            "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy", 
            "spks.npy", "iscell.npy", "redcell.npy"
        ]
        for f in plane_folders:
            np.save(os.path.join(f, "settings.npy"), settings)
            for fname in files_to_remove:
                if os.path.exists(os.path.join(f, fname)):
                    os.remove(os.path.join(f, fname))

    # if not set up files and copy tiffs/h5py to binary
    else:
        db["input_format"] = db.get("input_format", "tif")
        
        # find files
        fs, first_files = io.get_file_list(db)
        db["file_list"] = fs 
        db["first_files"] = first_files
        
        # copy dbs to list per plane + create folders
        dbs = io.init_dbs(db)
        db_paths = [db["db_path"] for db in dbs]
        settings_paths = [db["settings_path"] for db in dbs]
        save_folder = os.path.join(db["save_path0"], db["save_folder"])
        np.save(os.path.join(save_folder, "db.npy"), db)
        np.save(os.path.join(save_folder, "settings.npy"), settings)
        
        # open all binary files for writing
        with contextlib.ExitStack() as stack:
            raw_str = "raw" if db.get("keep_movie_raw", False) else "reg"
            fnames = [db[f"{raw_str}_file"] for db in dbs]
            files = [stack.enter_context(open(f, "wb")) for f in fnames]
            if db["nchannels"] > 1:
                fnames_chan2 = [db[f"{raw_str}_file_chan2"] for db in dbs]
                files_chan2 = [stack.enter_context(open(f, "wb")) for f in fnames_chan2]
            else:
                files_chan2 = None
            
            dbs = files_to_binary[db["input_format"]](dbs, settings, files, files_chan2)
        
        logger.info("Wrote {} frames per binary, {} folders + {} channels, {:0.2f}sec".format(
                dbs[0]["nframes"], len(dbs), dbs[0]["nchannels"], time.time() - t0))
        
    if settings["run"]["multiplane_parallel"]:
        if server:  # if user puts in server settings
            io.server.send_jobs(save_folder, **server)
        else: # otherwise use settings modified in io/server.py
            io.server.send_jobs(save_folder)
        return None
    else:
        for ipl, (settings_path, db_path) in enumerate(zip(settings_paths, db_paths)):
            if db.get("ignore_flyback", None) is not None:
                if ipl in db["ignore_flyback"]:
                    logger.info(">>>> skipping flyback PLANE", ipl)
                    continue
            op = np.load(settings_path, allow_pickle=True).item()
            op = {**op, **settings}
            db = np.load(db_path, allow_pickle=True).item()
            logger.info(f">>>>>>>>>>>>>>>>>>>>> PLANE {ipl} <<<<<<<<<<<<<<<<<<<<<<")
            outputs = run_plane(db=db, settings=settings, db_path=db_path)
        run_time = time.time() - t0
        logger.info("total = %0.2f sec." % run_time)

        #### COMBINE PLANES or FIELDS OF VIEW ####
        if len(settings_paths) > 1 and settings["io"]["combined"] and settings["run"]["do_detection"]:
            logger.info("Creating combined view")
            io.combined(save_folder, save=True)

        # save to NWB
        if settings["io"]["save_NWB"]:
            logger.info("Saving in nwb format")
            io.save_nwb(save_folder)

        logger.info("TOTAL RUNTIME %0.2f sec" % (time.time() - t0))
        return db_paths
