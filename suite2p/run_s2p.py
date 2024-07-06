"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import shutil
import time
from natsort import natsorted
from datetime import datetime
from getpass import getpass
import pathlib
import contextlib
import numpy as np
from scipy.stats import skew
import torch

from . import io, default_ops, default_db, pipeline

from functools import partial
from pathlib import Path

print = partial(print, flush=True)

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
    "raw":
        io.raw_to_binary,
    "bruker":
        io.ome_to_binary,
    "movie":
        io.movie_to_binary,
    "dcimg":
        io.dcimg_to_binary,
}


def _assign_torch_device(str_device):
    """
    Checks if CUDA is available and working with PyTorch.

    Args:
        gpu_number (int): The GPU device number to use (default is 0).

    Returns:
        bool: True if CUDA is available and working, False otherwise.
    """
    if str_device == "cpu":
        print("** using CPU **")
        return torch.device(str_device)
    else:
        try:
            device = torch.device(str_device)
            _ = torch.zeros([1, 2, 3]).to(device)
            print(f"** torch.device('{str_device}') installed and working. **")
            return torch.device(str_device)
        except:
            print(f"torch.device('{str_device}') not working, using cpu.")
            return torch.device("cpu")

def _check_run_registration(ops, db):
    if ops["run"]["do_registration"] > 0:
        reg_outputs_path = os.path.join(db["save_path"], "reg_outputs.npy")
        reg_outputs = (np.load(reg_outputs_path, allow_pickle=True).item() 
                       if os.path.exists(reg_outputs_path) else {})
        if "yoff" in reg_outputs and ops["run"]["do_registration"] > 1:
            print("Forced re-run of registration with ops['run']['do_registration']>1")
            print("(NOTE: final offsets will be relative to previous registration if keep_movie_raw is False)")
            # delete reg_outputs.npy
            os.remove(os.path.join(db["save_path"], "reg_outputs.npy"))
            run_registration = True
        elif "yoff" not in reg_outputs and ops["run"]["do_registration"] > 0:
            print("Running registration")
            run_registration = True 
        else:
            print("NOTE: not running registration, plane already registered")
            print("binary path: %s" % db["reg_file"])
            run_registration = False
    else:
        print("NOTE: not running registration, ops['run']['do_registration']=0")
        print("binary path: %s" % db["reg_file"])
        run_registration = False
    return run_registration

def _find_existing_binaries(plane_folders):
    db_paths = [os.path.join(f, "db.npy") for f in plane_folders]
    ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    db_found_flag = all([os.path.isfile(db_path) for db_path in db_paths])
    ops_found_flag = all([os.path.isfile(ops_path) for ops_path in ops_paths])
    binaries_found_flag = all([
        os.path.isfile(os.path.join(f, "data_raw.bin")) or
        os.path.isfile(os.path.join(f, "data.bin")) for f in plane_folders
    ])
    files_found_flag = db_found_flag and ops_found_flag and binaries_found_flag
    return files_found_flag, db_paths, ops_paths

def run_plane(db, ops, db_path=None, stat=None):
    """ run suite2p processing on a single binary file

    Parameters
    -----------
    ops : :obj:`dict` 
        specify "reg_file", "nchannels", "tau", "fs"

    ops_path: str
        absolute path to ops file (use if files were moved)

    stat: list of `dict`
        ROIs

    Returns
    --------
    ops : :obj:`dict` 
    """

    ops = {**default_ops(), **ops}
    ops["date_proc"] = datetime.now().astimezone()
    
    # torch device
    device = _assign_torch_device(ops["torch_device"])

    # for running on server or on moved files, specify db_path and paths are renamed
    if (db_path is not None and os.path.exists(db_path) and not
        (os.path.exists(db["reg_file"]) or os.path.exists(db["raw_file"]))):
        db["save_path"] = os.path.split(db_path)[0]
        db["db_path"] = db_path
        db["ops_path"] = os.path.join(db["save_path"], "ops.npy")
        if (os.path.exists(os.path.join(db["save_path"], "data.bin")) or 
            os.path.exists(os.path.join(db["save_path"], "data_raw.bin"))):
            for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
                if key in db:
                    db[key] = os.path.join(db["save_path"], os.path.split(db[key])[1])
        else:
            raise FileNotFoundError("binary file data.bin or data_raw.bin not found in db_path")
        # re-save db and ops in new path
        np.save(db["db_path"], db)
        np.save(db["ops_path"], ops)

    # check that there are sufficient numbers of frames
    if db["nframes"] < 50: raise ValueError("number of frames should be at least 50")
    elif db["nframes"] < 200:
        print("WARNING: number of frames < 200, unpredictable behaviors may occur")

    # check if registration should be done
    run_registration = _check_run_registration(ops, db)
    
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
    badframes_path = badframes_path if os.path.exists(badframes_path) else None
    # badframes from file (optional)
    badframes0 = np.zeros(db["nframes"], "bool")
    if badframes_path is not None and os.path.exists(badframes_path):
        bf_indices = np.load(badframes_path).flatten().astype("int")
        badframes0[bf_indices] = True
        print(f"badframes file: {badframes_path};\n # of badframes: {badframes0.sum()}")

    null = contextlib.nullcontext()
    with io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file, n_frames=n_frames, write=False) \
            if raw else null as f_raw, \
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames, write=True) as f_reg, \
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2, n_frames=n_frames, write=False) \
            if raw and twoc else null as f_raw_chan2,\
         io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2, n_frames=n_frames, write=True) \
            if twoc else null as f_reg_chan2:

        outputs = pipeline(db["save_path"], f_reg, f_raw, f_reg_chan2, f_raw_chan2, 
                   run_registration, ops, badframes=badframes0, stat=stat,
                   device=device)

    # save as matlab file
    if ops["io"]["save_mat"]:
        io.save_mat(db["save_path"], *outputs)

    if ops["io"]["move_bin"] and db["save_path"] != db["fast_disk"]:
        print("moving binary files to save_path")
        for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
            if key in db:
                shutil.move(db[key],
                        os.path.join(db["save_path"], os.path.split(db[key])[1]))
                
    elif ops["io"]["delete_bin"]:
        print("deleting binary files")
        for key in ["reg_file", "reg_file_chan2", "raw_file", "raw_file_chan2"]:
            if key in db:
                os.remove(db[key])

    return outputs


def run_s2p(db={}, ops=default_ops(), server={}):
    """Run suite2p pipeline.

    Args:
        db (dict): Specify "data_path", "nplanes", "nchannels", etc. for making binaries.
        ops (dict): Specify settings for running, e.g. "fs" sampling, "tau" timescale etc.
        server (dict): Specify "host", "username", "password", "server_root", "local_root", "n_cores" (for multiplane_parallel).

    Returns:
        list: Paths to db files.
    """

    t0 = time.time()
            
    ops = {**default_ops(), **ops}
    db = {**default_db(), **db}
    print(db)
    if db["save_path0"] is None or len(db["save_path0"])==0:
        db["save_path0"] = db["data_path"][0]

    # check if there are binaries already made
    if db["save_folder"] is None or len(db["save_folder"]) == 0:
        db["save_folder"] = "suite2p"
    save_folder = os.path.join(db["save_path0"], db["save_folder"])
    if os.path.exists(save_folder):
        plane_folders = natsorted([
            f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
        ])
    else:
        plane_folders = []
    
    if len(plane_folders) > 0 and (ops.get("input_format") and ops["input_format"]=="binary"):
        # TODO: fix this
        ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
    elif len(plane_folders) > 0:
        files_found_flag, db_paths, ops_paths = _find_existing_binaries(plane_folders)
    else:
        files_found_flag = False

    if files_found_flag:
        print(f"FOUND BINARIES AND DBS AND OPS IN {db_paths}")
        print("removing previous detection and extraction files, if present")
        files_to_remove = [
            "detect_outputs.npy", "stat.npy", 
            "F.npy", "Fneu.npy", "F_chan2.npy", "Fneu_chan2.npy", 
            "spks.npy", "iscell.npy", "redcell.npy"
        ]
        for f in plane_folders:
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
        ops_paths = [db["ops_path"] for db in dbs]
        save_folder = os.path.join(db["save_path0"], db["save_folder"])
        np.save(os.path.join(save_folder, "db.npy"), db)
        np.save(os.path.join(save_folder, "ops.npy"), ops)
        
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
            
            dbs = files_to_binary[db["input_format"]](dbs, ops, files, files_chan2)
        
        print("Wrote {} frames per binary, {} folders + {} channels, {:0.2f}sec".format(
                dbs[0]["nframes"], len(dbs), dbs[0]["nchannels"], time.time() - t0))
        
    if ops["run"]["multiplane_parallel"]:
        if server:  # if user puts in server settings
            io.server.send_jobs(save_folder, **server)
        else: # otherwise use settings modified in io/server.py
            io.server.send_jobs(save_folder)
        return None
    else:
        for ipl, (ops_path, db_path) in enumerate(zip(ops_paths, db_paths)):
            if db.get("ignore_flyback", None) is not None:
                if ipl in db["ignore_flyback"]:
                    print(">>>> skipping flyback PLANE", ipl)
                    continue
            op = np.load(ops_path, allow_pickle=True).item()
            op = {**op, **ops}
            db = np.load(db_path, allow_pickle=True).item()
            print(f">>>>>>>>>>>>>>>>>>>>> PLANE {ipl} <<<<<<<<<<<<<<<<<<<<<<")
            outputs = run_plane(db=db, ops=ops, db_path=db_path)
        run_time = time.time() - t0
        print("total = %0.2f sec." % run_time)

        #### COMBINE PLANES or FIELDS OF VIEW ####
        if len(ops_paths) > 1 and ops["io"]["combined"] and ops["run"]["do_detection"]:
            print("Creating combined view")
            io.combined(save_folder, save=True)

        # save to NWB
        if ops["io"]["save_NWB"]:
            print("Saving in nwb format")
            io.save_nwb(save_folder)

        print("TOTAL RUNTIME %0.2f sec" % (time.time() - t0))
        return db_paths
