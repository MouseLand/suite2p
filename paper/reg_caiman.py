import cv2
from IPython import get_ipython
import matplotlib.pyplot as plt
import numpy as np
import os.path
import logging
import argparse 
import os

try:
    cv2.setNumThreads(0)
except:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect, tile_and_correct, motion_correction_piecewise
from caiman.utils.utils import download_demo

from pathlib import Path
from natsort import natsorted
import time

logfile = None # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
logger.setLevel(logging.INFO)
logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
if logfile is not None:
    handler = logging.FileHandler(logfile)
else:
    handler = logging.StreamHandler()
handler.setFormatter(logfmt)
logger.addHandler(handler)

def timing(root, tfr, rigid=False, n_processes=64):
    os.makedirs(f'~/caiman_data/temp_{n_processes}_{tfr}_{rigid}/', exist_ok=True)
    os.environ['CAIMAN_TEMP'] = f'~/caiman_data/temp_{n_processes}_{tfr}_{rigid}/'
    nper_tif = 500
    ntiffs = tfr // nper_tif 

    
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=n_processes, single_thread=False)

    fname_green = Path(os.path.join(root, f"tiffs_roi0/chan1/")).glob("*.tif")
    fname_green = [str(f) for f in fname_green]
    fname_green = natsorted(fname_green)
    fname_green = [f for f in fname_green if "init" not in f]
    fname_green = fname_green[:ntiffs]
    print(ntiffs, rigid)

    strides =  (96, 96)  # default patch size
    overlaps = (32, 32)  # default overlap between patches (size of patch strides+overlaps)
    pw_rigid = (not rigid)  # flag for performing rigid or piecewise rigid motion correction
    max_shifts = (65, 65)  # maximum allowed rigid shift in pixels - set to same size as suite2p
    max_deviation_rigid = 10 # increase to same size as in suite2p
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    min_mov = -4000 # minimum pixel value in tiff movies
    
    tic = time.time()
    mc = MotionCorrect(fname_green, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  pw_rigid=pw_rigid,
                  border_nan=border_nan,
                  min_mov=min_mov)
    mc.motion_correct(save_movie=True)
    print(time.time() - tic)
    os.makedirs(os.path.join(root, "timings/"), exist_ok=True)
    np.save(os.path.join(root, f"timings/caiman_{n_processes}_{['', 'rigid_'][rigid]}{tfr}.npy"), time.time()-tic)

def reg_green(root, i, n_processes=64, rigid=False):
    os.makedirs(f'~/caiman_data/temp{i}/', exist_ok=True)
    os.environ['CAIMAN_TEMP'] = f'~/caiman_data/temp{i}/'
    
    alg_str = 'caiman'
    alg_str += '_rigid' if rigid else ''

    fname_green = Path(os.path.join(root, f"tiffs_roi{i}/chan1/")).glob("*.tif")
    fname_green = [str(f) for f in fname_green]
    fname_green = natsorted(fname_green)
    fname_green = [f for f in fname_green if "init" not in f]
    print(fname_green)

    strides =  (96, 96)  # default patch size
    overlaps = (32, 32)  # default overlap between patches (size of patch strides+overlaps)
    pw_rigid = (not rigid)  # flag for performing rigid or piecewise rigid motion correction
    max_shifts = (65, 65)  # maximum allowed rigid shift in pixels - set to same size as suite2p
    max_deviation_rigid = 10 # increase to same size as in suite2p
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    min_mov = -4000 # minimum pixel value in tiff movies
    
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=n_processes, single_thread=False)

    mc = MotionCorrect(fname_green, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  pw_rigid=pw_rigid,
                  border_nan=border_nan,
                  min_mov=min_mov)
    mc.motion_correct(save_movie=True)

    out_mmap = os.path.join(root, f"tiffs_roi{i}/chan1/{alg_str}_reg.mmap")
    n_frames = 32187
    Ly, Lx = 1112, 650
    f_reg = np.memmap(out_mmap, mode="w+", dtype="float32", shape=(n_frames, Ly, Lx))
    t = 0
    print(f"roi{i} motion shifts found")
    for m in mc.mmap_file:
        nfr = int(os.path.splitext(m)[0].split("_")[-1])
        fin = np.memmap(m, mode="r", shape=(Ly * Lx, nfr), dtype="float32", 
                            order="F").reshape(Lx, Ly, nfr).transpose(2, 1, 0)
        f_reg[t : t+nfr] = fin[:nfr]
        t += nfr
        print(t, m)

    print(f"roi{i} green channel written")

    fname_red = Path(os.path.join(root, f"tiffs_roi{i}/chan2/")).glob("*.tif")
    fname_red = [str(f) for f in fname_red]
    fname_red = natsorted(fname_red)
    save_base_name = os.path.join(root, f"tiffs_roi{i}/chan2/{alg_str}")
    mmap_file = mc.apply_shifts_movie(fname_red, save_base_name=save_base_name, 
                                    save_memmap=True, order='C')
    print(f"roi{i} motion shifts applied")
    print(mmap_file)
    
    if rigid:
        np.save(os.path.join(root, f"tiffs_roi{i}/chan1/shifts_{alg_str}.npy"), 
                {"shifts_rig": np.array(mc.shifts_rig)})
    else:
        np.save(os.path.join(root, f"tiffs_roi{i}/chan1/shifts_{alg_str}.npy"), 
                {"shifts_rig": np.array(mc.shifts_rig), "y_shifts_els": np.array(mc.y_shifts_els), 
                "x_shifts_els": np.array(mc.x_shifts_els)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/media/carsen/disk2/suite2p_paper/GT1/")
    parser.add_argument("--roi", type=int, default=0)
    parser.add_argument("--tfr", type=int, default=500)
    parser.add_argument("--n_processes", type=int, default=64)
    parser.add_argument("--rigid", action="store_true")
    parser.add_argument("--timing", action="store_true")

    
    args = parser.parse_args()

    tfr = args.tfr
    rigid = args.rigid
    root = args.root
    n_processes = args.n_processes
    i = args.roi 

    if args.timing:
        timing(root, tfr, rigid=rigid, n_processes=n_processes)
    else:
        reg_green(root, i, rigid=rigid, n_processes=n_processes)