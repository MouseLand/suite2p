from base64 import b64encode
import caiman as cm
from IPython.display import HTML, clear_output
import imageio
import logging
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyximport
pyximport.install()
import scipy
from tensorflow.python.client import device_lib
from time import time
    
from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator, visualize
from natsort import natsorted
from pathlib import Path 
import tifffile
from caiman.motion_correction import MotionCorrect
import shutil 
import time
import os
import argparse


logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.4.1

def timing(root, tfr, n_processes):
    nper_tif = 500
    ntiffs = tfr // nper_tif 

    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=n_processes, single_thread=False)

    iroi = 0
    fname_green = Path(os.path.join(root, f"tiffs_roi{iroi}/chan1/")).glob("*.tif")
    fname_green = [str(f) for f in fname_green]
    fname_green = natsorted(fname_green)
    fname_green = [f for f in fname_green if "init" not in f]
    fname_green = fname_green[:ntiffs]
    print(len(fname_green))

    fr = 6.76                         # sample rate of the movie

    mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
    ms = [65, 65]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
    center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
                    
    options = {
        'fnames': fname_green,
        'fr': fr,
        'mode': mode, 
        'ms': ms,
        'center_dims':center_dims, 
        }


    # copy first tiff to use as initialization
    fnames_init = fname_green[0].split('.')[0] + '_init.tif'
    shutil.copyfile(fname_green[0], fnames_init)

    tic = time.time()

    # run caiman to get initial template (will use same settings)
    strides =  (96, 96)  # default patch size
    overlaps = (32, 32)  # default overlap between patches (size of patch strides+overlaps)
    pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
    max_shifts = (65, 65)  # maximum allowed rigid shift in pixels - set to same size as suite2p
    max_deviation_rigid = 10 # increase to same size as in suite2p
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    min_mov = -4000
    mc = MotionCorrect(fnames_init, dview=dview, max_shifts=max_shifts,
                    strides=strides, overlaps=overlaps,
                    max_deviation_rigid=max_deviation_rigid, 
                    pw_rigid=pw_rigid,
                    border_nan=border_nan,
                    min_mov = min_mov)
    mc.motion_correct(save_movie=False)

    # template is total_template_rig 
    # (https://github.com/nel-lab/FIOLA/blob/ec7cb37c85831d14fb6d2402ca1aa5316e8ec886/fiola/demo_initialize_calcium.py#L218)
    template = mc.total_template_rig

    # open memmap to write fiola output
    out_mmap = os.path.join(root, f"tiffs_roi{iroi}/chan1/fiola_reg_test_{tfr}.mmap")
    n_frames, Ly, Lx = tfr, 1112, 650
    f_reg = np.memmap(out_mmap, mode="w+", dtype="float32", shape=(n_frames, Ly, Lx))

    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    # run motion correction on GPU on each tiff 
    # (trying to run all tiffs together gives GPU memory error)
    t = 0
    batch_size = 250
    for i, fname in enumerate(fname_green):
        mov = cm.load(fname, fr=fr)
        # (fiola will error if the movie is not a multiple of the batch size)
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, 
                                                                    batch_size=batch_size, 
                                                                    min_mov=min_mov)             
        print(mc_nn_mov.shape, f_reg.shape)
        f_reg[t : t+len(mc_nn_mov)] = mc_nn_mov
        t += len(mc_nn_mov)
        print(i, t)

    print(time.time() - tic)
    os.makedirs(os.path.join(root, "timings/"), exist_ok=True)
    np.save(os.path.join(root, f"timings/fiola_rigid_{tfr}.npy"), np.array([time.time()-tic]))

    os.remove(out_mmap)

def reg_green(root, iroi, n_processes=12):
    # start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=n_processes, single_thread=False)

    fname_green = Path(os.path.join(root, f"tiffs_roi{iroi}/chan1/")).glob("*.tif")
    fname_green = [str(f) for f in fname_green]
    fname_green = natsorted(fname_green)
    fname_green = [f for f in fname_green if "init" not in f]
    print(len(fname_green))

    fr = 6.76                         # sample rate of the movie

    mode = 'calcium'                # 'voltage' or 'calcium 'fluorescence indicator
    ms = [65, 65]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
    center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
                 
    options = {
        'fnames': fname_green,
        'fr': fr,
        'mode': mode, 
        'ms': ms,
        'center_dims':center_dims, 
        }
    
    # copy first tiff to use as initialization
    fnames_init = fname_green[0].split('.')[0] + '_init.tif'
    shutil.copyfile(fname_green[0], fnames_init)
    
    # run caiman to get initial template (will use same settings)
    strides =  (96, 96)  # default patch size
    overlaps = (32, 32)  # default overlap between patches (size of patch strides+overlaps)
    pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
    max_shifts = (65, 65)  # maximum allowed rigid shift in pixels - set to same size as suite2p
    max_deviation_rigid = 10 # increase to same size as in suite2p
    border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
    min_mov = -4000
    mc = MotionCorrect(fnames_init, dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid, 
                  pw_rigid=pw_rigid,
                  border_nan=border_nan,
                  min_mov = min_mov)
    mc.motion_correct(save_movie=False)

    # template is total_template_rig 
    # (https://github.com/nel-lab/FIOLA/blob/ec7cb37c85831d14fb6d2402ca1aa5316e8ec886/fiola/demo_initialize_calcium.py#L218)
    template = mc.total_template_rig

    # open memmap to write fiola output
    out_mmap = os.path.join(root, f"tiffs_roi{iroi}/chan1/fiola_reg.mmap")
    n_frames, Ly, Lx = 32187, 1112, 650
    f_reg = np.memmap(out_mmap, mode="w+", dtype="float32", shape=(n_frames, Ly, Lx))

    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    # run motion correction on GPU on each tiff 
    # (trying to run all tiffs together gives GPU memory error)
    t = 0
    for i, fname in enumerate(fname_green):
        mov = cm.load(fname, fr=fr)
        print(mov.shape[0])
        batch_size = mov.shape[0] // 2 if mov.shape[0] > 250 else mov.shape[0]
        # (fiola will error if the movie is not a multiple of the batch size)
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, 
                                                                    batch_size=batch_size, 
                                                                    min_mov=min_mov)             
        f_reg[t : t+len(mc_nn_mov)] = mc_nn_mov
        if i==0:
            shifts_all = shifts_fiola.copy()
        else:
            shifts_all = np.concatenate((shifts_all, shifts_fiola), axis=0)
        t += len(mc_nn_mov)
        print(i, t)
        
    np.save(os.path.join(root, f"tiffs_roi{iroi}/chan1/shifts_fiola.npy"), shifts_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/media/carsen/disk2/suite2p_paper/GT1/")
    parser.add_argument("--roi", type=int, default=0)
    parser.add_argument("--tfr", type=int, default=500)
    parser.add_argument("--n_processes", type=int, default=12)
    parser.add_argument("--timing", action="store_true")

    
    args = parser.parse_args()

    tfr = args.tfr
    root = args.root
    n_processes = args.n_processes
    i = args.roi 

    if args.timing:
        timing(root, tfr, n_processes)
    else:
        reg_green(root, i, n_processes)
