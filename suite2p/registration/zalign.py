"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time

import numpy as np
from scipy.signal import medfilt
import torch

import logging 
logger = logging.getLogger(__name__)

from . import nonrigid, rigid, utils
from .. import default_settings
from .register import register_frames

def register_to_zstack(f_align_in, refImgs, nonrigid=False, settings=default_settings()["registration"], 
                       bidiphase=0, device=torch.device("cuda")):
    n_frames, Ly, Lx = f_align_in.shape

    ### ----- register frames to reference image -------------- ###
    outputs = register_frames(f_align_in, f_align_out=None, bidiphase=bidiphase,
                            refImg=refImgs, tif_root=None, 
                            batch_size=settings["batch_size"], 
                            norm_frames=settings["norm_frames"], smooth_sigma=settings["smooth_sigma"], 
                            spatial_taper=settings["spatial_taper"], block_size=settings["block_size"], 
                            nonrigid=nonrigid,
                            maxregshift=settings["maxregshift"], smooth_sigma_time=settings["smooth_sigma_time"],
                                snr_thresh=settings["snr_thresh"], maxregshiftNR=settings["maxregshiftNR"],
                                device=device, apply_shifts=False)
    rmin, rmax, mean_img, offsets_all, blocks = outputs
    yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all = offsets_all

    return cmax_all

def compute_zpos():
    return None