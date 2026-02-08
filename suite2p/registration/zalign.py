"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import torch

import logging 
logger = logging.getLogger(__name__)

from .. import default_settings
from .register import register_frames

def register_to_zstack(f_align_in, refImgs, nonrigid=False, settings=default_settings()["registration"],
                       bidiphase=0, device=torch.device("cuda")):
    """
    Register frames to a z-stack of reference images and return the max correlation per z-plane.

    Runs `register_frames` with `apply_shifts=False` to compute phase-correlation between
    each frame and every reference image in `refImgs`, without actually shifting the data.

    Parameters
    ----------
    f_align_in : torch.Tensor or numpy.ndarray
        Input frames of shape (n_frames, Ly, Lx).
    refImgs : torch.Tensor or numpy.ndarray
        Reference images from the z-stack, passed directly to `register_frames` as `refImg`.
    nonrigid : bool, optional (default False)
        Whether to use nonrigid registration in addition to rigid registration.
    settings : dict, optional
        Registration settings dictionary (from `default_settings()["registration"]`).
        Controls batch_size, norm_frames, smooth_sigma, spatial_taper, block_size,
        maxregshift, smooth_sigma_time, snr_thresh, and maxregshiftNR.
    bidiphase : int, optional (default 0)
        Bidirectional phase offset to correct for bidirectional scanning artifacts.
    device : torch.device, optional (default torch.device("cuda"))
        Device on which to run the registration.

    Returns
    -------
    cmax_all : numpy.ndarray
        Maximum correlation values for each frame across z-planes.
    """
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
    """
    Compute z-position estimates from registered frames.

    Returns
    -------
    None
        Not yet implemented.
    """
    return None
