"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from multiprocessing import Pool

import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import logging 
logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from . import register
from .. import default_settings

import torch 

def pclowhigh(mov, nlowhigh, nPC, random_state):
    """
    Compute mean of top and bottom PC weights using sklearn PCA.

    Computes nPC principal components of the movie and returns the average
    frames at the top and bottom of each PC's temporal weights.

    Parameters
    ----------
    mov : np.ndarray
        Subsampled movie frames of shape (n_frames, Ly, Lx).
    nlowhigh : int
        Number of frames to average at the top and bottom of each PC.
    nPC : int
        Number of principal components to compute.
    random_state : int or None
        Seed for the PCA random state, used for reproducibility.

    Returns
    -------
    pclow : np.ndarray
        Average of bottom-weighted frames for each PC, shape (nPC, Ly, Lx).
    pchigh : np.ndarray
        Average of top-weighted frames for each PC, shape (nPC, Ly, Lx).
    w : np.ndarray
        Singular values from the PCA decomposition, shape (nPC,).
    v : np.ndarray
        Temporal PC weights of shape (n_frames, nPC), describing how each PC
        varies across frames.
    """
    nframes, Ly, Lx = mov.shape
    mov = mov.reshape((nframes, -1))
    mov = mov.astype(np.float32)
    mimg = mov.mean(axis=0)
    mov -= mimg
    pca = PCA(n_components=nPC, random_state=random_state).fit(mov.T)
    v = pca.components_.T
    w = pca.singular_values_
    mov += mimg
    mov = np.transpose(np.reshape(mov, (-1, Ly, Lx)), (1, 2, 0))
    pclow = np.zeros((nPC, Ly, Lx), np.float32)
    pchigh = np.zeros((nPC, Ly, Lx), np.float32)
    isort = np.argsort(v, axis=0)
    for i in range(nPC):
        pclow[i] = mov[:, :, isort[:nlowhigh, i]].mean(axis=-1)
        pchigh[i] = mov[:, :, isort[-nlowhigh:, i]].mean(axis=-1)
    return pclow, pchigh, w, v

def pclowhigh_torch(mov, nlowhigh, nPC, random_state):
    """
    Compute mean of top and bottom PC weights using torch SVD.

    Computes nPC principal components of the movie via torch SVD and returns the
    average frames at the top and bottom of each PC's temporal weights.

    Parameters
    ----------
    mov : torch.Tensor
        Subsampled movie frames of shape (n_frames, Ly, Lx).
    nlowhigh : int
        Number of frames to average at the top and bottom of each PC.
    nPC : int
        Number of principal components to compute.
    random_state : int or None
        Unused, kept for API compatibility with pclowhigh.

    Returns
    -------
    pclow : torch.Tensor
        Average of bottom-weighted frames for each PC, shape (nPC, Ly, Lx).
    pchigh : torch.Tensor
        Average of top-weighted frames for each PC, shape (nPC, Ly, Lx).
    w : torch.Tensor
        Singular values from the SVD decomposition.
    v : torch.Tensor
        Temporal PC weights of shape (n_frames, nPC), describing how each PC
        varies across frames.
    """
    nframes, Ly, Lx = mov.shape
    mov = mov.reshape((nframes, -1))
    mimg = mov.mean(axis=0)
    mov -= mimg
    w, v = torch.linalg.svd(mov.T, full_matrices=False)[1:]
    v = v.T
    mov += mimg
    mov = mov.reshape(nframes, Ly, Lx)
    pclow = torch.zeros((nPC, Ly, Lx), dtype=torch.float, device=mov.device)
    pchigh = torch.zeros((nPC, Ly, Lx), dtype=torch.float, device=mov.device)
    isort = v.argsort(axis=0)
    for i in range(nPC):
        pclow[i] = mov[isort[:nlowhigh, i]].mean(axis=0)
        pchigh[i] = mov[isort[-nlowhigh:, i]].mean(axis=0)
    return pclow, pchigh, w, v


def pc_register(pclow, pchigh, smooth_sigma=1.15, block_size=(128, 128),
                maxregshift=0.25, maxregshiftNR=15, snr_thresh=1.25,
                spatial_taper=3.45):
    """
    Register top and bottom PC averages to each other and compute shift magnitudes.

    For each PC, the bottom-weighted average image is used as a reference and the
    top-weighted average is registered to it using rigid and nonrigid shifts. The
    resulting shift magnitudes quantify registration quality.

    Parameters
    ----------
    pclow : torch.Tensor
        Average of bottom-weighted frames for each PC, shape (nPC, Ly, Lx).
    pchigh : torch.Tensor
        Average of top-weighted frames for each PC, shape (nPC, Ly, Lx).
    smooth_sigma : float
        Standard deviation (in pixels) of the Gaussian smoothing applied to the
        reference image during registration.
    block_size : tuple of int
        Block size (Ly_block, Lx_block) used for nonrigid registration.
    maxregshift : float
        Maximum allowed rigid registration shift as a fraction of the smaller
        image dimension.
    maxregshiftNR : int
        Maximum allowed nonrigid registration shift in pixels.
    snr_thresh : float
        Signal-to-noise ratio threshold for accepting nonrigid block shifts.
    spatial_taper : float
        Scalar controlling the slope of the spatial taper mask applied at image
        borders during registration.

    Returns
    -------
    X : np.ndarray
        Shift metrics of shape (nPC, 4) where X[:, 0] is the rigid shift magnitude,
        X[:, 1] is the mean nonrigid shift magnitude, X[:, 2] is the max nonrigid
        shift magnitude, and X[:, 3] is the mean combined rigid+nonrigid shift.
    """
    # registration settings
    nPC, Ly, Lx = pclow.shape

    X = np.zeros((nPC, 4))
    for i in range(nPC):
        refImg = pclow[i].cpu().numpy().copy()
        Img = pchigh[i][np.newaxis, :, :]

        refAndMasks = register.compute_filters_and_norm(refImg, norm_frames=True, spatial_smooth=smooth_sigma,
                                           spatial_taper=spatial_taper, 
                                           block_size=block_size,
                                                    device=Img.device)  
        fr_reg = Img.clone()
        offsets = register.compute_shifts(refAndMasks, fr_reg, maxregshift=maxregshift, smooth_sigma_time=0, 
                                          maxregshiftNR=maxregshiftNR, snr_thresh=snr_thresh)
        ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all = offsets
    
        X[i, 0] = ((ymax[0]**2 + xmax[0]**2)**.5).mean().cpu().numpy()
        X[i, 1] = ((ymax1**2 + xmax1**2)**.5).mean().cpu().numpy()
        X[i, 2] = ((ymax1**2 + xmax1**2)**.5).max().cpu().numpy()
        X[i, 3] = (((ymax[0] + ymax1)**2 + (xmax[0] + xmax1)**2)**0.5).mean().cpu().numpy()
    return X

def get_pc_metrics(f_reg, yrange=None, xrange=None, settings=default_settings()["registration"], 
                   device=torch.device("cpu")):

    """
    Compute registration metrics using top PCs of a registered movie.

    Subsamples frames from the registered movie, computes PCA to find the top and
    bottom weighted frames, then registers them to each other. The resulting shift
    magnitudes indicate registration quality: large shifts suggest residual motion.

    Parameters
    ----------
    f_reg : np.ndarray
        Registered movie of shape (n_frames, Ly, Lx).
    yrange : list of int or None
        [y_start, y_end] row range to crop the movie. If None, uses the full
        vertical extent.
    xrange : list of int or None
        [x_start, x_end] column range to crop the movie. If None, uses the full
        horizontal extent.
    settings : dict
        Registration settings dictionary containing keys such as "smooth_sigma",
        "block_size", "maxregshift", "maxregshiftNR", "snr_thresh",
        "spatial_taper", and optionally "reg_metrics_rs" and "reg_metric_n_pc".
    device : torch.device
        Torch device (CPU or CUDA) on which to run the PC registration.

    Returns
    -------
    tPC : np.ndarray
        Temporal PC weights of shape (n_samples, nPC), describing how each PC
        varies across the subsampled frames.
    regPC : np.ndarray
        Average of top and bottom weighted frames for each PC, shape
        (2, nPC, Ly_crop, Lx_crop) where index 0 is pclow and index 1 is pchigh.
    regDX : np.ndarray
        Shift metrics of shape (nPC, 4) from pc_register; see pc_register for
        column definitions.
    """
    n_frames, Ly, Lx = f_reg.shape
    yrange = [0, Ly] if yrange is None else yrange 
    xrange = [0, Lx] if xrange is None else xrange

    # n frames to pick from full movie
    nsamp = 2000 if n_frames < 5000 or Ly > 700 or Lx > 700 else 5000
    nsamp = min(nsamp, n_frames)
    inds = np.linspace(0, n_frames - 1, nsamp).astype("int")
    mov = f_reg[inds][:, yrange[0] : yrange[-1], xrange[0] : xrange[-1]]
    
    random_state = settings["reg_metrics_rs"] if "reg_metrics_rs" in settings else None
    nPC = settings["reg_metric_n_pc"] if "reg_metric_n_pc" in settings else 30
    pclow, pchigh, sv, tPC = pclowhigh(
        mov, nlowhigh=np.minimum(300, mov.shape[0] // 2), nPC=nPC,
        random_state=random_state)
    pclow = torch.from_numpy(pclow).to(device).float()
    pchigh = torch.from_numpy(pchigh).to(device).float()
    regPC = torch.stack((pclow, pchigh), dim=0).cpu().numpy()
    regDX = pc_register(
        pclow, pchigh, smooth_sigma=settings["smooth_sigma"], block_size=settings["block_size"],
        maxregshift=settings["maxregshift"], maxregshiftNR=settings["maxregshiftNR"], 
        snr_thresh=settings["snr_thresh"], spatial_taper=settings["spatial_taper"])
    return tPC, regPC, regDX