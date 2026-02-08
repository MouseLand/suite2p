"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import os
from os import path
from typing import Dict, Any
from warnings import warn
from tqdm import trange

import numpy as np
import torch
from scipy.signal import medfilt

import logging 
logger = logging.getLogger(__name__)


from .. import default_settings
from ..logger import TqdmToLogger
from . import bidiphase as bidi
from . import utils, rigid, nonrigid

device = torch.device("cuda")

def save_tiff(mov: np.ndarray, fname: str) -> None:
    """
    Save image stack array to a tiff file.

    Parameters
    ----------
    mov : np.ndarray
        Image stack of shape (nimg, Ly, Lx) to save. Values are floored and
        cast to int16 before writing.
    fname : str
        Output tiff file path.
    """
    from tifffile import TiffWriter
    with TiffWriter(fname) as tif:
        for frame in np.floor(mov).astype(np.int16):
            tif.write(frame, contiguous=True)

def compute_crop(xoff: int, yoff: int, corrXY, th_badframes, badframes, maxregshift,
                 Ly: int, Lx: int):
    """
    Determine how much to crop the FOV based on registration motion offsets.

    Identifies badframes (frames with large outlier shifts, thresholded by
    th_badframes) and excludes them when computing valid y and x ranges for
    cropping the field of view.

    Parameters
    ----------
    xoff : np.ndarray
        1-D array of length n_frames with x (column) rigid registration offsets.
    yoff : np.ndarray
        1-D array of length n_frames with y (row) rigid registration offsets.
    corrXY : np.ndarray
        1-D array of length n_frames with phase-correlation values for each frame.
    th_badframes : float
        Threshold multiplier for detecting bad frames based on the ratio of shift
        deviation to correlation quality.
    badframes : np.ndarray
        1-D boolean array of length n_frames with pre-existing bad frame labels.
    maxregshift : float
        Maximum allowed registration shift as a fraction of the image dimension.
        Frames exceeding 95% of this limit are marked as bad.
    Ly : int
        Height of a frame in pixels.
    Lx : int
        Width of a frame in pixels.

    Returns
    -------
    badframes : np.ndarray
        Updated 1-D boolean array of length n_frames indicating bad frames.
    yrange : list of int
        [ymin, ymax] valid row range after cropping for motion.
    xrange : list of int
        [xmin, xmax] valid column range after cropping for motion.
    """
    filter_window = min((len(yoff) // 2) * 2 - 1, 101)
    dx = xoff - medfilt(xoff, filter_window)
    dy = yoff - medfilt(yoff, filter_window)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy = dxy / dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, filter_window)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    if badframes.mean() < 0.5:
        ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
        xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    else:
        warn(
            "WARNING: >50% of frames have large movements, registration likely problematic"
        )
        ymin = np.ceil(np.abs(yoff).max())
        xmin = np.ceil(np.abs(xoff).max())
    ymax = Ly - ymin
    xmax = Lx - xmin
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange

def pick_initial_reference(frames: torch.Tensor):
    """
    Compute the initial reference image by finding the most correlated frame.

    The seed frame is the frame with the largest mean pairwise correlation with its
    20 top correlated frame pairs. The initial reference is the average of that seed frame and its
    top 20 most correlated frames.

    Parameters
    ----------
    frames : torch.Tensor
        Input frames of shape (n_frames, Ly, Lx).

    Returns
    -------
    refImg : np.ndarray
        Initial reference image of shape (Ly, Lx), dtype int16.
    """
    nimg, Ly, Lx = frames.shape
    fr_z = frames.clone().reshape(nimg, -1).double()
    fr_z -= fr_z.mean(dim=1, keepdim=True)
    cc = fr_z @ fr_z.T 
    ndiag = torch.diag(cc)**0.5
    cc = cc / torch.outer(ndiag, ndiag)
    CCsort = -torch.sort(-cc, dim=1)[0]
    # find frame most correlated to other frames
    bestCC = CCsort[:, 1:20].mean(dim=1) # 1-20 to exclude own frame
    imax = torch.argmax(bestCC)
    # average top 20 frames most correlated to imax
    indsort = torch.argsort(-cc[imax, :])
    refImg = fr_z[indsort[:20]].mean(axis=0).cpu().numpy().astype("int16")
    refImg = refImg.reshape(Ly, Lx)
    return refImg
    
def compute_reference(frames, settings=default_settings(), device=torch.device("cuda")):
    """
    Compute the reference image by iterative rigid alignment.

    Picks an initial reference via pick_initial_reference, then iteratively
    registers frames to the current reference and updates the reference as the
    mean of the best-correlated frames.

    Parameters
    ----------
    frames : np.ndarray
        Frames of shape (nimg_init, Ly, Lx), dtype int16, used to build the
        reference image.
    settings : dict
        Registration settings dictionary containing keys "batch_size",
        "smooth_sigma", "spatial_taper", and "maxregshift".
    device : torch.device
        Torch device (CPU or CUDA) on which to run registration.

    Returns
    -------
    refImg : np.ndarray
        Reference image of shape (Ly, Lx), dtype int16.
    """
    fr_reg = torch.from_numpy(frames)
    refImg = pick_initial_reference(fr_reg)
    
    niter = 8
    batch_size = settings["batch_size"]
    for iter in range(0, niter):
        # rigid registration shifts to reference
        maskMul, maskOffset, cfRefImg = compute_filters_and_norm(refImg, False, settings["smooth_sigma"],
                                           settings["spatial_taper"], block_size=None, device=device)[:3]

        for k in range(0, fr_reg.shape[0], batch_size):
            fr_reg_batch = fr_reg[k:min(k + batch_size, fr_reg.shape[0])].to(device)
            ymax, xmax, cmax = rigid.phasecorr(fr_reg_batch, cfRefImg, maskMul, maskOffset,
                maxregshift=settings["maxregshift"],
                smooth_sigma_time=settings["smooth_sigma_time"])[:3]
            
            # shift frames to reference
            fr_reg_batch = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                                for frame, dy, dx in zip(fr_reg_batch, ymax, xmax)], axis=0)
            fr_reg[k:min(k + batch_size, fr_reg.shape[0])] = fr_reg_batch.cpu()

        # frames to average for new reference
        nmax = max(2, int(frames.shape[0] * (1. + iter) / (2 * niter)))
        isort = torch.argsort(-cmax)[:nmax].cpu()
        refImg = fr_reg[isort].double().mean(dim=0)
        
        # recenter reference image
        if device.type == 'mps':
            # MPS backend currently can not support float64
            dy, dx = -torch.round(ymax[isort].to(torch.float32).mean()).int(), -torch.round(xmax[isort].to(torch.float32).mean()).int()
        else:
            dy, dx = -torch.round(ymax[isort].double().mean()).int(), -torch.round(xmax[isort].double().mean()).int()
        refImg = torch.roll(refImg, shifts=(-dy, -dx), dims=(0, 1))
        refImg = refImg.numpy().astype("int16")
        
    del fr_reg_batch 
    if device.type == "cuda":
        torch.cuda.empty_cache()    
        torch.cuda.synchronize()

    if device.type == "mps":
        torch.mps.empty_cache()
        torch.mps.synchronize()

    return refImg

def compute_filters_and_norm(refImg, norm_frames=True, spatial_smooth=1.15, spatial_taper=3.45,
                             block_size=(128, 128), lpad=3, subpixel=10, device=torch.device("cuda")):
    """
    Compute registration masks, smoothed reference FFTs, and normalization bounds.

    Builds rigid and (optionally) nonrigid spatial taper masks, smoothed
    Fourier-domain reference images, and intensity normalization bounds from the
    reference image. If refImg is a list (multi-plane), recurses for each plane.

    Parameters
    ----------
    refImg : np.ndarray or list of np.ndarray
        Reference image of shape (Ly, Lx), or a list of reference images for
        multi-plane registration.
    norm_frames : bool
        If True, clip the reference image to [1st, 99th] percentile and return
        the clipping bounds.
    spatial_smooth : float
        Standard deviation (in pixels) of Gaussian smoothing applied to the
        reference image in the frequency domain.
    spatial_taper : float
        Scalar controlling the slope of the sigmoid spatial taper mask at image
        borders.
    block_size : tuple of int or None
        Block size (Ly_block, Lx_block) for nonrigid registration. If None,
        nonrigid masks are not computed.
    lpad : int
        Number of pixels to pad each nonrigid block.
    subpixel : int
        Subpixel accuracy factor for nonrigid block shifts.
    device : torch.device
        Torch device to move the masks and reference FFTs to.

    Returns
    -------
    tuple
        If refImg is a single image, returns (maskMul, maskOffset, cfRefImg,
        maskMulNR, maskOffsetNR, cfRefImgNR, blocks, rmin, rmax). If refImg is
        a list, returns a list of such tuples.
    """
    if isinstance(refImg, list):
        refAndMasks_all = []
        for rimg in refImg:

            refAndMasks = compute_filters_and_norm(rimg, norm_frames=norm_frames, 
                                                   spatial_smooth=spatial_smooth, 
                                                   spatial_taper=spatial_taper, 
                                                   lpad=lpad, subpixel=subpixel, 
                                                   block_size=block_size, device=device)

            refAndMasks_all.append(refAndMasks)
        return refAndMasks_all
    else:
        if norm_frames:
            refImg, rmin, rmax = normalize_reference_image(refImg)
        else:
            rmin, rmax = -np.inf, np.inf

        rimg = torch.from_numpy(refImg)
        maskMul, maskOffset, cfRefImg = rigid.compute_masks_ref_smooth_fft(refImg=rimg, maskSlope=spatial_taper,
                                                                 smooth_sigma=spatial_smooth)
        Ly, Lx = refImg.shape
        maskMul, maskOffset = maskMul.to(device), maskOffset.to(device)
        cfRefImg = cfRefImg.to(device)
        blocks = []
        if block_size is not None:
            blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=block_size,
                                          lpad=lpad, subpixel=subpixel)
            maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.compute_masks_ref_smooth_fft(
                refImg0=rimg, maskSlope=spatial_taper, smooth_sigma=spatial_smooth, 
                yblock=blocks[0], xblock=blocks[1],
            )
            maskMulNR, maskOffsetNR = maskMulNR.to(device), maskOffsetNR.to(device)
            cfRefImgNR = cfRefImgNR.to(device)

        else:
            maskMulNR, maskOffsetNR, cfRefImgNR = None, None, None
        
        return (maskMul, maskOffset, cfRefImg, 
                maskMulNR, maskOffsetNR, cfRefImgNR, 
                blocks,
                rmin, rmax)

def compute_shifts(refAndMasks, fr_reg, maxregshift=0.1, smooth_sigma_time=0,
                   snr_thresh=1.2, maxregshiftNR=5, nZ=1):
    """
    Compute rigid and nonrigid registration shifts for a batch of frames.

    Performs rigid phase-correlation registration, then (if nonrigid masks are
    provided) applies rigid shifts and computes nonrigid block shifts. For
    multi-plane data (nZ > 1), selects the best z-plane per frame by maximum
    correlation.

    Parameters
    ----------
    refAndMasks : tuple or list of tuple
        Registration masks and reference FFTs from compute_filters_and_norm. If
        nZ > 1, a list of tuples (one per z-plane).
    fr_reg : torch.Tensor
        Frames to register, shape (N, Ly, Lx).
    maxregshift : float
        Maximum allowed rigid shift as a fraction of the smaller image dimension.
    smooth_sigma_time : float
        Sigma for temporal smoothing of phase-correlation maps. If <= 0, no
        temporal smoothing is applied.
    snr_thresh : float
        Signal-to-noise ratio threshold for accepting nonrigid block shifts.
    maxregshiftNR : int
        Maximum allowed nonrigid shift in pixels.
    nZ : int
        Number of z-planes. If > 1, performs multi-plane registration.

    Returns
    -------
    ymax : torch.LongTensor
        1-D rigid y shifts of length N.
    xmax : torch.LongTensor
        1-D rigid x shifts of length N.
    cmax : torch.Tensor
        1-D rigid correlation values of length N.
    ymax1 : torch.Tensor or None
        Nonrigid y shifts of shape (N, n_blocks), or None if nonrigid is disabled.
    xmax1 : torch.Tensor or None
        Nonrigid x shifts of shape (N, n_blocks), or None if nonrigid is disabled.
    cmax1 : torch.Tensor or None
        Nonrigid correlation values of shape (N, n_blocks), or None.
    zest : np.ndarray or None
        Best z-plane index per frame of length N (only if nZ > 1), else None.
    cmax_all : np.ndarray or None
        Correlation values across all z-planes of shape (N, nZ) (only if nZ > 1),
        else None.
    """
    n_fr = fr_reg.shape[0]
    if nZ > 1:
        # find best plane
        offsets_all = []
        for z in range(nZ):
            fr_reg0 = fr_reg.clone()
            offsets0 = compute_shifts(refAndMasks[z], fr_reg0, maxregshift, 
                                      smooth_sigma_time, snr_thresh, 
                                      maxregshiftNR, nZ=1)
            offsets_all.append(offsets0)
        cmax_all = np.array([offsets[2].cpu().numpy() for offsets in offsets_all]).T
        zest = cmax_all.argmax(axis=1)

        nb = refAndMasks[0][3].shape[0] if refAndMasks[0][3] is not None else 0
        device = fr_reg.device
        shapes = [(n_fr,), (n_fr,), (n_fr,), (n_fr, nb), (n_fr, nb), (n_fr, nb)]
        offsets_best = [torch.zeros(shapes[i], device=device, 
                                    dtype=torch.float32 if i > 1 else torch.long) 
                        for i in range(6)]
        for z in range(nZ):
            iz = np.nonzero(zest == z)[0]
            if len(iz) > 0:
                for i, offsets in enumerate(offsets_all[z][:6]):
                    offsets_best[i][iz] = offsets[iz] if offsets is not None else 0
        
        return *offsets_best[:6], zest, cmax_all
            
    else:
        (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, 
        blocks, rmin, rmax) = refAndMasks
        device = fr_reg.device

        fr_reg = torch.clip(fr_reg, rmin, rmax) if rmin > -np.inf else fr_reg

        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(fr_reg, cfRefImg, maskMul, maskOffset, 
                                        maxregshift, smooth_sigma_time)[:3]
            
        # non-rigid registration
        if maskMulNR is not None and maxregshiftNR > 0:     
            # shift torch frames to reference
            fr_reg = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                                for frame, dy, dx in zip(fr_reg, ymax, xmax)], axis=0)
            ymax1, xmax1, cmax1 = nonrigid.phasecorr(fr_reg, blocks, 
                                                    maskMulNR, maskOffsetNR, cfRefImgNR, 
                                                    snr_thresh, maxregshiftNR)[:3]
        else:    
            ymax1, xmax1, cmax1 = None, None, None

        del fr_reg
        if device.type == "cuda":
            torch.cuda.empty_cache()    

        if device.type == "mps":
            torch.mps.empty_cache()

    return ymax, xmax, cmax, ymax1, xmax1, cmax1, None, None

def shift_frames(fr_torch, yoff, xoff, yoff1=None, xoff1=None, blocks=None, device=torch.device("cuda")):
    """
    Apply rigid and optionally nonrigid shifts to frames and return as numpy int16.

    Parameters
    ----------
    fr_torch : torch.Tensor
        Frames to shift, shape (N, Ly, Lx).
    yoff : torch.LongTensor
        1-D rigid y shifts of length N.
    xoff : torch.LongTensor
        1-D rigid x shifts of length N.
    yoff1 : torch.Tensor or np.ndarray or None
        Nonrigid y shifts of shape (N, n_blocks). If None, only rigid shifts are
        applied.
    xoff1 : torch.Tensor or np.ndarray or None
        Nonrigid x shifts of shape (N, n_blocks).
    blocks : list or None
        Block definitions from nonrigid.make_blocks, used for nonrigid
        interpolation.
    device : torch.device
        Torch device for nonrigid shift tensors.

    Returns
    -------
    frames_out : np.ndarray
        Shifted frames of shape (N, Ly, Lx), dtype matching the torch output.
    """
    fr_torch = torch.stack([torch.roll(frame, shifts=(-dy, -dx), dims=(0, 1))
                               for frame, dy, dx in zip(fr_torch, yoff, xoff)], axis=0)

    if yoff1 is not None:
        if isinstance(yoff1, np.ndarray):
            if fr_torch.device.type == "cuda":
                yoff1 = torch.from_numpy(yoff1).pin_memory().to(device)
                xoff1 = torch.from_numpy(xoff1).pin_memory().to(device)
            else:
                yoff1 = torch.from_numpy(yoff1).to(device)
                xoff1 = torch.from_numpy(xoff1).to(device)
        fr_torch = nonrigid.transform_data(fr_torch, blocks[2], blocks[1], blocks[0], yoff1, xoff1)
    
    frames_out = np.empty(fr_torch.shape, dtype="int16")
    frames_out = fr_torch.cpu().numpy()
        
    return frames_out

def normalize_reference_image(refImg):
    """
    Clip reference image to [1st, 99th] intensity percentiles.

    Parameters
    ----------
    refImg : np.ndarray
        Reference image of shape (Ly, Lx).

    Returns
    -------
    refImg : np.ndarray
        Clipped reference image of shape (Ly, Lx).
    rmin : np.int16
        1st percentile intensity value used as the lower clip bound.
    rmax : np.int16
        99th percentile intensity value used as the upper clip bound.
    """
    rmin, rmax = np.percentile(refImg, [1, 99]).astype(np.int16)
    refImg = np.clip(refImg, rmin, rmax)
    return refImg, rmin, rmax


def register_frames(f_align_in, refImg, f_align_out=None, batch_size=100, 
                    bidiphase=0, 
                    norm_frames=True, smooth_sigma=1.15, spatial_taper=3.45, 
                    block_size=(128,128), nonrigid=True, maxregshift=0.1, 
                    smooth_sigma_time=0, snr_thresh=1.2, maxregshiftNR=5,
                    device=torch.device("cuda"), tif_root=None, apply_shifts=True):
    """
    Register frames to a reference image using rigid and optionally nonrigid shifts.

    Computes registration masks from the reference, then processes frames in
    batches: computes shifts, applies them, accumulates a mean image, and
    optionally writes registered frames to f_align_out. Supports multi-plane
    registration when refImg is a list.

    Parameters
    ----------
    f_align_in : np.ndarray or BinaryFile
        Input frames of shape (n_frames, Ly, Lx), supporting slice indexing.
    refImg : np.ndarray or list of np.ndarray
        Reference image of shape (Ly, Lx), or a list for multi-plane registration.
    f_align_out : np.ndarray or BinaryFile or None
        Output array for registered frames. If None, registered frames are
        written back to f_align_in.
    batch_size : int
        Number of frames to process per batch.
    bidiphase : int
        Bidirectional phase offset in pixels. If non-zero, frames are corrected
        before registration.
    norm_frames : bool
        If True, clip frames to the reference image's [1st, 99th] percentile range.
    smooth_sigma : float
        Standard deviation of Gaussian smoothing applied to the reference image.
    spatial_taper : float
        Slope of the sigmoid spatial taper mask at image borders.
    block_size : tuple of int
        Block size (Ly_block, Lx_block) for nonrigid registration.
    nonrigid : bool
        If True, compute nonrigid shifts in addition to rigid shifts.
    maxregshift : float
        Maximum rigid shift as a fraction of the smaller image dimension.
    smooth_sigma_time : float
        Sigma for temporal smoothing of phase-correlation maps.
    snr_thresh : float
        SNR threshold for accepting nonrigid block shifts.
    maxregshiftNR : int
        Maximum nonrigid shift in pixels.
    device : torch.device
        Torch device for computation.
    tif_root : str or None
        If provided, save registered frames as tiffs in this directory.
    apply_shifts : bool
        If True, apply computed shifts to frames. If False, only compute shifts.

    Returns
    -------
    rmin : np.int16 or list
        Lower intensity clip bound(s) from reference normalization.
    rmax : np.int16 or list
        Upper intensity clip bound(s) from reference normalization.
    mean_img : np.ndarray
        Mean registered image of shape (Ly, Lx).
    offsets_all : list
        List of [yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all]
        concatenated across all batches.
    blocks : list
        Block definitions from nonrigid.make_blocks.
    """

    n_frames, Ly, Lx = f_align_in.shape

    if isinstance(refImg, list):
        nZ = len(refImg)
        logger.info(f"List of reference frames len = {nZ}")
    else:
        nZ = 1

    refAndMasks = compute_filters_and_norm(refImg, norm_frames=norm_frames, 
                                           spatial_smooth=smooth_sigma,
                                           spatial_taper=spatial_taper, 
                                           block_size=block_size if nonrigid else None, 
                                           device=device)
    blocks = refAndMasks[-3] if nZ==1 else refAndMasks[0][-3]
    rmin = refAndMasks[-2] if nZ==1 else [refAndMasks[z][-2] for z in range(nZ)]
    rmax = refAndMasks[-1] if nZ==1 else [refAndMasks[z][-1] for z in range(nZ)]
    ### ------------- register frames to reference image ------------ ###

    mean_img = np.zeros((Ly, Lx), "float32")
    
    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info(f"Registering {n_frames} frames in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        frames = f_align_in[tstart : tend]
        if device.type == "cuda":
            fr_torch = torch.from_numpy(frames).pin_memory().to(device)
        else:
            fr_torch = torch.from_numpy(frames).to(device)
        if bidiphase != 0:
            fr_torch = bidi.shift(fr_torch, bidiphase)

        fr_reg = fr_torch.clone()
        offsets = compute_shifts(refAndMasks, fr_reg, maxregshift=maxregshift, 
                                 smooth_sigma_time=smooth_sigma_time, 
                                 snr_thresh=snr_thresh, maxregshiftNR=maxregshiftNR, 
                                 nZ=nZ)
        ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all = offsets

        if apply_shifts:
            frames = shift_frames(fr_torch, ymax, xmax, ymax1, xmax1, blocks, device)
        
        # convert to numpy and concatenate offsets
        ymax, xmax, cmax = ymax.cpu().numpy(), xmax.cpu().numpy(), cmax.cpu().numpy()
        if ymax1 is not None:
            ymax1, xmax1 = ymax1.cpu().numpy(), xmax1.cpu().numpy()
            cmax1 = cmax1.cpu().numpy()
        offsets = [ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all]
        offsets_all = ([np.concatenate((offset_all, offset), axis=0) 
                       if offset is not None else None
                       for offset_all, offset in zip(offsets_all, offsets)] 
                        if n > 0 else offsets)
        
        # make mean image from all registered frames
        mean_img += frames.sum(axis=0) / n_frames

        # save aligned frames to bin file
        if apply_shifts:
            if f_align_out is not None:
                f_align_out[tstart : tend] = frames
            else:
                f_align_in[tstart : tend] = frames

            # save aligned frames to tiffs
            if tif_root:
                fname = os.path.join(tif_root, f"file{n : 05d}.tif")
                save_tiff(mov=frames, fname=fname)

    return rmin, rmax, mean_img, offsets_all, blocks

def check_offsets(yoff, xoff, yoff1, xoff1, n_frames):
    """
    Validate that registration offset arrays have the expected number of frames.

    Parameters
    ----------
    yoff : np.ndarray or None
        Rigid y offsets of length n_frames.
    xoff : np.ndarray or None
        Rigid x offsets of length n_frames.
    yoff1 : np.ndarray or None
        Nonrigid y offsets of shape (n_frames, n_blocks), or None.
    xoff1 : np.ndarray or None
        Nonrigid x offsets of shape (n_frames, n_blocks), or None.
    n_frames : int
        Expected number of frames.

    Raises
    ------
    ValueError
        If rigid offsets are None or any offset array length does not match
        n_frames.
    """
    if yoff is None or xoff is None:
        raise ValueError("no rigid registration offsets provided")
    elif yoff.shape[0] != n_frames or xoff.shape[0] != n_frames:
        raise ValueError(
            "rigid registration offsets are not the same size as input frames")
    if yoff1 is not None and (yoff1.shape[0] != n_frames or xoff1.shape[0] != n_frames):
        raise ValueError(
                "nonrigid registration offsets are not the same size as input frames")

def shift_frames_and_write(f_alt_in, f_alt_out=None, batch_size=100, yoff=None, xoff=None, yoff1=None,
                           xoff1=None, blocks=None, bidiphase=0, 
                           device=torch.device("cuda"), tif_root=None):
    """
    Apply pre-computed registration shifts to an alternate channel and write results.

    Applies rigid (and optionally nonrigid) shifts that were computed on the
    primary channel to the alternate channel frames, in batches. Writes the
    shifted frames to f_alt_out if provided, otherwise overwrites f_alt_in.

    Parameters
    ----------
    f_alt_in : np.ndarray or BinaryFile
        Alternate channel input frames of shape (n_frames, Ly, Lx).
    f_alt_out : np.ndarray or BinaryFile or None
        Output array for shifted frames. If None, writes back to f_alt_in.
    batch_size : int
        Number of frames per batch.
    yoff : np.ndarray
        Rigid y offsets of length n_frames.
    xoff : np.ndarray
        Rigid x offsets of length n_frames.
    yoff1 : np.ndarray or None
        Nonrigid y offsets of shape (n_frames, n_blocks).
    xoff1 : np.ndarray or None
        Nonrigid x offsets of shape (n_frames, n_blocks).
    blocks : list or None
        Block definitions from nonrigid.make_blocks.
    bidiphase : int
        Bidirectional phase offset in pixels.
    device : torch.device
        Torch device for computation.
    tif_root : str or None
        If provided, save shifted frames as tiffs in this directory.

    Returns
    -------
    mean_img : np.ndarray
        Mean image of the shifted alternate channel, shape (Ly, Lx).
    """
    n_frames, Ly, Lx = f_alt_in.shape
    check_offsets(yoff, xoff, yoff1, xoff1, n_frames)

    mean_img = np.zeros((Ly, Lx), "float32")
    yoff1k, xoff1k = None, None
    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info(f"Second channel: Shifting {n_frames} frames in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        frames = f_alt_in[tstart : tend]
        yoffk, xoffk = yoff[tstart : tend].astype(int), xoff[tstart : tend].astype(int)
        if yoff1 is not None:
            yoff1k, xoff1k = yoff1[tstart : tend], xoff1[tstart : tend]
            
        if device.type == "cuda":
            fr_torch = torch.from_numpy(frames).pin_memory().to(device)
        else:
            fr_torch = torch.from_numpy(frames).to(device)

        if bidiphase != 0:
            fr_torch = bidi.shift(fr_torch, bidiphase)
        frames = shift_frames(fr_torch, yoffk, xoffk, yoff1k, xoff1k, blocks, device=device)
        mean_img += frames.sum(axis=0) / n_frames

        if f_alt_out is None:
            f_alt_in[tstart : tend] = frames
        else:
            f_alt_out[tstart : tend] = frames

        # save aligned frames to tiffs
        if tif_root:
            fname = os.path.join(tif_root, f"file{n : 05d}.tif")
            save_tiff(mov=frames, fname=fname)

    return mean_img


def assign_reg_io(f_reg, f_raw=None, f_reg_chan2=None, 
               f_raw_chan2=None, align_by_chan2=False, 
               save_path=None,
               reg_tif=False, reg_tif_chan2=False):
    """
    Assign input/output arrays and tiff directories for registration I/O.

    Determines which channel is the alignment source and which is the alternate,
    based on align_by_chan2. Sets up tiff output directories if requested.

    Parameters
    ----------
    f_reg : np.ndarray or BinaryFile
        Registered functional channel frames.
    f_raw : np.ndarray or BinaryFile or None
        Raw functional channel frames, used as input when available.
    f_reg_chan2 : np.ndarray or BinaryFile or None
        Registered second channel frames.
    f_raw_chan2 : np.ndarray or BinaryFile or None
        Raw second channel frames.
    align_by_chan2 : bool
        If True, use the second channel as the alignment source.
    save_path : str or None
        Base directory for saving registered tiff files.
    reg_tif : bool
        If True, save registered functional channel frames as tiffs.
    reg_tif_chan2 : bool
        If True, save registered second channel frames as tiffs.

    Returns
    -------
    f_align_in : np.ndarray or BinaryFile
        Input frames for alignment.
    f_align_out : np.ndarray or BinaryFile or None
        Output destination for aligned frames.
    f_alt_in : np.ndarray or BinaryFile or None
        Input frames for the alternate channel.
    f_alt_out : np.ndarray or BinaryFile or None
        Output destination for shifted alternate channel frames.
    tif_root_align : str or None
        Tiff output directory for the alignment channel.
    tif_root_alt : str or None
        Tiff output directory for the alternate channel.
    """
    if f_reg_chan2 is None or not align_by_chan2:
        f_align_in = f_reg if not f_raw else f_raw
        f_alt_in = f_reg_chan2 if not f_raw_chan2 else f_raw_chan2
        f_align_out = f_reg if f_raw else None
        f_alt_out = f_reg_chan2 if f_raw_chan2 else None
    else:
        f_align_in = f_reg_chan2 if not f_raw_chan2 else f_raw_chan2
        f_alt_in = f_reg if not f_raw else f_raw
        f_align_out  = f_reg_chan2 if f_raw_chan2 else None
        f_alt_out = f_reg if f_raw else None

    if f_alt_in is not None:
        if f_align_in.shape[0] != f_alt_in.shape[0]:
            raise ValueError("number of frames in f_align_in and f_alt_in must match")
        
    if save_path:
        tif_root_align, tif_root_alt = None, None
        if reg_tif:
            tifroot = os.path.join(save_path, "reg_tif")
            os.makedirs(tifroot, exist_ok=True)
            if not align_by_chan2:
                tif_root_align = tifroot
            else:
                tif_root_alt = tifroot
        if reg_tif_chan2:
            tifroot = os.path.join(save_path, "reg_tif_chan2")
            os.makedirs(tifroot, exist_ok=True)
            if align_by_chan2:
                tif_root_align = tifroot
            else:
                tif_root_alt = tifroot

    return f_align_in, f_align_out, f_alt_in, f_alt_out, tif_root_align, tif_root_alt


def registration_wrapper(f_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None,
                        refImg=None, align_by_chan2=False, save_path=None, aspect=1.,
                        badframes=None, settings=default_settings(), device=torch.device("cuda")):
    """
    Main registration function for single- or dual-channel movies.

    Computes a reference image (if not provided), estimates bidirectional phase
    offset, registers the primary channel, optionally performs a two-step
    registration, applies shifts to an alternate channel if present, and returns
    all registration outputs as a dictionary.

    Parameters
    ----------
    f_reg : np.ndarray or BinaryFile
        Registered functional channel frames of shape (n_frames, Ly, Lx).
    f_raw : np.ndarray or BinaryFile or None
        Raw functional channel frames. If provided, used as the registration
        input with f_reg as the output destination.
    f_reg_chan2 : np.ndarray or BinaryFile or None
        Registered second channel frames.
    f_raw_chan2 : np.ndarray or BinaryFile or None
        Raw second channel frames.
    refImg : np.ndarray or None
        Reference image of shape (Ly, Lx), dtype int16. If None, a reference is
        computed from the data.
    align_by_chan2 : bool
        If True, use the second channel as the alignment source.
    save_path : str or None
        Base directory for saving registered tiff files.
    aspect : float
        Pixel aspect ratio used for computing the enhanced mean image.
    badframes : np.ndarray or None
        1-D boolean array of pre-existing bad frame labels. If None, initialized
        to all False.
    settings : dict
        Registration settings dictionary from default_settings().
    device : torch.device
        Torch device for computation.

    Returns
    -------
    reg_outputs : dict
        Dictionary containing registration results with keys: "refImg", "rmin",
        "rmax", "meanImg", "yoff", "xoff", "corrXY", "yoff1", "xoff1",
        "corrXY1", "meanImg_chan2", "badframes", "badframes0", "yrange",
        "xrange", "bidiphase", "meanImgE", and optionally "zpos_registration"
        and "cmax_registration".
    """
    out = assign_reg_io(f_reg, f_raw, f_reg_chan2, f_raw_chan2, align_by_chan2,
                        save_path, settings["reg_tif"], settings["reg_tif_chan2"])
    f_align_in, f_align_out, f_alt_in, f_alt_out, tif_root_align, tif_root_alt = out

    nchannels = 2 if f_alt_in is not None else 1
    logger.info(f"registering {nchannels} channels")
    
    ### ----- compute reference image and bidiphase shift -------------- ###
    n_frames, Ly, Lx = f_align_in.shape
    badframes0 = np.zeros(n_frames, "bool") if badframes is None else badframes.copy()

    compute_bidi = settings["do_bidiphase"] and settings["bidiphase"] == 0
    # grab frames
    if refImg is None or compute_bidi:
        ix_frames = np.linspace(0, n_frames, 1 + min(settings["nimg_init"], n_frames), 
                                dtype=int)[:-1]
        frames = f_align_in[ix_frames].copy()
    
    # compute bidiphase shift
    if compute_bidi:
        bidiphase = bidi.compute(frames)
        logger.info("Estimated bidiphase offset from data: %d pixels" % bidiphase)
        # shift frames for reference image computation
    else:
        bidiphase = settings["bidiphase"]
    
    if bidiphase != 0 and refImg is None:
        frames = bidi.shift(frames, int(settings["bidiphase"])) 
    
    if refImg is None:
        t0 = time.time()
        refImg = compute_reference(frames, settings=settings, device=device)
        logger.info("Reference frame, %0.2f sec." % (time.time() - t0))
    refImg_orig = refImg.copy()
    
    for step in range(1 + (settings["two_step_registration"] and f_raw is not None)):
        if step == 1:
            logger.info("starting step 2 of two-step registration")
            logger.info("(making new reference image without badframes)")
            nsamps = min(n_frames, settings["nimg_init"])
            inds = np.linspace(0, n_frames, 1 + nsamps).astype(np.int64)[:-1]
            inds = inds[~np.isin(inds, np.nonzero(badframes)[0])]
            refImg = f_align_out[inds].astype(np.float32).mean(axis=0)
            refImg_orig = refImg.copy()
            
        ### ----- register frames to reference image -------------- ###
        outputs = register_frames(f_align_in, f_align_out=f_align_out, bidiphase=bidiphase,
                                refImg=refImg, tif_root=tif_root_align, 
                                batch_size=settings["batch_size"], 
                                norm_frames=settings["norm_frames"], smooth_sigma=settings["smooth_sigma"], 
                                spatial_taper=settings["spatial_taper"], block_size=settings["block_size"], 
                                nonrigid=settings["nonrigid"],
                                maxregshift=settings["maxregshift"], smooth_sigma_time=settings["smooth_sigma_time"],
                                    snr_thresh=settings["snr_thresh"], maxregshiftNR=settings["maxregshiftNR"],
                                    device=device)
        rmin, rmax, mean_img, offsets_all, blocks = outputs
        yoff, xoff, corrXY, yoff1, xoff1, corrXY1, zest, cmax_all = offsets_all

        # compute valid region and timepoints to exclude
        badframes, yrange, xrange = compute_crop(xoff=xoff, yoff=yoff, corrXY=corrXY,
                                             th_badframes=settings["th_badframes"],
                                             badframes=badframes0.copy(),
                                             maxregshift=settings["maxregshift"], Ly=Ly,
                                             Lx=Lx)
        
    ### ----- register second channel -------------- ###
    if nchannels > 1:
        mean_img_alt = shift_frames_and_write(f_alt_in, f_alt_out, settings["batch_size"], yoff, xoff, yoff1,
                                              xoff1, blocks=blocks, bidiphase=bidiphase,
                                              tif_root=tif_root_align, device=device)
    else:
        mean_img_alt = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    if device.type == "mps":
        torch.mps.empty_cache()

    meanImg = mean_img if nchannels == 1 or not align_by_chan2 else mean_img_alt
    if nchannels == 2:
        meanImg_chan2 = mean_img_alt if not align_by_chan2 else mean_img
    else:
        meanImg_chan2 = None

    reg_outputs = registration_outputs_to_dict(refImg_orig, rmin, rmax, meanImg, 
                                               (yoff, xoff, corrXY), 
                                               (yoff1, xoff1, corrXY1), 
                                               (zest, cmax_all), meanImg_chan2, 
                                               badframes, badframes0, 
                                               yrange, xrange, bidiphase)
    
    # add enhanced mean image
    meanImgE = utils.highpass_mean_image(meanImg.astype("float32"), aspect=aspect)
    reg_outputs["meanImgE"] = meanImgE
    return reg_outputs

def registration_outputs_to_dict(refImg, rmin, rmax, meanImg, rigid_offsets,
                                 nonrigid_offsets, zest, meanImg_chan2,
                                 badframes, badframes0, yrange, xrange, bidiphase):
    """
    Pack registration results into a dictionary.

    Parameters
    ----------
    refImg : np.ndarray
        Reference image of shape (Ly, Lx).
    rmin : np.int16
        Lower intensity clip bound.
    rmax : np.int16
        Upper intensity clip bound.
    meanImg : np.ndarray
        Mean registered image of shape (Ly, Lx).
    rigid_offsets : tuple
        Tuple of (yoff, xoff, corrXY) rigid registration offsets.
    nonrigid_offsets : tuple
        Tuple of (yoff1, xoff1, corrXY1) nonrigid offsets, elements may be None.
    zest : tuple
        Tuple of (zpos, cmax_all) for multi-plane registration, elements may be
        None.
    meanImg_chan2 : np.ndarray or None
        Mean image of the second channel, shape (Ly, Lx).
    badframes : np.ndarray
        1-D boolean array of detected bad frames.
    badframes0 : np.ndarray
        1-D boolean array of initial bad frames before registration.
    yrange : list of int
        [ymin, ymax] valid row range.
    xrange : list of int
        [xmin, xmax] valid column range.
    bidiphase : int
        Bidirectional phase offset in pixels.

    Returns
    -------
    reg_outputs : dict
        Dictionary with keys "refImg", "rmin", "rmax", "yoff", "xoff",
        "corrXY", "meanImg", "badframes", "badframes0", "yrange", "xrange",
        "bidiphase", and optionally "yoff1", "xoff1", "corrXY1",
        "meanImg_chan2", "zpos_registration", "cmax_registration".
    """
    reg_outputs = {}
    # assign reference image and normalizers
    reg_outputs["refImg"] = refImg
    reg_outputs["rmin"], reg_outputs["rmax"] = rmin, rmax
    # assign rigid offsets to reg_outputs
    reg_outputs["yoff"], reg_outputs["xoff"], reg_outputs["corrXY"] = rigid_offsets
    # assign nonrigid offsets to reg_outputs
    if nonrigid_offsets[0] is not None:
        reg_outputs["yoff1"], reg_outputs["xoff1"], reg_outputs["corrXY1"] = nonrigid_offsets
    # assign mean images
    reg_outputs["meanImg"] = meanImg
    if meanImg_chan2 is not None:
        reg_outputs["meanImg_chan2"] = meanImg_chan2
    # assign crop computation and badframes
    reg_outputs["badframes"], reg_outputs["badframes0"] = badframes, badframes0
    reg_outputs["yrange"], reg_outputs["xrange"] = yrange, xrange
    if zest[0] is not None:
        reg_outputs["zpos_registration"] = np.array(zest[0])
        reg_outputs["cmax_registration"] = np.array(zest[1])
    reg_outputs["bidiphase"] = bidiphase
    return reg_outputs
