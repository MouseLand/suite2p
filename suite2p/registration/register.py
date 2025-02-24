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
    Save image stack array to tiff file.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to save
    fname: str
        The tiff filename to save to

    """
    from tifffile import TiffWriter
    with TiffWriter(fname) as tif:
        for frame in np.floor(mov).astype(np.int16):
            tif.write(frame, contiguous=True)

def compute_crop(xoff: int, yoff: int, corrXY, th_badframes, badframes, maxregshift,
                 Ly: int, Lx: int):
    """ determines how much to crop FOV based on motion
    
    determines badframes which are frames with large outlier shifts
    (threshold of outlier is th_badframes) and
    it excludes these badframes when computing valid ranges
    from registration in y and x

    Parameters
    __________
    xoff: int
    yoff: int
    corrXY
    th_badframes
    badframes
    maxregshift
    Ly: int
        Height of a frame
    Lx: int
        Width of a frame

    Returns
    _______
    badframes
    yrange
    xrange
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
    """ computes the initial reference frames

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

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
    """ computes the reference image

    picks initial reference then iteratively aligns frames to create reference

    Parameters
    ----------
    
    settings : dictionary
        need registration options

    frames : 3D array, int16
        size [nimg_init x Ly x Lx], frames to use to create initial reference

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

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
    ### ------------- compute registration masks ----------------- ###
    if isinstance(refImg, list):
        refAndMasks_all = []
        for rimg in refImg:
            refAndMasks = compute_filters_and_norm(rimg, norm_frames, spatial_smooth, 
                                                   spatial_taper, block_size, device)
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
                   snr_thresh=1.2, maxregshiftNR=5):
    
    (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, 
     blocks, rmin, rmax) = refAndMasks
    device = fr_reg.device

    fr_reg = torch.clip(fr_reg, rmin, rmax) if rmin > -np.inf else fr_reg

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(fr_reg, cfRefImg, maskMul, maskOffset, 
                                       maxregshift, smooth_sigma_time)[:3]
        
    # non-rigid registration
    if maskMulNR is not None:
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

def shift_frames(fr_torch, yoff, xoff, yoff1=None, xoff1=None, blocks=None):
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
    if isinstance(refImg, list):
        rmins = []
        rmaxs = []
        for rimg in refImg:
            rimg[:], rmin, rmax = normalize_reference_image(rimg)
            rmins.append(rmin)
            rmaxs.append(rmax)
        return refImg, rmins, rmaxs
    else:
        rmin, rmax = np.percentile(refImg, [1, 99]).astype(np.int16)
        refImg = np.clip(refImg, rmin, rmax)
        return refImg, rmin, rmax


def register_frames(f_align_in, refImg, f_align_out=None, batch_size=100, 
                    bidiphase=0, 
                    norm_frames=True, smooth_sigma=1.15, spatial_taper=3.45, 
                    block_size=(128,128), nonrigid=True, maxregshift=0.1, 
                    smooth_sigma_time=0, snr_thresh=1.2, maxregshiftNR=5,
                    device=torch.device("cuda"), tif_root=None):
    """ align frames in f_align_in to reference 
    
    if f_align_out is not None, registered frames are written to f_align_out

    f_align_in, f_align_out can be a BinaryFile or any type of array that can be slice-indexed
    
    """

    n_frames, Ly, Lx = f_align_in.shape

    if isinstance(refImg, list):
        nZ = len(refImg)
        logger.info(f"List of reference frames len = {nZ}")

    refAndMasks = compute_filters_and_norm(refImg, norm_frames=norm_frames, 
                                           spatial_smooth=smooth_sigma,
                                           spatial_taper=spatial_taper, 
                                           block_size=block_size if nonrigid else None, 
                                           device=device)
    (maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, 
        cfRefImgNR, blocks, rmin, rmax) = refAndMasks
    ### ------------- register frames to reference image ------------ ###

    mean_img = np.zeros((Ly, Lx), "float32")
    rigid_offsets, nonrigid_offsets, zpos, cmax_all = [], [], [], []

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
                                 snr_thresh=snr_thresh, maxregshiftNR=maxregshiftNR)
        ymax, xmax, cmax, ymax1, xmax1, cmax1, zest, cmax_all = offsets
        frames = shift_frames(fr_torch, ymax, xmax, ymax1, xmax1, blocks)
        
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
        if f_align_out is not None:
            f_align_out[tstart : tend] = frames
        else:
            f_align_in[tstart : tend] = frames

        # save aligned frames to tiffs
        if tif_root:
            fname = os.path.join(tif_root, f"file{n : 05d}.tif")
            io.save_tiff(mov=frames, fname=fname)

    return rmin, rmax, mean_img, offsets_all, blocks

def check_offsets(yoff, xoff, yoff1, xoff1, n_frames):
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
    """ shift frames for alternate channel in f_alt_in and write to f_alt_out if not None (else write to f_alt_in) """
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
            
        if device.type == "cpu":
            fr_torch = torch.from_numpy(frames).to(device)
        else:
            fr_torch = torch.from_numpy(frames).pin_memory().to(device)

        if bidiphase != 0:
            fr_torch = bidi.shift(fr_torch, bidiphase)
        frames = shift_frames(fr_torch, yoffk, xoffk, yoff1k, xoff1k, blocks)
        mean_img += frames.sum(axis=0) / n_frames

        if f_alt_out is None:
            f_alt_in[tstart : tend] = frames
        else:
            f_alt_out[tstart : tend] = frames

        # save aligned frames to tiffs
        if tif_root:
            fname = os.path.join(tif_root, f"file{n : 05d}.tif")
            io.save_tiff(mov=frames, fname=fname)

    return mean_img


def assign_reg_io(f_reg, f_raw=None, f_reg_chan2=None, 
               f_raw_chan2=None, align_by_chan2=False, 
               save_path=None,
               reg_tif=False, reg_tif_chan2=False):
    """ check inputs and assign input arrays to appropriate variables """
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
    """Main registration function.

    Args:
        f_reg (array): Array of registered functional frames, np.ndarray or io.BinaryFile.
        f_raw (array, optional): Array of raw functional frames, np.ndarray or io.BinaryFile. Defaults to None.
        f_reg_chan2 (array, optional): Array of registered anatomical frames, np.ndarray or io.BinaryFile. Defaults to None.
        f_raw_chan2 (array, optional): Array of raw anatomical frames, np.ndarray or io.BinaryFile. Defaults to None.
        refImg (2D array, optional): 2D array of int16, size [Ly x Lx], initial reference image. Defaults to None.
        align_by_chan2 (bool, optional): Whether to align by non-functional channel. Defaults to False.
        save_path (str, optional): Path to save registered tiffs. Defaults to None.
        settings (dict or list of dicts, optional): Dictionary containing input arguments for suite2p pipeline. Defaults to default_settings().

    Returns:
        tuple: Tuple containing the following:
            refImg (2D array): 2D array of int16, size [Ly x Lx], initial reference image (if not registered).
            rmin (int): Clip frames at rmin.
            rmax (int): Clip frames at rmax.
            meanImg (np.ndarray): Computed Mean Image for functional channel, size [Ly x Lx].
            rigid_offsets (tuple): Tuple of length 3, rigid shifts computed between each frame and reference image. Shifts for each frame in x, y, and z directions.
            nonrigid_offsets (tuple): Tuple of length 3, non-rigid shifts computed between each frame and reference image.
            zest (tuple): Tuple of length 2.
            meanImg_chan2 (np.ndarray): Computed Mean Image for non-functional channel, size [Ly x Lx].
            badframes (np.ndarray): Boolean array of frames that have large outlier shifts that may make registration problematic, size [n_frames, ].
            yrange (list): Valid ranges for registration along y-axis of frames.
            xrange (list): Valid ranges for registration along x-axis of frames.
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
        if bidiphase != 0 and refImg is None:
            frames = bidi.shift(frames, int(settings["bidiphase"])) 
        settings["bidiphase"] = bidiphase
    else:
        bidiphase = 0

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

    reg_outputs = registration_outputs_to_dict(refImg_orig, rmin, rmax, meanImg, (yoff, xoff, corrXY), (yoff1, xoff1, corrXY1), (zest, cmax_all), meanImg_chan2, badframes, yrange, xrange)
    
    # add enhanced mean image
    meanImgE = utils.highpass_mean_image(meanImg.astype("float32"), aspect=aspect)
    reg_outputs["meanImgE"] = meanImgE
    return reg_outputs

def registration_outputs_to_dict(refImg, rmin, rmax, meanImg, rigid_offsets, 
                                 nonrigid_offsets, zest, meanImg_chan2, 
                                 badframes, yrange, xrange):
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
    reg_outputs["badframes"], reg_outputs["yrange"], reg_outputs["xrange"] = badframes, yrange, xrange
    if zest[0] is not None:
        reg_outputs["zpos_registration"] = np.array(zest[0])
        reg_outputs["cmax_registration"] = np.array(zest[1])
    return reg_outputs
