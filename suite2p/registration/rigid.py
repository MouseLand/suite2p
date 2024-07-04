"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from typing import Tuple

import numpy as np

from .utils import convolve, complex_fft2, spatial_taper, gaussian_fft, temporal_smooth, ref_smooth_fft

import torch

def compute_masks(refImg, maskSlope) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns maskMul and maskOffset from an image and slope parameter

    Parameters
    ----------
    refImg: Ly x Lx
        The image
    maskSlope

    Returns
    -------
    maskMul: float arrray
    maskOffset: float array
    """
    Ly, Lx = refImg.shape
    maskMul = spatial_taper(maskSlope, Ly, Lx)
    maskOffset = refImg.float().mean() * (1. - maskMul)
    return maskMul.float(), maskOffset.float()

def compute_masks_ref_smooth_fft(refImg, maskSlope, smooth_sigma) -> Tuple[np.ndarray, np.ndarray]:
    maskMul, maskOffset = compute_masks(refImg, maskSlope)
    cfRefImg = ref_smooth_fft(refImg=refImg, smooth_sigma=smooth_sigma)
    return maskMul, maskOffset, cfRefImg

def phasecorr(frames, cfRefImg, maskMul, maskOffset, maxregshift, smooth_sigma_time):
    device = frames.device
    data = (frames.float() * maskMul + maskOffset).type(torch.complex64)
    min_dim = np.minimum(*data.shape[1:])  # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))

    data = convolve(data, cfRefImg)
    cc = torch.cat((torch.cat((data[:, -lcorr:, -lcorr:], data[:, -lcorr:, :lcorr + 1]), axis=2),   
                    torch.cat((data[:, :lcorr + 1, -lcorr:], data[:, :lcorr + 1, :lcorr + 1]), axis=2)), axis=1)
    cc = torch.real(cc)
    
    cc = temporal_smooth(cc, smooth_sigma_time) if smooth_sigma_time > 0 else cc

    imax = torch.stack([torch.argmax(cc[t]) for t in range(data.shape[0])], dim=0)
    ymax, xmax = torch.div(imax, 2 * lcorr + 1, rounding_mode="floor"), imax % (2 * lcorr + 1)
    cmax = cc[torch.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr
    
    del data, cc
    if device.type == "cuda":
       torch.cuda.empty_cache()
    return ymax, xmax, cmax
