"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np

from .utils import convolve, complex_fft2, spatial_taper, temporal_smooth, ref_smooth_fft

import torch

def compute_masks_ref_smooth_fft(refImg, maskSlope, smooth_sigma):
    """
    Compute multiplicative and additive masks used for spatial tapering in rigid registration, 
    and smooth with Gaussian.
    
    Parameters
    ----------
    refImg : torch.Tensor
        2D reference image of shape (Ly, Lx).
    maskSlope : float
        Scalar parameter controlling the slope of the sigmoid of the spatial taper. Higher
        values increase tapered region size.
    smooth_sigma : float
        Standard deviation (in pixels) of the Gaussian smoothing applied to each
        block. Smoothing is performed in the frequency domain (via ref_smooth_fft). Typical values are >= 0. A value of 0 should behave as no
        smoothing (identity).

    Returns
    -------
    maskMul : torch.Tensor
        Floating-point multiplicative mask of shape (Ly, Lx), intended to smoothly
        reduce the influence of border pixels during registration.
    maskOffset : torch.Tensor
        Floating-point additive offset mask of shape (Ly, Lx), computed as
        mean(refImg) * (1.0 - maskMul), setting the border pixels to the mean.
    """
    Ly, Lx = refImg.shape
    maskMul = spatial_taper(maskSlope, Ly, Lx)
    maskOffset = refImg.float().mean() * (1. - maskMul)
    cfRefImg = ref_smooth_fft(refImg=refImg, smooth_sigma=smooth_sigma)
    return maskMul, maskOffset, cfRefImg

def phasecorr(frames, cfRefImg, maskMul, maskOffset, maxregshift, smooth_sigma_time, 
              return_cc=False):
    """
    Compute rigid-registration shifts using phase correlation with an optional temporal smoothing.
    This function performs a Fourier-domain phase-correlation based registration between each frame in
    `frames` and a provided (complex) reference image `cfRefImg`. It computes the integer pixel shifts
    (y, x) that maximize the phase-correlation within a limited search window, defined by `maxregshift`.

    Parameters
    ----------
    frames : torch.Tensor
        Input image sequence, expected shape (N, Ly, Lx) where N is the number of frames.
        The tensor may be on CPU or CUDA; it is converted to float and then to complex for the
        Fourier-domain operations performed by the helper `convolve`.
    cfRefImg : torch.Tensor
        Complex-valued reference of shape (Ly, Lx) in the Fourier domain used to compute 
        cross-correlation with each frame
    maskMul : torch.Tensor
        Multiplicative mask applied to `frames` before correlation. Broadcasted over frames.
    maskOffset : torch.Tensor
        Additive offset applied after `maskMul`. Broadcasted over frames.
    maxregshift : float
        Maximum allowed registration shift expressed as a fraction of the smaller spatial image
        dimension. The actual integer search half-window `lcorr` is computed as
        min(round(maxregshift * min(Ly, Lx)), floor(min(Ly, Lx) / 2)).
    smooth_sigma_time : float
        If > 0, applies temporal smoothing (via helper `temporal_smooth`) to the phase-correlation maps
        along the time axis with this sigma before finding maxima. If <= 0, no temporal smoothing is used.
    return_cc : bool, optional (default False)
        If True, return the computed local phase-correlation maps as a NumPy array on CPU;
        otherwise the correlation maps are freed to save memory and None is returned in their place.
    
    Returns
    -------
    ymax : torch.LongTensor
        1-D integer tensor of length N with the y (row) shift for each frame that maximizes the
        phase-correlation. 
    xmax : torch.LongTensor
        1-D integer tensor of length N with the x (column) shift for each frame that maximizes the
        phase-correlation. 
    cmax : torch.Tensor
        1-D tensor of length N containing the maximum phase-correlation value found for each frame.
    cc : numpy.ndarray or None
        If `return_cc` is True, a NumPy array of shape (N, 2*lcorr+1, 2*lcorr+1) with the real-valued
        local phase-correlation maps (dtype float32) is returned. If `return_cc` is False, cc is None.
    """

    device = frames.device
    data = (frames.float() * maskMul + maskOffset).type(torch.complex64)
    min_dim = min(data.shape[1], data.shape[2])  # maximum registration shift allowed
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
    
    del data
    if return_cc: 
        cc = cc.cpu().numpy()
    else:
        del cc
        cc = None
    if device.type == "cuda":
       torch.cuda.empty_cache()
    
    return ymax, xmax, cmax, cc
