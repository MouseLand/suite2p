"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import warnings
from typing import Tuple

import numpy as np
from numpy import fft
from scipy.fftpack import next_fast_len
import torch
import torch.nn.functional as F

from .utils import spatial_taper, gaussian_fft, kernelD2, mat_upsample, convolve, ref_smooth_fft

def calculate_nblocks(L: int, block_size: int = 128) -> Tuple[int, int]:
    """
    Returns block_size and nblocks from dimension length and desired block size

    Parameters
    ----------
    L: int
    block_size: int

    Returns
    -------
    block_size: int
    nblocks: int
    """
    return (L, 1) if block_size >= L else (block_size,
                                           int(np.ceil(1.5 * L / block_size)))

def make_blocks(Ly, Lx, block_size=(128, 128), lpad=3, subpixel=10):
    """
    Computes overlapping blocks to split FOV into to register separately

    Parameters
    ----------
    Ly: int
        Number of pixels in the vertical dimension
    Lx: int
        Number of pixels in the horizontal dimension
    block_size: int, int
        block size

    Returns
    -------
    yblock: float array
    xblock: float array
    nblocks: int, int
    block_size: int, int
    NRsm: array
    """
    block_size = (int(block_size[0]), int(block_size[1]))
    block_size_y, ny = calculate_nblocks(L=Ly, block_size=block_size[0])
    block_size_x, nx = calculate_nblocks(L=Lx, block_size=block_size[1])
    block_size = (block_size_y, block_size_x)

    # todo: could rounding to int here over-represent some pixels over others?
    ystart = np.linspace(0, Ly - block_size[0], ny).astype("int")
    xstart = np.linspace(0, Lx - block_size[1], nx).astype("int")
    yblock = [
        np.array([ystart[iy], ystart[iy] + block_size[0]])
        for iy in range(ny)
        for _ in range(nx)
    ]
    xblock = [
        np.array([xstart[ix], xstart[ix] + block_size[1]])
        for _ in range(ny)
        for ix in range(nx)
    ]

    NRsm = kernelD2(xs=torch.arange(nx), ys=torch.arange(ny)).T.numpy()
    Kmat, nup = mat_upsample(lpad=lpad, subpixel=subpixel)
    return yblock, xblock, [ny, nx], block_size, NRsm, Kmat, nup


def compute_masks_ref_smooth_fft(refImg0: np.ndarray, maskSlope, smooth_sigma,
                        yblock: np.ndarray, xblock: np.ndarray):
    """
    Computes taper and fft"ed reference image for phasecorr.

    Parameters
    ----------
    refImg0: array
    maskSlope
    smooth_sigma
    yblock: float array
    xblock: float array
    
    Returns
    -------
    maskMul
    maskOffset
    cfRefImg

    """
    nb, Ly, Lx = len(yblock), yblock[0][1] - yblock[0][0], xblock[0][1] - xblock[0][0]
    dims = (nb, Ly, Lx)
    cfRef_dims = dims
    gaussian_filter = gaussian_fft(smooth_sigma, *cfRef_dims[1:])
    cfRefImg1 = torch.zeros(cfRef_dims, dtype=torch.complex64)

    maskMul = spatial_taper(maskSlope, *refImg0.shape)
    maskMul1 = torch.zeros(dims, dtype=torch.float)
    maskMul1[:] = spatial_taper(2 * smooth_sigma, Ly, Lx)
    maskOffset1 = torch.zeros(dims, dtype=torch.float)
    for yind, xind, maskMul1_n, maskOffset1_n, cfRefImg1_n in zip(
            yblock, xblock, maskMul1, maskOffset1, cfRefImg1):
        ix = np.ix_(
            np.arange(yind[0], yind[-1]).astype("int"),
            np.arange(xind[0], xind[-1]).astype("int"))
        refImg = refImg0[ix]

        # mask params
        maskMul1_n *= maskMul[yind[0] : yind[-1], xind[0] : xind[-1]]
        maskOffset1_n[:] = (refImg.float().mean() * (1. - maskMul1_n))

        # gaussian filter
        cfRefImg1_n[:] = ref_smooth_fft(refImg, smooth_sigma)
        
    return maskMul1, maskOffset1, cfRefImg1

def getSNR(cc: np.ndarray, lcorr: int, lpad: int) -> float:
    """
    Compute SNR of phase-correlation.

    Parameters
    ----------
    cc: Ly x Lx
        The frame data to analyze
    lcorr: int
    lpad: int
        border padding width

    Returns
    -------
    snr: float
    """
    cc0 = cc[:, lpad:-lpad, lpad:-lpad].reshape(cc.shape[0], -1)
    # set to 0 all pts +-lpad from ymax,xmax
    cc1 = cc.copy()
    for c1, ymax, xmax in zip(
            cc1,
            *np.unravel_index(cc0.argmax(axis=1), (2 * lcorr + 1, 2 * lcorr + 1))):
        c1[ymax:ymax + 2 * lpad, xmax:xmax + 2 * lpad] = 0
    
    snr = cc0.max(axis=1) / np.maximum(1e-10, cc1.max(axis=(1, 2)))
    return snr

def phasecorr(data: np.ndarray, blocks, maskMul, maskOffset, cfRefImg, snr_thresh,
              maxregshiftNR, subpixel: int = 10, lpad: int = 3):
    """
    Compute phase correlations for each block
    
    Parameters
    ----------
    data : nimg x Ly x Lx
    maskMul: ndarray
        gaussian filter
    maskOffset: ndarray
        mask offset
    cfRefImg
        FFT of reference image
    snr_thresh : float
        signal to noise ratio threshold
    NRsm
    xblock: float array
    yblock: float array
    maxregshiftNR: int
    subpixel: int
    lpad: int
        upsample from a square +/- lpad

    Returns
    -------
    ymax1
    xmax1
    cmax1
    """

    yblock, xblock, _, _, NRsm, Kmat, nup = blocks

    device = data.device
    
    nimg = data.shape[0]
    ly, lx = cfRefImg.shape[-2:]

    # maximum registration shift allowed
    lcorr = int(
        np.minimum(np.round(maxregshiftNR),
                   np.floor(np.minimum(ly, lx) / 2.) - lpad))
    nb = len(yblock)

    # shifts and corrmax
    Y = torch.zeros((nimg, nb, ly, lx), dtype=torch.int16, device=device)
    for n in range(nb):
        yind, xind = yblock[n], xblock[n]
        Y[:, n] = data[:, yind[0]:yind[-1], xind[0]:xind[-1]]
    Y = (Y.float() * maskMul + maskOffset).type(torch.complex64)
    batch = min(64, Y.shape[1])  #16
    for n in np.arange(0, nb, batch):
        nend = min(Y.shape[1], n + batch)
        Y[:, n:nend] = convolve(mov=Y[:, n:nend], img=cfRefImg[n:nend])
    
    # calculate ccsm
    lhalf = lcorr + lpad
    cc0 = torch.cat((torch.cat((Y[..., -lhalf:, -lhalf:], Y[..., -lhalf:, :lhalf + 1]), axis=-1),   
                    torch.cat((Y[..., :lhalf + 1, -lhalf:], Y[..., :lhalf + 1, :lhalf + 1]), axis=-1)), axis=-2)
    cc0 = torch.real(cc0)
    cc0 = cc0.permute(1, 0, 2, 3)
    cc0 = cc0.reshape(cc0.shape[0], -1)
    cc0 = cc0.cpu().numpy()

    del Y
    if device.type == "cuda":
        torch.cuda.empty_cache()    
        torch.cuda.synchronize()
    
    cc2 = [cc0, NRsm @ cc0, NRsm @ NRsm @ cc0]
    cc2 = [
        c2.reshape(nb, nimg, 2 * lcorr + 2 * lpad + 1, 2 * lcorr + 2 * lpad + 1)
        for c2 in cc2
    ]
    ccsm = cc2[0]
    
    for n in range(nb):
        snr = np.ones(nimg, dtype="float32")
        for j, c2 in enumerate(cc2):
            ism = snr < snr_thresh
            if ism.sum() == 0:
                break
            cc = c2[n, ism, :, :]
            if j > 0:
                ccsm[n, ism, :, :] = cc#.cpu().numpy()
            snr[ism] = getSNR(cc, lcorr, lpad)

    # calculate ymax1, xmax1, cmax1
    mdpt = nup // 2
    ymax1 = np.empty((nimg, nb), "float32")
    cmax1 = np.empty((nimg, nb), "float32")
    xmax1 = np.empty((nimg, nb), "float32")
    ymax = np.empty((nb,), "int32")
    xmax = np.empty((nb,), "int32")

    imax = ccsm[..., lpad:-lpad, lpad:-lpad].reshape(nb, nimg, -1).argmax(axis=-1)
    ymax, xmax = np.unravel_index(imax, (2 * lcorr + 1, 2 * lcorr + 1))
    ccmat = np.empty((nb, nimg, 2 * lpad + 1, 2 * lpad + 1), "float32")
    for t in range(nimg):
        for n in range(nb):
            ym, xm = ymax[n, t], xmax[n, t]
            ccmat[n, t] = ccsm[n, t, ym:ym + 2 * lpad + 1, xm:xm + 2 * lpad + 1]
    ccmat = torch.from_numpy(ccmat.reshape(nb * nimg, -1)).to(device)
    ccb = (ccmat @ Kmat.to(device)).reshape(nb, nimg, -1)
    cmax1, imax1 = ccb.max(axis=-1)
    ymax1, xmax1 = torch.div(imax1, nup, rounding_mode="floor"), imax1 % nup
    ymax1 = (ymax1 - mdpt) / subpixel + torch.from_numpy(ymax).to(device) - lcorr
    xmax1 = (xmax1 - mdpt) / subpixel + torch.from_numpy(xmax).to(device) - lcorr
    
    return ymax1.T.float(), xmax1.T.float(), cmax1.T, ccsm, ccb

def transform_data(data, nblocks, xblock, yblock, ymax1, xmax1):
    """
    Piecewise affine transformation of data using block shifts ymax1, xmax1
    
    Parameters
    ----------

    data : nimg x Ly x Lx
    nblocks: (int, int)
    xblock: float array
    yblock: float array
    ymax1 : nimg x nblocks
        y shifts of blocks
    xmax1 : nimg x nblocks
        y shifts of blocks
    bilinear: bool (optional, default=True)
        do bilinear interpolation, if False do nearest neighbor

    Returns
    -----------
    Y : float32, nimg x Ly x Lx
        shifted data
    """
    _, Ly, Lx = data.shape
    #device = torch.device("cuda")
    #data = torch.from_numpy(data).to(device).float()
    device = data.device
    ymax1 = ymax1.reshape(-1, *nblocks)
    xmax1 = xmax1.reshape(-1, *nblocks)
    mshy, mshx = torch.meshgrid(torch.arange(Ly, dtype=torch.float, device=device),
                         torch.arange(Lx, dtype=torch.float, device=device), indexing="ij")
    yb = np.array(yblock[::nblocks[1]]).mean(axis=1).astype("int")
    xb = np.array(xblock[:nblocks[1]]).mean(axis=1).astype("int")
    Lyc, Lxc = int(yb.max() - yb.min()), int(xb.max() - xb.min())
    yxup = F.interpolate(torch.stack((ymax1, xmax1), dim=1), 
                         size=(Lyc, Lxc), mode="bilinear", align_corners=True)
    yxup = F.pad(yxup, (int(xb.min()), Lx - int(xb.max()), 
                        int(yb.min()), Ly - int(yb.max())), mode="replicate")
    yxup[:,0] += mshy
    yxup[:,1] += mshx
    yxup /= torch.Tensor([Ly-1, Lx-1]).to(device).unsqueeze(-1).unsqueeze(-1)
    yxup *= 2 
    yxup -= 1
    yxup = yxup.permute(0, 2, 3, 1)
    if device.type == "mps":
        # Manually pad the input tensor with the border values
        data_padded = F.pad(data.float().unsqueeze(1), (1, 1, 1, 1), mode="replicate")
        height, width = data.shape[-2:]  # Get the height and width of the original data tensor
        # Adjust the grid to account for the padding
        adjusted_yxup = yxup + torch.tensor([[[[1 / width, 1 / height]]]]).to(yxup.device)  # Adjust grid
        # Perform grid sampling on the padded tensor
        fr_shift = F.grid_sample(
            data_padded,
            adjusted_yxup[:, :, :, [1, 0]],
            mode="bilinear",
            padding_mode="zeros",  # Default or any supported mode
            align_corners=True
        )
    else:
        fr_shift = F.grid_sample(data.float().unsqueeze(1), yxup[:,:,:,[1,0]], 
                             mode="bilinear", padding_mode="border", align_corners=True)
    return fr_shift.squeeze().short()#.cpu().numpy()