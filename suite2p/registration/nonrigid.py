"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from numpy import fft
from scipy.fftpack import next_fast_len
import torch
import torch.nn.functional as F

from .utils import spatial_taper, kernelD2, mat_upsample, convolve, ref_smooth_fft

def calculate_nblocks(L: int, block_size: int):
    """
    Returns block_size and nblocks from dimension length and desired block size

    Parameters
    ----------
    L: int
        Number of pixels in one dimension in image.
    block_size: int
        Block size in pixels.

    Returns
    -------
    block_size: int
        min(L, block_size).
    nblocks: int
        Number of blocks to make along dimension.
    """
    return (L, 1) if block_size >= L else (block_size,
                                           int(np.ceil(1.5 * L / block_size)))

def make_blocks(Ly, Lx, block_size, lpad=3, subpixel=10):
    """
    Compute overlapping registration blocks covering a 2D field of view.
    This function splits a full-frame image of size (Ly, Lx) into an array of
    overlapping rectangular blocks to be processed independently for nonrigid
    registration. Block start positions are computed so that blocks tile the image
    with (approximately) equal spacing and specified overlap determined by the
    requested block_size. The function also computes a spatial smoothing matrix
    (NRsm) over the block grid and an upsampling convolution matrix (Kmat) used
    for subpixel shift estimation.

    Parameters
    ----------
    Ly : int
        Number of pixels in the vertical dimension (image height).
    Lx : int
        Number of pixels in the horizontal dimension (image width).
    block_size : tuple[int, int]
        Block size in pixels as (block_height, block_width).
    lpad : int, optional
        Padding in pixels used when constructing the upsampling matrix.
        Passed to mat_upsample(...). Default is 3.
    subpixel : int, optional
        Subpixel upsampling factor. Passed to mat_upsample(...). Default is 10.
    
    Returns
    -------
    yblock : list[numpy.ndarray]
        List of length (ny * nx) giving the vertical (row) slice for each block.
        Each element is a 1D integer numpy array [y_start, y_end] specifying the
        inclusive start (y_start) and exclusive end (y_end) indices of the block
        along the vertical axis. Blocks are ordered row-major by block-grid row
        (iy) then column (ix): block_idx = iy * nx + ix.
    xblock : list[numpy.ndarray]
        List of length (ny * nx) giving the horizontal (column) slice for each
        block. Each element is a 1D integer numpy array [x_start, x_end]
        specifying the inclusive start and exclusive end indices along the
        horizontal axis. Ordering matches yblock (row-major block-grid order).
    nblocks : list[int, int]
        Two-element list [ny, nx] with the number of blocks in the vertical and
        horizontal directions respectively (ny = number of block rows,
        nx = number of block columns).
    block_size : tuple[int, int]
        Effective block size used, min of input block size and frame size.
    NRsm : numpy.ndarray
        2D smoothing kernel matrix defined on the block grid. Shape is (ny, nx).
        This matrix (derived from kernelD2 over block grid coordinates) is
        used to smooth or regularize blockwise motion estimates spatially.
    Kmat : numpy.ndarray
        Upsampling kriging interpolation matrix returned by mat_upsample(lpad, subpixel).
        This matrix is used for subpixel shift estimation within +/- lpad pixels.
    nup : int
        Kmat.shape[-1].
   
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


def compute_masks_ref_smooth_fft(refImg0, maskSlope, smooth_sigma,
                                 yblock, xblock):
    """
    Compute per-block taper masks, offsets, and FFT-smoothed reference images for
    nonrigid phase-correlation registration.
    This function extracts blocks from a full 2D reference image, applies a
    spatial taper (window) to each block, computes a per-block constant offset
    to compensate for masked/background regions, and computes a Gaussian-smoothed
    version of each block in the frequency domain (complex FFT) for use in
    phase-correlation based registration.

    Parameters
    ----------
    refImg0 : torch.Tensor
        2D reference image array of shape (Ly_full, Lx_full). Expected numeric
        image type (e.g. uint16, float32 or torch tensor). The function will
        extract sub-blocks using the indices supplied in yblock and xblock.
    maskSlope : float
        Scalar parameter controlling the slope of the sigmoid of the spatial taper. 
        Higher values increase tapered region size.
    smooth_sigma : float
        Standard deviation (in pixels) of the Gaussian smoothing applied to each
        block. Smoothing is performed in the frequency domain (via ref_smooth_fft). 
        Typical values are >= 0. A value of 0 should behave as no
        smoothing (identity).
    yblock : list[numpy.ndarray]
        List of length (ny * nx) giving the vertical (row) slice for each block.
        Each element is a 1D integer numpy array [y_start, y_end] specifying the
        inclusive start (y_start) and exclusive end (y_end) indices of the block
        along the vertical axis. Blocks are ordered row-major by block-grid row
        (iy) then column (ix): block_idx = iy * nx + ix.
    xblock : list[numpy.ndarray]
        List of length (ny * nx) giving the horizontal (column) slice for each
        block. Each element is a 1D integer numpy array [x_start, x_end]
        specifying the inclusive start and exclusive end indices along the
        horizontal axis. Ordering matches yblock (row-major block-grid order).

    Returns
    -------
    maskMul_block : torch.Tensor
        Float32 tensor of shape (nb, Ly, Lx). Per-block multiplicative taper
        masks obtained by multiplying a local block taper.
    maskOffset_block : torch.Tensor
        Float32 tensor of shape (nb, Ly, Lx). Per-block additive offset fields
        computed as block_mean * (1 - maskMul_block) so that masked regions are
        filled with the local block mean scaled by the complement of the taper.
    cfRefImg_block : torch.Tensor (complex64)
        Complex32 tensor of shape (nb, Ly, Lx). Frequency-domain (FFT) representation
        of the Gaussian-smoothed reference blocks (output of ref_smooth_fft). These
        are intended for use in phase-correlation registration.
    
    """
    nb, Ly, Lx = len(yblock), yblock[0][1] - yblock[0][0], xblock[0][1] - xblock[0][0]
    dims = (nb, Ly, Lx)
    cfRef_dims = dims
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

def getSNR(cc, lcorr, lpad):
    """
    Compute the signal-to-noise ratio (SNR) of phase-correlation maps.
    This function estimates the SNR for one or more phase-correlation maps by
    (1) locating the peak value within the central search region of each map,
    (2) zeroing a square neighborhood around that peak in a copy of the full map
        to exclude the main peak energy, and
    (3) taking the ratio of the peak value to the maximum remaining value in the
        map (with a small epsilon to avoid division by zero).

    Parameters
    ----------
    cc : torch.Tensor
        Array of phase-correlation maps with shape (n_maps, H, W). Each spatial
        dimension is expected to equal (2 * lcorr + 1) + 2 * lpad, i.e. the
        central searchable region of size (2*lcorr+1) is padded on all sides by
        lpad pixels. The first axis indexes independent maps (e.g. frames).
    lcorr : int
        Half-size of the central correlation search window. The central region
        searched for the peak is of size (2 * lcorr + 1) x (2 * lcorr + 1).
    lpad : int
        Padding width (in pixels) around the central search region. When masking
        the peak, a square of side length 2 * lpad is zeroed around the detected
        peak location in the copy of the map to measure the maximum background
        response.

    Returns
    -------
    snr : ndarray
        Array of SNR values, one per input map, with shape (n_maps,). Each entry
        is the peak value found inside the central region divided by the maximum
        value remaining in the map after masking the peak neighborhood. Values
        are finite due to a small numerical epsilon (1e-10) used in the
        denominator.
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

def phasecorr(data, blocks, maskMul, maskOffset, cfRefImg, snr_thresh,
              maxregshiftNR, subpixel = 10, lpad = 3):
    """
    Compute per-block shifts using phase correlation.
    This function performs a Fourier-domain phase-correlation based registration between each frame and each block in
    `data` and a provided (complex) reference image `cfRefImg`, in blocks. It computes the integer pixel shifts
    (y, x) that maximize the phase-correlation within a limited search window, defined by `maxregshiftNR`.
    The phase-correlations are smoothed across blocks, and these smoothed phase-correlations are used if the 
    block SNR is below `snr_thresh`. A small neighborhood around each peak is then upsampled via Kriging interpolation 
    using the provided Kmat kernel, and the peak of the upsampled phase-correlation is used to obtain subpixel-level shifts.

    Parameters
    ----------
    data : torch.Tensor
        Input image sequence, expected shape (nimg, Ly, Lx) where nimg is the number of frames.
        The tensor may be on CPU or CUDA; it is converted to float and then to complex for the
        Fourier-domain operations performed by the helper `convolve`.
    blocks : tuple
        Tuple of block descriptors produced by the caller, unpacked in this function as:
            (yblock, xblock, _, _, NRsm, Kmat, nup)
    maskMul : torch.Tensor
        Multiplicative mask applied to `data` per-block before correlation. Broadcasted over frames.
    maskOffset : torch.Tensor
        Additive offset applied after `maskMul` per-block. Broadcasted over frames.
    cfRefImg : torch.Tensor
        Complex-valued reference of shape (Ly, Lx) in the Fourier domain used to compute 
        cross-correlation with each frame.
    snr_thresh : float
        SNR threshold used to decide whether to replace a block's raw correlation map with
        progressively more-smoothed versions computed via NRsm. Lower values make smoothing
        less likely.
    maxregshiftNR : int
        Maximum allowed registration shift (interpreted as pixels and rounded).
    lpad : int, optional
        Padding in pixels used when constructing the upsampling matrix. Default is 3.
    subpixel : int, optional
        Subpixel upsampling factor. Default is 10.
    
    Returns
    -------
    ymax1 : torch.Tensor
        Tensor of shape (nblocks, N) with the y (row) shift for each frame and block that maximizes the
        phase-correlation. 
    xmax1 : torch.LongTensor
        Tensor of shape (nblocks, N) with the x (row) shift for each frame and block that maximizes the
        phase-correlation. 
    cmax1 : torch.Tensor
        Tensor of shape (nblocks, N) containing the maximum phase-correlation value found for each frame and block.
    ccsm : numpy.ndarray
        Phase-correlation maps (potentially smoothed) used for peak selection for each frame and block. Shape:
            (n_blocks, N, 2*lcorr + 2*lpad + 1, 2*lcorr + 2*lpad + 1)
    ccb : torch.Tensor
        Tensor of shape (n_blocks, N, y+x pixels) containing upsampled phase-correlation values for each frame and block.

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
    Apply bilinear interpolation to transform image data using block-wise shifts.
    This function performs non-rigid image registration by interpolating block-wise 
    shift values across the image and applying the resulting displacement field via 
    the `grid_sample` function. It handles both standard GPU and Apple Silicon (MPS) devices.

    data : torch.Tensor
        Input image data of shape (nimg, Ly, Lx) where nimg is the number of images,
        Ly is the height, and Lx is the width.
    nblocks : tuple of int
        Number of blocks in (y, x) dimensions for the registration grid.
    xblock : np.ndarray
        X-coordinates of block boundaries of length nblocks[0]*nblocks[1].
    yblock : np.ndarray
        Y-coordinates of block boundaries of length nblocks[0]*nblocks[1].
    ymax1 : torch.Tensor
        Tensor of shape (nblocks, N) with the y (row) shift for each frame and block that maximizes the
        phase-correlation. 
    xmax1 : torch.Tensor
        Tensor of shape (nblocks, N) with the x (row) shift for each frame and block that maximizes the
        phase-correlation. 

    Returns
    -------
    fr_shift : torch.Tensor
        Shifted image data of shape (nimg, Ly, Lx) with dtype int16 (short).
        The input images are warped according to the interpolated displacement field.
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
