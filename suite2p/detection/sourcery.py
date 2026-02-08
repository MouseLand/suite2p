"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import math
import time

from tqdm import trange
import numpy as np
from scipy.ndimage import filters, gaussian_filter
import logging 
logger = logging.getLogger(__name__)

from .utils import circleMask


def getSVDdata(mov: np.ndarray, diameter):
    """
    Compute SVD spatial components from temporally binned movie data.

    Smooths each frame with a 2D Gaussian filter, computes the temporal covariance
    matrix, and returns the top SVD spatial components reshaped as images.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx).
    diameter : float or list of float
        Cell diameter used to set the Gaussian smoothing sigma (sigma = diameter / 10).

    Returns
    -------
    U : numpy.ndarray
        Spatial SVD components of shape (Ly, Lx, nsvd).
    u : numpy.ndarray
        Temporal SVD components of shape (nbins, nsvd).
    """
    nbins, Lyc, Lxc = np.shape(mov)

    sig = diameter / 10.  # PICK UP
    for j in range(nbins):
        mov[j, :, :] = gaussian_filter(mov[j, :, :], sig)

    # compute noise variance across frames
    mov = np.reshape(mov, (-1, Lyc * Lxc))
    # compute covariance of binned frames
    cov = mov @ mov.transpose() / mov.shape[1]
    cov = cov.astype("float32")

    nsvd_for_roi = min(nbins, int(cov.shape[0] / 2))
    u, s, v = np.linalg.svd(cov)

    u = u[:, :nsvd_for_roi]
    U = u.transpose() @ mov
    U = np.reshape(U, (-1, Lyc, Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()
    return U, u


def getSVDproj(mov: np.ndarray, u, diameter, smooth_masks=False):
    """
    Project temporally binned movie data onto existing SVD temporal components.

    Optionally smooths frames before projection. Used during refinement to
    recompute spatial components from updated movie data.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx).
    u : numpy.ndarray
        Temporal SVD components of shape (nbins, nsvd).
    diameter : float or list of float
        Cell diameter used to set the Gaussian smoothing sigma.
    smooth_masks : bool, optional (default False)
        If True, smooth each frame before projection.

    Returns
    -------
    U : numpy.ndarray
        Spatial SVD projections of shape (Ly, Lx, nsvd).
    """
    nbins, Lyc, Lxc = np.shape(mov)
    if smooth_masks:
        sig = np.maximum([.5, .5], diameter / 20.)
        for j in range(nbins):
            mov[j, :, :] = gaussian_filter(mov[j, :, :], sig)
    
    mov = np.reshape(mov, (-1, Lyc * Lxc))

    U = u.transpose() @ mov
    U = U.transpose().copy().reshape((Lyc, Lxc, -1))
    
    return U


def getStU(diameter, U):
    """
    Compute neuropil basis functions and their covariances with the SVD spatial components.

    Parameters
    ----------
    diameter : float or list of float
        Cell diameter used for neuropil basis construction.
    U : numpy.ndarray
        Spatial SVD components of shape (Ly, Lx, nsvd).

    Returns
    -------
    S : numpy.ndarray
        Neuropil basis functions of shape (Ly, Lx, nbasis).
    StU : numpy.ndarray
        Cross-covariance of neuropil basis with SVD components, shape (nbasis, nsvd).
    StS : numpy.ndarray
        Auto-covariance of neuropil basis, shape (nbasis, nbasis).
    """
    Lyc, Lxc, nbins = np.shape(U)
    S = create_neuropil_basis(diameter, Lyc, Lxc)
    # compute covariance of neuropil masks with spatial masks
    StU = S.reshape((Lyc * Lxc, -1)).transpose() @ U.reshape((Lyc * Lxc, -1))
    StS = S.reshape((Lyc * Lxc, -1)).transpose() @ S.reshape((Lyc * Lxc, -1))
    #U = np.reshape(U, (-1,Lyc,Lxc))
    return S, StU, StS


def create_neuropil_basis(diameter, Ly, Lx):
    """
    Compute Fourier-based neuropil basis functions for the image.

    Creates a set of 2D Fourier basis functions tiled across the image,
    used to model spatially varying neuropil contamination.

    Parameters
    ----------
    diameter : float or list of float
        Cell diameter used to determine the tiling density.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.

    Returns
    -------
    S : numpy.ndarray
        Normalized neuropil basis functions of shape (Ly, Lx, nbasis).
    """

    ratio_neuropil = 6.
    tile_factor = 1.
    
    ntilesY = 1 + 2 * int(
        np.ceil(tile_factor * Ly / (ratio_neuropil * diameter[0] / 2)) / 2)
    ntilesX = 1 + 2 * int(
        np.ceil(tile_factor * Lx / (ratio_neuropil * diameter[1] / 2)) / 2)
    ntilesY = np.maximum(2, ntilesY)
    ntilesX = np.maximum(2, ntilesX)
    yc = np.linspace(1, Ly, ntilesY)
    xc = np.linspace(1, Lx, ntilesX)
    ys = np.arange(0, Ly)
    xs = np.arange(0, Lx)

    Kx = np.ones((Lx, ntilesX), "float32")
    Ky = np.ones((Ly, ntilesY), "float32")
    if 1:
        # basis functions are fourier modes
        for k in range(int((ntilesX - 1) / 2)):
            Kx[:, 2 * k + 1] = np.sin(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
            Kx[:, 2 * k + 2] = np.cos(2 * math.pi * (xs + 0.5) * (1 + k) / Lx)
        for k in range(int((ntilesY - 1) / 2)):
            Ky[:, 2 * k + 1] = np.sin(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
            Ky[:, 2 * k + 2] = np.cos(2 * math.pi * (ys + 0.5) * (1 + k) / Ly)
    else:
        for k in range(ntilesX):
            Kx[:, k] = np.cos(math.pi * (xs + 0.5) * k / Lx)
        for k in range(ntilesY):
            Ky[:, k] = np.cos(math.pi * (ys + 0.5) * k / Ly)

    S = np.zeros((ntilesY, ntilesX, Ly, Lx), np.float32)
    for kx in range(ntilesX):
        for ky in range(ntilesY):
            S[ky, kx, :, :] = np.outer(Ky[:, ky], Kx[:, kx])

    S = np.reshape(S, (ntilesY * ntilesX, Ly * Lx))
    S = S / np.reshape(np.sum(S**2, axis=-1)**0.5, (-1, 1))
    S = np.transpose(S, (1, 0)).copy()
    S = np.reshape(S, (Ly, Lx, -1))
    return S


def morphOpen(V, footprint):
    """
    Compute the morphological opening of a correlation map.

    Applies a minimum filter followed by a maximum filter (negated minimum
    of negation) using the given footprint to remove small bright features.

    Parameters
    ----------
    V : numpy.ndarray
        2D correlation map of shape (Ly, Lx).
    footprint : numpy.ndarray
        2D boolean footprint for the morphological operation.

    Returns
    -------
    vrem : numpy.ndarray
        Morphologically opened image of shape (Ly, Lx).
    """
    vrem = filters.minimum_filter(V, footprint=footprint)
    vrem = -filters.minimum_filter(-vrem, footprint=footprint)
    return vrem


def localMax(V, footprint, thres):
    """
    Find local maxima of a correlation map above a threshold.

    Uses a maximum filter with the given footprint to identify pixels that
    are local maxima and exceed the threshold.

    Parameters
    ----------
    V : numpy.ndarray
        2D correlation map of shape (Ly, Lx).
    footprint : numpy.ndarray
        2D boolean footprint for the maximum filter (usually circular).
    thres : float
        Minimum value for a local maximum to be included.

    Returns
    -------
    i : numpy.ndarray
        Y-indices of local maxima, dtype int32.
    j : numpy.ndarray
        X-indices of local maxima, dtype int32.
    """
    maxV = filters.maximum_filter(V, footprint=footprint, mode="reflect")
    imax = V > np.maximum(thres, maxV - 1e-10)
    i, j = imax.nonzero()
    i = i.astype(np.int32)
    j = j.astype(np.int32)
    return i, j




def getVmap(Ucell, sig):
    """
    Compute the variance ratio map from SVD spatial components.

    Smooths the spatial components and computes the ratio of smoothed
    variance to total variance, producing a map that highlights regions
    with locally correlated activity.

    Parameters
    ----------
    Ucell : numpy.ndarray
        Residual SVD spatial components of shape (Ly, Lx, nsvd).
    sig : numpy.ndarray or list of float
        Gaussian smoothing sigma [sy, sx] in pixels.

    Returns
    -------
    log_variances : numpy.ndarray
        Variance ratio map of shape (Ly, Lx), dtype float64.
    us : numpy.ndarray
        Smoothed spatial components of shape (Ly, Lx, nsvd).
    """
    us = gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode="wrap")
    # compute log variance at each location
    log_variances = (us**2).mean(axis=-1) / gaussian_filter(
        (Ucell**2).mean(axis=-1), sig, mode="wrap")
    return log_variances.astype("float64"), us





def get_connected(Ly, Lx, stat):
    """
    Extract the connected component of an ROI starting from its brightest pixel.

    Grows outward from the pixel with maximum weight, keeping only pixels
    that are contiguous and have non-zero weight.

    Parameters
    ----------
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    stat : dict
        ROI statistics dictionary containing "ypix", "xpix", and "lam".
        Modified in-place.

    Returns
    -------
    stat : dict
        Updated ROI dictionary with connected pixels only.
    """
    ypix, xpix, lam = stat["ypix"], stat["xpix"], stat["lam"]
    i0 = lam.argmax()
    mask = np.zeros((Ly, Lx))
    mask[ypix, xpix] = lam
    ypix, xpix = ypix[i0], xpix[i0]
    nsel = 1
    while 1:
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx)
        ix = mask[ypix, xpix] > 1e-10
        ypix, xpix = ypix[ix], xpix[ix]
        if len(ypix) <= nsel:
            break
        nsel = len(ypix)
    lam = mask[ypix, xpix]
    stat["ypix"], stat["xpix"], stat["lam"] = ypix, xpix, lam
    return stat


def connected_region(stat, connected, Ly, Lx):
    """
    Optionally restrict each ROI to its largest connected component.

    Parameters
    ----------
    stat : list of dict
        List of ROI statistics dictionaries, each containing "ypix", "xpix",
        and "lam".
    connected : bool
        If True, apply connected component extraction to each ROI.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.

    Returns
    -------
    stat : list of dict
        Updated list of ROI dictionaries.
    """
    if connected:
        for j in range(len(stat)):
            stat[j] = get_connected(Ly, Lx, stat[j])
    return stat


def extendROI(ypix, xpix, Ly, Lx, niter=1):
    """
    Expand an ROI by one pixel in each cardinal direction.

    Adds the 4-connected neighbors of the current pixel set and removes
    any that fall outside the image boundaries. Repeated for `niter`
    iterations.

    Parameters
    ----------
    ypix : numpy.ndarray
        Y-coordinates of the current ROI pixels.
    xpix : numpy.ndarray
        X-coordinates of the current ROI pixels.
    Ly : int
        Height of the image in pixels.
    Lx : int
        Width of the image in pixels.
    niter : int, optional (default 1)
        Number of expansion iterations.

    Returns
    -------
    ypix : numpy.ndarray
        Expanded Y-coordinates.
    xpix : numpy.ndarray
        Expanded X-coordinates.
    """
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix - 1, ypix + 1), (xpix, xpix + 1, xpix - 1, xpix,
                                                       xpix))
        yx = np.array(yx)
        yx = yx.reshape((2, -1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0] >= 0, yu[0] < Ly, yu[1] >= 0, yu[1] < Lx), axis=0)
        ypix, xpix = yu[:, ix]
    return ypix, xpix


def iter_extend(ypix, xpix, Ucell, code, refine=-1, change_codes=False):
    """
    Iteratively extend an ROI by projecting onto the SVD code vector.

    Starting from seed pixels, repeatedly expands the region and keeps
    pixels whose projection onto the code vector exceeds a threshold.
    Stops when no new pixels are added or the region exceeds 10000 pixels.

    Parameters
    ----------
    ypix : numpy.ndarray or int
        Initial Y-coordinates of the ROI seed.
    xpix : numpy.ndarray or int
        Initial X-coordinates of the ROI seed.
    Ucell : numpy.ndarray
        Residual SVD spatial components of shape (Ly, Lx, nsvd).
    code : numpy.ndarray
        SVD code vector for this ROI, shape (nsvd,).
    refine : int, optional (default -1)
        Refinement stage indicator. Negative means initial detection.
    change_codes : bool, optional (default False)
        If True and refine < 0, update the code vector during extension.

    Returns
    -------
    ypix : numpy.ndarray
        Y-coordinates of the extended ROI.
    xpix : numpy.ndarray
        X-coordinates of the extended ROI.
    lam : numpy.ndarray
        Normalized pixel weights (projections onto code vector).
    ix : numpy.ndarray
        Boolean mask of pixels kept in the final iteration.
    code : numpy.ndarray
        Updated code vector, shape (nsvd,).
    """
    Lyc, Lxc, nsvd = Ucell.shape
    npix = 0
    iter = 0
    while npix < 10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)
        usub = Ucell[ypix, xpix, :]
        lam = usub @ np.expand_dims(code, axis=1)
        lam = np.squeeze(lam, axis=1)
        # ix = lam>max(0, np.mean(lam)/3)
        ix = lam > max(0, lam.max() / 5.0)
        if ix.sum() == 0:
            break
        ypix, xpix, lam = ypix[ix], xpix[ix], lam[ix]
        lam = lam / np.sum(lam**2 + 1e-10)**.5
        if refine < 0 and change_codes:
            code = lam @ usub[ix, :]
        if iter == 0:
            sgn = 1.
            #sgn = np.sign(ix.sum()-npix)
        if np.sign(sgn * (ix.sum() - npix)) <= 0:
            break
        else:
            npix = ypix.size
        iter += 1
    return ypix, xpix, lam, ix, code


def sourcery(mov: np.ndarray, sdmov, diameter, threshold_scaling=1.0,
             connected=True, max_iterations=20, smooth_masks=False):
    """
    Detect ROIs using the Sourcery algorithm (SVD-based iterative detection).

    Computes SVD components of the movie, builds neuropil basis functions 
    and subtracts them, and iteratively detects ROIs by finding local 
    maxima in the variance ratio map and extending them via similarity 
    in SVD projection.

    Parameters
    ----------
    mov : numpy.ndarray
        Temporally binned movie of shape (nbins, Ly, Lx). Divided by
        `sdmov` in-place before processing.
    sdmov : numpy.ndarray
        Standard deviation of the movie, shape (Ly * Lx,) or (Ly, Lx),
        used for normalization.
    diameter : float or list of float
        Expected cell diameter in pixels.
    threshold_scaling : float, optional (default 1.0)
        Scaling factor for the peak detection threshold.
    connected : bool, optional (default True)
        If True, restrict each detected ROI to its largest connected component.
    max_iterations : int, optional (default 20)
        Maximum number of detection and refinement iterations.
    smooth_masks : bool, optional (default False)
        If True, spatially smooth frames before SVD projection during refinement.

    Returns
    -------
    new_settings : dict
        Dictionary with detection metadata including "diameter", "Vcorr",
        and placeholder keys "Vmax", "ihop", "Vsplit", "Vmap",
        "spatscale_pix".
    stat : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix",
        "xpix", and "lam".
    """
    mov /= sdmov
    
    change_codes = True
    t0 = time.time()
    
    U, u = getSVDdata(mov=mov, diameter=diameter)  # get SVD components
    S, StU, StS = getStU(diameter, U)
    Ly, Lx, nsvd = U.shape
    d0 = diameter
    sig = np.ceil(d0 / 4)  # smoothing constant
    # make array of radii values of size (2*d0+1,2*d0+1)
    rs, dy, dx = circleMask(d0)
    nsvd = U.shape[-1]
    nbasis = S.shape[-1]
    codes = np.zeros((0, nsvd), np.float32)
    LtU = np.zeros((0, nsvd), np.float32)
    LtS = np.zeros((0, nbasis), np.float32)
    L = np.zeros((Ly, Lx, 0), np.float32)
    # regress maps onto basis functions and subtract neuropil contribution
    neu = np.linalg.solve(StS, StU).astype("float32")
    Ucell = U - (S.reshape((-1, nbasis)) @ neu).reshape(U.shape)

    it = 0
    ncells = 0
    refine = -1

    # initialize
    ypix, xpix, lam = [], [], []
    logger.info(f"max_iterations = {max_iterations}; will stop when no more peaks above threshold, or max_iterations reached")
    for it in range(max_iterations):
        if refine < 0:
            V, us = getVmap(Ucell, sig)
            if it == 0:
                vrem = morphOpen(V, rs <= 1.)
            V = V - vrem  # make V more uniform
            if it == 0:
                V = V.astype("float64")
                # find indices of all maxima in +/- 1 range
                maxV = filters.maximum_filter(V, footprint=np.ones((3, 3)),
                                              mode="reflect")
                imax = V > (maxV - 1e-10)
                peaks = V[imax]
                # use the median of these peaks to decide if ROI is accepted
                thres = threshold_scaling * np.median(peaks[peaks > 1e-4])
                Vcorr = V.copy()
            V = np.minimum(V, Vcorr)

            # add extra ROIs here
            n = ncells
            while n < ncells + 200:
                ind = np.argmax(V)
                i, j = np.unravel_index(ind, V.shape)
                if V[i, j] < thres:
                    break
                yp, xp, la, ix, code = iter_extend(i, j, Ucell, us[i, j, :],
                                                   change_codes=change_codes)
                codes = np.append(codes, np.expand_dims(code, axis=0), axis=0)
                ypix.append(yp)
                xpix.append(xp)
                lam.append(la)
                Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n, :])

                yp, xp = extendROI(yp, xp, Ly, Lx, int(np.mean(d0)))
                V[yp, xp] = 0
                n += 1
            newcells = len(ypix) - ncells
            if it == 0:
                Nfirst = newcells
            L = np.append(L, np.zeros((Ly, Lx, newcells), "float32"), axis=-1)
            LtU = np.append(LtU, np.zeros((newcells, nsvd), "float32"), axis=0)
            LtS = np.append(LtS, np.zeros((newcells, nbasis), "float32"), axis=0)
            for n in range(ncells, len(ypix)):
                L[ypix[n], xpix[n], n] = lam[n]
                LtU[n, :] = lam[n] @ U[ypix[n], xpix[n], :]
                LtS[n, :] = lam[n] @ S[ypix[n], xpix[n], :]
            ncells += newcells

            # regression with neuropil
            LtL = L.reshape((-1, ncells)).transpose() @ L.reshape((-1, ncells))
            cellcode = np.concatenate((LtL, LtS), axis=1)
            neucode = np.concatenate((LtS.transpose(), StS), axis=1)
            codes = np.concatenate((cellcode, neucode), axis=0)
            Ucode = np.concatenate((LtU, StU), axis=0)
            codes = np.linalg.solve(codes + 1e-3 * np.eye((codes.shape[0])),
                                    Ucode).astype("float32")
            neu = codes[ncells:, :]
            codes = codes[:ncells, :]

        Ucell = U - (S.reshape((-1, nbasis)) @ neu + L.reshape(
            (-1, ncells)) @ codes).reshape(U.shape)
        # reestimate masks
        n, k = 0, 0
        while n < len(ypix):
            Ucell[ypix[n], xpix[n], :] += np.outer(lam[n], codes[k, :])
            ypix[n], xpix[n], lam[n], ix, codes[n, :] = iter_extend(
                ypix[n], xpix[n], Ucell, codes[k, :], refine, change_codes=change_codes)
            k += 1
            if ix.sum() == 0:
                logger.info("dropped ROI with no pixels")
                del ypix[n], xpix[n], lam[n]
                continue
            Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n, :])
            n += 1
        codes = codes[:n, :]
        ncells = len(ypix)
        L = np.zeros((Ly, Lx, ncells), "float32")
        LtU = np.zeros((ncells, nsvd), "float32")
        LtS = np.zeros((ncells, nbasis), "float32")
        for n in range(ncells):
            L[ypix[n], xpix[n], n] = lam[n]
            if refine < 0:
                LtU[n, :] = lam[n] @ U[ypix[n], xpix[n], :]
                LtS[n, :] = lam[n] @ S[ypix[n], xpix[n], :]
        err = (Ucell**2).mean()
        t1 = time.time() - t0
        logger.info(f"iter {it},\tROIs: {ncells},\terr: {err:0.4f}, \ttime: {t1:0.2f} sec")

        if refine == 0:
            break
        if refine == 2:
            # good place to get connected regions
            stat = [{
                "ypix": ypix[n],
                "lam": lam[n],
                "xpix": xpix[n]
            } for n in range(ncells)]
            stat = connected_region(stat, connected, Ly, Lx)
            ypix = [stat[n]["ypix"] for n in range(len(stat))]
            xpix = [stat[n]["xpix"] for n in range(len(stat))]
            lam = [stat[n]["lam"] for n in range(len(stat))]
            ncells = len(ypix)
        if refine > 0:
            Ucell = Ucell + (S.reshape((-1, nbasis)) @ neu).reshape(U.shape)
        if refine < 0 and (newcells < Nfirst / 10 or it == max_iterations - 1):
            refine = 3
            U = getSVDproj(mov, u, diameter, smooth_masks)
            Ucell = U
        if refine >= 0:
            StU = S.reshape((Ly * Lx, -1)).transpose() @ Ucell.reshape(
                (Ly * Lx, -1))
            #StU = np.reshape(S, (Lyc*Lxc,-1)).transpose() @ np.reshape(Ucell, (Lyc*Lxc, -1))
            neu = np.linalg.solve(StS, StU).astype("float32")
        refine -= 1
    Ucell = U - (S.reshape((-1, nbasis)) @ neu).reshape(U.shape)

    sdmov = np.reshape(sdmov, (Ly, Lx))
    stat = [{
        "ypix": ypix[n],
        "lam": lam[n] * sdmov[ypix[n], xpix[n]],
        "xpix": xpix[n]
    } for n in range(ncells)]

    stat = connected_region(stat, connected, Ly, Lx)
    # Remove empty cells
    stat = [s for s in stat if len(s["ypix"]) != 0]
    stat = np.array(stat)
    
    new_settings = {
        "diameter": diameter,
        "Vmax": 0,
        "ihop": 0,
        "Vsplit": 0,
        "Vcorr": Vcorr,
        "Vmap": 0,
        "spatscale_pix": 0
    }
    return new_settings, stat
