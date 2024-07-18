"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from multiprocessing import Pool

import numpy as np
from numpy.linalg import norm
from scipy.signal import convolve2d
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from . import register
from .. import io, default_settings


import torch 

def pclowhigh(mov, nlowhigh, nPC, random_state):
    """
    Compute mean of top and bottom PC weights for nPC"s of mov

    computes nPC PCs of mov and returns average of top and bottom

    Parameters
    ----------
    mov : frames x Ly x Lx
        subsampled frames from movie
    nlowhigh : int
        number of frames to average at top and bottom of each PC
    nPC : int
        number of PCs to compute
    random_state:
        a value that sets the seed for the PCA randomizer.

    Returns
    -------
        pclow : float, array
            average of bottom of spatial PC: nPC x Ly x Lx
        pchigh : float, array
            average of top of spatial PC: nPC x Ly x Lx
        w : float, array
            singular values of decomposition of mov
        v : float, array
            frames x nPC, how the PCs vary across frames
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
                maxregshift=0.1, maxregshiftNR=10, snr_thresh=1.25,
                spatial_taper=3.45):
    """
    register top and bottom of PCs to each other

    Parameters
    ----------
    pclow : float, array
        average of bottom of spatial PC: nPC x Ly x Lx
    pchigh : float, array
        average of top of spatial PC: nPC x Ly x Lx
    smooth_sigma : int
        see registration settings
    block_size : int, int
        see registration settings
    maxregshift : float
        see registration settings
    maxregshiftNR : int
        see registration settings
    snr_thresh: float
        signal to noise threshold to use.
    spatial_taper: float

    Returns
    -------
        X : float array
            nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
    """
    # registration settings
    nPC, Ly, Lx = pclow.shape

    X = np.zeros((nPC, 3))
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
    return X

def get_pc_metrics(mov, settings, use_red=False):

    """
    Computes registration metrics using top PCs of registered movie

    movie saved as binary file settings["reg_file"]
    metrics saved to settings["regPC"] and settings["X"]
    "regDX" is nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
    "regPC" is average of top and bottom frames for each PC
    "tPC" is PC across time frames

    Parameters
    ----------
    settings : dict
        "nframes", "Ly", "Lx", "reg_file" (if use_red=True, "reg_file_chan2")
        (optional, "refImg", "block_size", "maxregshiftNR", "smooth_sigma", "maxregshift", "1Preg")
    use_red : :obj:`bool`, optional
        default False, whether to use "reg_file" or "reg_file_chan2"

    Returns
    -------
    settings : dict
        The same as the settings input, but will now include "regPC", "tPC", and "regDX".

    """
    random_state = settings["reg_metrics_rs"] if "reg_metrics_rs" in settings else None
    nPC = settings["reg_metric_n_pc"] if "reg_metric_n_pc" in settings else 30
    pclow, pchigh, sv, settings["tPC"] = pclowhigh(
        mov, nlowhigh=np.minimum(300, int(settings["nframes"] / 2)), nPC=nPC,
        random_state=random_state)
    settings["regPC"] = torch.stack((pclow, pchigh), dim=0).cpu().numpy()
    settings["regDX"] = pc_register(
        pclow, pchigh, smooth_sigma=settings["smooth_sigma"], block_size=settings["block_size"],
        maxregshift=settings["maxregshift"], maxregshiftNR=settings["maxregshiftNR"], 
        snr_thresh=settings["snr_thresh"], spatial_taper=settings["spatial_taper"])
    return settings


def filt_worker(inputs):
    X, filt = inputs
    for n in range(X.shape[0]):
        X[n, :, :] = convolve2d(X[n, :, :], filt, "same")
    return X


def filt_parallel(data, filt, num_cores):
    nimg = data.shape[0]
    nbatch = int(np.ceil(nimg / float(num_cores)))
    inputs = np.arange(0, nimg, nbatch)
    irange = []
    dsplit = []
    for i in inputs:
        ilist = i + np.arange(0, np.minimum(nbatch, nimg - i), 1, int)
        irange.append(ilist)
        dsplit.append([data[ilist, :, :], filt])
    if num_cores > 1:
        with Pool(num_cores) as p:
            results = p.map(filt_worker, dsplit)
        results = np.concatenate(results, axis=0)
    else:
        results = filt_worker(dsplit[0])
    return results


def local_corr(mov, batch_size, num_cores):
    """ computes correlation image on mov (nframes x pixels x pixels) """
    nframes, Ly, Lx = mov.shape

    filt = np.ones((3, 3), np.float32)
    filt[1, 1] = 0
    filt /= norm(filt)
    ix = 0
    k = 0
    filtnorm = convolve2d(np.ones((Ly, Lx)), filt, "same")

    img_corr = np.zeros((Ly, Lx), np.float32)
    while ix < nframes:
        ifr = np.arange(ix, min(ix + batch_size, nframes), 1, int)

        X = mov[ifr, :, :]
        X = X.astype(np.float32)
        X -= X.mean(axis=0)
        Xstd = X.std(axis=0)
        Xstd[Xstd == 0] = np.inf
        #X /= np.maximum(1, X.std(axis=0))
        X /= Xstd
        #for n in range(X.shape[0]):
        #    X[n,:,:] *= convolve2d(X[n,:,:], filt, "same")
        X *= filt_parallel(X, filt, num_cores)
        img_corr += X.mean(axis=0)
        ix += batch_size
        k += 1
    img_corr /= filtnorm
    img_corr /= float(k)
    return img_corr


def bin_median(mov, window=10):
    nframes, Ly, Lx = mov.shape
    if nframes < window:
        window = nframes
    mov = np.nanmedian(
        np.reshape(mov[:int(np.floor(nframes / window) * window), :, :],
                   (-1, window, Ly, Lx)).mean(axis=1), axis=0)
    return mov


def corr_to_template(mov, tmpl):
    nframes, Ly, Lx = mov.shape
    tmpl_flat = tmpl.flatten()
    tmpl_flat -= tmpl_flat.mean()
    tmpl_std = tmpl_flat.std()

    mov_flat = np.reshape(mov, (nframes, -1)).astype(np.float32)
    mov_flat -= mov_flat.mean(axis=1)[:, np.newaxis]
    mov_std = (mov_flat**2).mean(axis=1)**0.5

    correlations = (mov_flat * tmpl_flat).mean(axis=1) / (tmpl_std * mov_std)

    return correlations


def optic_flow(mov, tmpl, nflows):
    """ optic flow computation using farneback """
    window = int(1 / 0.2)  # window size
    nframes, Ly, Lx = mov.shape
    mov = mov.astype(np.float32)
    mov = np.reshape(mov[:int(np.floor(nframes / window) * window), :, :],
                     (-1, window, Ly, Lx)).mean(axis=1)

    mov = mov[np.random.permutation(mov.shape[0])[:min(nflows, mov.shape[0])], :, :]

    pyr_scale = .5
    levels = 3
    winsize = 100
    iterations = 15
    poly_n = 5
    poly_sigma = 1.2 / 5
    flags = 0

    nframes, Ly, Lx = mov.shape
    norms = np.zeros((nframes,))
    flows = np.zeros((nframes, Ly, Lx, 2))

    for n in range(nframes):
        flow = cv2.calcOpticalFlowFarneback(tmpl, mov[n, :, :], None, pyr_scale, levels,
                                            winsize, iterations, poly_n, poly_sigma,
                                            flags)

        flows[n, :, :, :] = flow
        norms[n] = norm(flow)

    return flows, norms


def get_flow_metrics(settings):
    """ get farneback optical flow and some other stats from normcorre paper """
    # done in batches for memory reasons
    Ly = settings["Ly"]
    Lx = settings["Lx"]
    reg_file = open(settings["reg_file"], "rb")
    nbatch = settings["batch_size"]
    nbytesread = 2 * Ly * Lx * nbatch

    Lyc = settings["yrange"][1] - settings["yrange"][0]
    Lxc = settings["xrange"][1] - settings["xrange"][0]
    img_corr = np.zeros((Lyc, Lxc), np.float32)
    img_median = np.zeros((Lyc, Lxc), np.float32)
    correlations = np.zeros((0,), np.float32)
    flows = np.zeros((0, Lyc, Lxc, 2), np.float32)
    norms = np.zeros((0,), np.float32)
    smoothness = 0
    smoothness_corr = 0

    nflows = np.minimum(settings["nframes"], int(np.floor(100 / (settings["nframes"] / nbatch))))
    ncorrs = np.minimum(settings["nframes"], int(np.floor(1000 / (settings["nframes"] / nbatch))))

    k = 0
    while True:
        buff = reg_file.read(nbytesread)
        mov = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        if mov.size == 0:
            break
        mov = np.reshape(mov, (-1, Ly, Lx))

        mov = mov[np.ix_(np.arange(0, mov.shape[0], 1, int),
                         np.arange(settings["yrange"][0], settings["yrange"][1], 1, int),
                         np.arange(settings["xrange"][0], settings["xrange"][1], 1, int))]

        img_corr += local_corr(mov[:, :, :], 1000, settings["num_workers"])
        img_median += bin_median(mov)
        k += 1

        smoothness += np.sqrt(
            np.sum(np.sum(np.array(np.gradient(np.mean(mov, 0)))**2, 0)))
        smoothness_corr += np.sqrt(np.sum(np.sum(np.array(np.gradient(img_corr))**2,
                                                 0)))

        tmpl = img_median / k

        correlations0 = corr_to_template(mov, tmpl)
        correlations = np.hstack((correlations, correlations0))
        if HAS_CV2:
            flows0, norms0 = optic_flow(mov, tmpl, nflows)
        else:
            flows0 = []
            norms0 = []
            print("flows not computed, cv2 not installed / did not import correctly")

        flows = np.vstack((flows, flows0))
        norms = np.hstack((norms, norms0))

    img_corr /= float(k)
    img_median /= float(k)

    smoothness /= float(k)
    smoothness_corr /= float(k)

    return tmpl, correlations, flows, norms, smoothness, smoothness_corr, img_corr
