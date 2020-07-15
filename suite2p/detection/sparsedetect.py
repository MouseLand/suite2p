from typing import Tuple, Dict, List, Any
from copy import deepcopy
from enum import Enum

import numpy as np
from numpy.linalg import norm

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import uniform_filter
from scipy.stats import mode

from . import utils
from .utils import temporal_high_pass_filter


def neuropil_subtraction(mov: np.ndarray, filter_size: int) -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""
    nbinned, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode='constant')
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filter_size, mode='constant') / c1)
    return movt


def square_convolution_2d(mov: np.ndarray, filter_size: int) -> np.ndarray:
    """Returns movie convolved by uniform kernel with width 'filter_size'."""
    movt = np.zeros_like(mov, dtype=np.float32)
    for frame, framet in zip(mov, movt):
        framet[:] = filter_size * uniform_filter(frame, size=filter_size, mode='constant')
    return movt


def multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp):
    # given a set of masks on the raw image, this functions returns the downsampled masks for all spatial scales
    xs = [xpix0]
    ys = [ypix0]
    lms = [lam0]
    for j in range(1,len(Lyp)):
        ipix, ind = np.unique(np.int32(xs[j-1]/2)+np.int32(ys[j-1]/2)*Lxp[j], return_inverse=True)
        LAM = np.zeros(len(ipix))
        for i in range(len(xs[j-1])):
            LAM[ind[i]] += lms[j-1][i]/2
        lms.append(LAM)
        ys.append(np.int32(ipix/Lxp[j]))
        xs.append(np.int32(ipix%Lxp[j]))
    for j in range(len(Lyp)):
        ys[j], xs[j], lms[j] = extend_mask(ys[j], xs[j], lms[j], Lyp[j], Lxp[j])
    return ys, xs, lms


def add_square(yi,xi,lx,Ly,Lx):
    """ return square of pixels around peak with norm 1
    
    Parameters
    ----------------

    yi : int
        y-center

    xi : int
        x-center

    lx : int
        x-width

    ly : int
        y-width

    Ly : int
        full y frame

    Lx : int
        full x frame

    Returns
    ----------------

    y0 : array
        pixels in y
    
    x0 : array
        pixels in x
    
    mask : array
        pixel weightings

    """
    lhf = int((lx-1)/2)
    ipix = np.tile(np.arange(-lhf, -lhf + lx, dtype=np.int32), reps=(lx, 1))
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask = np.ones_like(ipix, dtype=np.float32)
    ix = np.all((y0 >= 0, y0 < Ly, x0 >= 0, x0 < Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / norm(mask)
    return y0.flatten(), x0.flatten(), mask.flatten()


def iter_extend(ypix, xpix, mov, Lyc, Lxc, active_frames):
    """ extend mask based on activity of pixels on active frames
    ACTIVE frames determined by threshold
    
    Parameters
    ----------------
    
    ypix : array
        pixels in y
    
    xpix : array
        pixels in x
    
    mov : 2D array
        binned residual movie [nbinned x Lyc*Lxc]

    active_frames : 1D array
        list of active frames

    Returns
    ----------------
    ypix : array
        extended pixels in y
    
    xpix : array
        extended pixels in x
    lam : array
        pixel weighting
    """
    npix = 0
    iter = 0
    while npix<10000:
        npix = ypix.size
        # extend ROI by 1 pixel on each side
        ypix, xpix = extendROI(ypix, xpix, Lyc, Lxc, 1)
        # activity in proposed ROI on ACTIVE frames
        usub = mov[np.ix_(active_frames, ypix*Lxc+ xpix)]
        lam = np.mean(usub,axis=0)
        ix = lam>max(0, lam.max()/5.0)
        if ix.sum()==0:
            print('break')
            break;
        ypix, xpix,lam = ypix[ix],xpix[ix], lam[ix]
        if iter == 0:
            sgn = 1.
        if np.sign(sgn * (ix.sum()-npix))<=0:
            break
        else:
            npix = ypix.size
        iter += 1
    lam = lam/np.sum(lam**2)**.5
    return ypix, xpix, lam

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    """ extend ypix and xpix by niter pixel(s) on each side """
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix

def two_comps(mpix0, lam, Th2):
    """ check if splitting ROI increases variance explained

    Parameters
    ----------------
    
    mpix0 : 2D array
        binned movie for pixels in ROI [nbinned x npix]

    lam : array
        pixel weighting

    Th2 : float
        intensity threshold


    Returns
    ----------------

    vrat : array
        extended pixels in y
    
    ipick : tuple
        new ROI

    """
    mpix = mpix0.copy()
    xproj = mpix @ lam
    gf0 = xproj>Th2

    mpix[gf0, :] -= np.outer(xproj[gf0] , lam)
    vexp0 = np.sum(mpix0**2) - np.sum(mpix**2)

    k = np.argmax(np.sum(mpix * np.float32(mpix>0), axis=1))
    mu = [lam * np.float32(mpix[k] < 0), lam * np.float32(mpix[k] > 0)]

    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for mu0 in mu:
        mu0[:] /= norm(mu0) + 1e-6
        xp = mpix @ mu0
        mpix[gf0, :] -= np.outer(xp[gf0], mu0)
        goodframe.append(gf0)
        xproj.append(xp[gf0])

    flag = [False, False]
    V = np.zeros(2)
    for t in range(3):
        for k in range(2):
            if flag[k]:
                continue
            mpix[goodframe[k],:] += np.outer(xproj[k], mu[k])
            xp = mpix @ mu[k]
            goodframe[k]  = xp > Th2
            V[k] = np.sum(xp**2)
            if np.sum(goodframe[k])==0:
                flag[k] = True
                V[k] = -1
                continue
            xproj[k] = xp[goodframe[k]]
            mu[k] = np.mean(mpix[goodframe[k], :] * xproj[k][:,np.newaxis], axis=0)
            mu[k][mu[k]<0]  = 0
            mu[k] /=(1e-6 + np.sum(mu[k]**2)**.5)
            mpix[goodframe[k],:] -= np.outer(xproj[k], mu[k])
    k = np.argmax(V)
    vexp = np.sum(mpix0**2) - np.sum(mpix**2)
    vrat = vexp / vexp0
    return vrat, (mu[k], xproj[k], goodframe[k])

def extend_mask(ypix, xpix, lam, Ly, Lx):
    """ extend mask into 8 surrrounding pixels """
    nel = len(xpix)
    yx = ((ypix, ypix, ypix, ypix-1, ypix-1,ypix-1, ypix+1,ypix+1,ypix+1),
          (xpix, xpix+1,xpix-1,xpix, xpix+1,xpix-1,xpix, xpix+1,xpix-1))
    yx = np.array(yx)
    yx = yx.reshape((2,-1))
    yu, ind = np.unique(yx, axis=1, return_inverse=True)
    LAM = np.zeros(yu.shape[1])
    for j in range(len(ind)):
        LAM[ind[j]] += lam[j%nel]/3
    ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
    ypix1,xpix1 = yu[:, ix]
    lam1 = LAM[ix]
    return ypix1,xpix1,lam1


class EstimateMode(Enum):
    Forced = 'FORCED'
    Estimated = 'estimated'


def estimate_spatial_scale(I: np.ndarray) -> int:
    I0 = I.max(axis=0)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11, 11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, _ = mode(imap[ipk][isort[:50]])
    if im == 0:
        raise ValueError('ERROR: best scale was 0, everything should break now!')
    return im



def find_best_scale(I: np.ndarray, spatial_scale: int) -> Tuple[int, EstimateMode]:
    """
    Returns best scale and estimate method (if the spatial scale was forced (if positive) or estimated (the top peaks).
    """
    if spatial_scale > 0:
        return max(1, min(4, spatial_scale)), EstimateMode.Forced
    else:
        return estimate_spatial_scale(I=I), EstimateMode.Estimated


def sparsery(mov: np.ndarray, high_pass: int, neuropil_high_pass: int, batch_size: int, spatial_scale: int, threshold_scaling,
             max_iterations: int, yrange, xrange) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Returns stats and ops from 'mov' using correlations in time."""

    mov = temporal_high_pass_filter(mov=mov, width=int(high_pass))
    max_proj = mov.max(axis=0)

    sdmov = utils.standard_deviation_over_time(mov, batch_size=batch_size)
    mov = neuropil_subtraction(mov=mov / sdmov, filter_size=neuropil_high_pass)  # subtract low-pass filtered movie

    _, Lyc, Lxc = mov.shape
    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype('float32')]
    dmov = mov
    movu = []

    # downsample movie at various spatial scales
    Lyp, Lxp = np.zeros(5, 'int32'), np.zeros(5, 'int32')  # downsampled sizes
    for j in range(5):
        movu0 = square_convolution_2d(dmov, 3)
        dmov = 2 * utils.downsample(dmov)
        gxy0 = utils.downsample(gxy[j], False)
        gxy.append(gxy0)
        _, Lyp[j], Lxp[j] = movu0.shape
        movu.append(movu0)

    # spline over scales
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for movu0, gxy0, I0 in zip(movu, gxy, I):
        gmodel = RectBivariateSpline(gxy0[1, :, 0], gxy0[0, 0, :], movu0.max(axis=0),
                                     kx=min(3, gxy0.shape[1] - 1), ky=min(3, gxy0.shape[2] - 1))
        I0[:] = gmodel(gxy[0][1, :, 0], gxy[0][0, 0, :])
    v_corr = I.max(axis=0)

    # to set threshold, find best scale based on scale of top peaks
    im, estimate_mode = find_best_scale(I=I, spatial_scale=spatial_scale)
    spatscale_pix = 3 * 2 ** im
    Th2 = threshold_scaling * 5 * max(1, im)  # threshold for accepted peaks (scale it by spatial scale)
    vmultiplier = max(1, mov.shape[0] / 1200)
    print('NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f ' % (estimate_mode.value, spatscale_pix, vmultiplier, vmultiplier * Th2))

    # get standard deviation for pixels for all values > Th2
    v_map = [utils.threshold_reduce(movu0, Th2) for movu0 in movu]
    movu = [movu0.reshape(movu0.shape[0], -1) for movu0 in movu]

    mov = np.reshape(mov, (-1, Lyc * Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    v_max = np.zeros(max_iterations)
    ihop = np.zeros(max_iterations)
    v_split = np.zeros(max_iterations)
    V1 = deepcopy(v_map)
    stats = []
    for tj in range(max_iterations):
        # find peaks in stddev's
        v0max = np.array([V1[j].max() for j in range(5)])
        imap = np.argmax(v0max)
        imax = np.argmax(V1[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        # position of peak
        yi, xi = gxy[imap][1,yi,xi], gxy[imap][0,yi,xi]

        # check if peak is larger than threshold * max(1,nbinned/1200)
        v_max[tj] = v0max.max()
        if v_max[tj] < vmultiplier*Th2:
            break
        ls = lxs[imap]

        ihop[tj] = imap

        # make square of initial pixels based on spatial scale of peak
        ypix0, xpix0, lam0 = add_square(int(yi), int(xi), ls, Lyc, Lxc)
        
        # project movie into square to get time series
        tproj = mov[:, ypix0*Lxc + xpix0] @ lam0
        active_frames = np.nonzero(tproj>Th2)[0] # frames with activity > Th2
        
        # extend mask based on activity similarity
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, mov, Lyc, Lxc, active_frames)
            tproj = mov[:, ypix0*Lxc+ xpix0] @ lam0
            active_frames = np.nonzero(tproj>Th2)[0]
            if len(active_frames)<1:
                break
        if len(active_frames)<1:
            break

        # check if ROI should be split
        v_split[tj], ipack = two_comps(mov[:, ypix0 * Lxc + xpix0], lam0, Th2)
        if v_split[tj] > 1.25:
            lam0, xp, active_frames = ipack
            tproj[active_frames] = xp
            ix = lam0 > lam0.max() / 5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]

        # update residual on raw movie
        mov[np.ix_(active_frames, ypix0*Lxc+ xpix0)] -= tproj[active_frames][:,np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(active_frames, xs[j]+Lxp[j]*ys[j])] -= np.outer(tproj[active_frames], lms[j])
            Mx = movu[j][:,xs[j]+Lxp[j]*ys[j]]
            V1[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx>Th2)).sum(axis=0)**.5

        stats.append({
            'ypix': ypix0 + yrange[0],
            'xpix': xpix0 + xrange[0],
            'lam': lam0 * sdmov[ypix0, xpix0],
            'footprint': ihop[tj]
        })

        if tj % 1000 == 0:
            print('%d ROIs, score=%2.2f' % (tj, v_max[tj]))
    
    new_ops = {
        'max_proj': max_proj,
        'Vmax': v_max,
        'ihop': ihop,
        'Vsplit': v_split,
        'Vcorr': v_corr,
        'Vmap': v_map,
        'spatscale_pix': spatscale_pix,
    }

    return new_ops, stats
