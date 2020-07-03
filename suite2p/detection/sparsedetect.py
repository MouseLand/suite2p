from copy import deepcopy

import numpy as np
from numpy.linalg import norm

from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import uniform_filter
from scipy.stats import mode

from . import utils


def neuropil_subtraction(mov: np.ndarray, filter_size: int) -> None:
    """Returns movie subtracted by a low-pass filtered version of itself to help ignore neuropil."""
    nbinned, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly, Lx)), size=filter_size, mode='constant')
    movt = np.zeros_like(mov)
    for frame, framet in zip(mov, movt):
        framet[:] = frame - (uniform_filter(frame, size=filter_size, mode='constant') / c1)
    return movt


def square_conv2(mov: np.ndarray, filter_size: int) -> np.ndarray:
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

def iter_extend(ypix, xpix, rez, Lyc,Lxc):
    """ extend mask based on activity of pixels on active frames

    ACTIVE frames determined by threshold
    
    Parameters
    ----------------
    
    ypix : array
        pixels in y
    
    xpix : array
        pixels in x
    
    rez : 2D array
        binned movie on active frames [nactive x Lyc*Lxc]

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
        usub = rez[:, ypix*Lxc+ xpix]
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
    mu = [mpix[k].copy()]
    mu.append(mpix[k].copy())
    mu[0] = lam * np.float32(mu[0]<0)
    mu[1] = lam * np.float32(mu[1]>0)
    mpix = mpix0.copy()
    goodframe = []
    xproj = []
    for k in range(2):
        mu[k] /=(1e-6 + np.sum(mu[k]**2)**.5)
        xp = mpix @ mu[k]
        goodframe.append(gf0)
        xproj.append(xp[goodframe[k]])
        mpix[goodframe[k],:] -= np.outer(xproj[k], mu[k])

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


def sparsery(mov: np.ndarray, high_pass: int, neuropil_high_pass: int, batch_size: int, ops):
    """detect ROIs in 'mov' using correlations in time.
    
    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'yrange', 'xrange', 'tau', 'fs', 'nframes'


    Returns
    ----------------

    ops : dictionary
        adds 'max_proj', 'Vcorr', 'Vmap', 'Vsplit'
    
    stat : array of dicts
        list of ROIs

    """
    mov = utils.hp_gaussian_filter(mov, high_pass) if high_pass < 10 else utils.hp_rolling_mean_filter(mov, high_pass)  # gaussian is slower

    ops['max_proj'] = mov.max(axis=0)
    nbinned, Lyc, Lxc = mov.shape
    sdmov = utils.standard_deviation_over_time(mov, batch_size=batch_size)
    mov = mov / sdmov
    mov = neuropil_subtraction(mov, filter_size=neuropil_high_pass)  # subtract low-pass filtered version of binned movie

    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype('float32')]
    dmov = mov
    movu = []

    # downsample movie at various spatial scales
    # downsampled sizes
    Lyp = np.zeros(5, 'int32')
    Lxp = np.zeros(5,'int32')
    for j in range(5):
        # convolve
        movu.append(square_conv2(dmov, 3))
        # downsample
        dmov = 2 * utils.downsample(dmov)
        gxy0 = utils.downsample(gxy[j], False)
        gxy.append(gxy0)
        nbinned, Lyp[j], Lxp[j] = movu[j].shape

    # spline over scales
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for t in range(1,len(gxy)-1):
        gmodel = RectBivariateSpline(gxy[t][1,:,0], gxy[t][0, 0,:], movu[t].max(axis=0),
                                     kx=min(3, gxy[t][1,:,0].size-1), ky=min(3, gxy[t][0,0,:].size-1))
        I[t] = gmodel.__call__(gxy[0][1,:,0], gxy[0][0, 0,:])
    I0 = I.max(axis=0)
    ops['Vcorr'] = I0

    # find best scale based on scale of top peaks
    # (used  to set threshold)
    imap = np.argmax(I, axis=0).flatten()
    ipk = np.abs(I0 - maximum_filter(I0, size=(11,11))).flatten() < 1e-4
    isort = np.argsort(I0.flatten()[ipk])[::-1]
    im, nm = mode(imap[ipk][isort[:50]])
    if ops['spatial_scale'] > 0:
        im = max(1, min(4, ops['spatial_scale']))
        fstr = 'FORCED'
    else:
        fstr = 'estimated'
    if im==0:
        print('ERROR: best scale was 0, everything should break now!')

    # threshold for accepted peaks (scale it by spatial scale)
    Th2 = ops['threshold_scaling']*5*max(1,im)
    vmultiplier = max(1, np.float32(mov.shape[0]) / 1200)
    print('NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f '%(fstr, 3*2**im, vmultiplier, vmultiplier*Th2))
    ops['spatscale_pix'] = 3*2**im

    # get standard deviation for pixels for all values > Th2
    V0 = [utils.threshold_reduce(movu0, Th2) for movu0 in movu]
    ops['Vmap'] = deepcopy(V0)
    movu = [movu0.reshape(movu0.shape[0], -1) for movu0 in movu]

    mov = np.reshape(mov, (-1, Lyc * Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    niter = 250 * ops['max_iterations']
    Vmax = np.zeros(niter)
    ihop = np.zeros(niter)
    vrat = np.zeros(niter)
    stats = []
    for tj in range(niter):
        # find peaks in stddev's
        v0max = np.array([V0[j].max() for j in range(5)])
        imap = np.argmax(v0max)
        imax = np.argmax(V0[imap])
        yi, xi = np.unravel_index(imax, (Lyp[imap], Lxp[imap]))
        # position of peak
        yi, xi = gxy[imap][1,yi,xi], gxy[imap][0,yi,xi]

        # check if peak is larger than threshold * max(1,nbinned/1200)
        Vmax[tj] = v0max.max()
        if Vmax[tj] < vmultiplier*Th2:
            break
        ls = lxs[imap]

        ihop[tj] = imap

        # make square of initial pixels based on spatial scale of peak
        ypix0, xpix0, lam0 = add_square(int(yi), int(xi), ls, Lyc, Lxc)
        
        # project movie into square to get time series
        tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
        goodframe = np.nonzero(tproj>Th2)[0] # frames with activity > Th2
        
        # extend mask based on activity similarity
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, mov[goodframe], Lyc, Lxc)
            tproj = mov[:, ypix0 * Lxc + xpix0] @ lam0
            goodframe = np.nonzero(tproj>Th2)[0]
            if len(goodframe)<1:
                break
        if len(goodframe)<1:
            break

        # check if ROI should be split
        vrat[tj], ipack = two_comps(mov[:, ypix0 * Lxc + xpix0], lam0, Th2)
        if vrat[tj]>1.25:
            lam0, xp, goodframe = ipack
            tproj[goodframe] = xp
            ix = lam0>lam0.max()/5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]

        # update residual on raw movie
        mov[np.ix_(goodframe, ypix0 * Lxc + xpix0)] -= tproj[goodframe][:, np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(goodframe, xs[j]+Lxp[j]*ys[j])] -= np.outer(tproj[goodframe], lms[j])
            Mx = movu[j][:,xs[j]+Lxp[j]*ys[j]]
            V0[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx>Th2)).sum(axis=0)**.5

        stats.append({
            'ypix': ypix0 + ops['yrange'][0],
            'lam': lam0 * sdmov[ypix0, xpix0],
            'xpix': xpix0 + ops['xrange'][0],
            'footprint': ihop[tj]
        })

        if tj%1000==0:
            print('%d ROIs, score=%2.2f'%(tj, Vmax[tj]))
    
    ops.update({
        'Vmax': Vmax,
        'ihop': ihop,
        'Vsplit': vrat,
    })

    return ops, stats
