import time

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import maximum_filter
from scipy.ndimage.filters import uniform_filter
from scipy.stats import mode

from . import utils


def neuropil_subtraction(mov,lx):
    """ subtract low-pass filtered version of binned movie

    low-pass filtered version ~ neuropil
    subtract to help ignore neuropil
    
    Parameters
    ----------------

    mov : 3D array
        binned movie, size [nbins x Ly x Lx]

    lx : int
        size of filter

    Returns
    ----------------

    mov : 3D array
        binned movie with "neuropil" subtracted, size [nbins x Ly x Lx]

    """
    if len(mov.shape)<3:
        mov = mov[np.newaxis, :, :]
    nbinned, Ly, Lx = mov.shape
    c1 = uniform_filter(np.ones((Ly,Lx)), size=[lx, lx], mode = 'constant')
    for j in range(nbinned):
        mov[j] -= uniform_filter(mov[j], size=[lx, lx], mode = 'constant') / c1
    return mov

def square_conv2(mov,lx):
    """ convolve in pixels binned movie
    
    Parameters
    ----------------

    mov : 3D array
        binned movie, size [nbinned x Lyc x Lxc]

    lx : int
        filter size

    Returns
    ----------------

    movt : 3D array
        convolved + binned movie, size [nbinned x Lyc x Lxc]

    """
    if len(mov.shape)<3:
        mov = mov[np.newaxis, :, :]
    nbinned, Ly, Lx = mov.shape

    movt = np.zeros((nbinned, Ly, Lx), 'float32')
    for t in range(nbinned):
        movt[t] = lx * uniform_filter(mov[t], size=[lx, lx], mode = 'constant')
    return movt

def downsample(mov, flag=True):
    """ downsample in pixels binned movie
    
    Parameters
    ----------------

    mov : 3D array
        binned movie, size [nbinned x Lyc x Lxc]

    flag : bool (optional, default True)
        whether or not to edge taper

    Returns
    ----------------

    mov2 : 2D array
        downsampled + binned movie, size [nbinned x Lyp x Lxp]

    """
    if flag:
        nu = 2
    else:
        nu = 1
    if len(mov.shape)<3:
        mov = mov[np.newaxis, :, :]
    nbinned, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((nbinned,int(np.ceil(Ly/2)),Lx), 'float32')
    Ly0 = 2*int(Ly/2)
    for t in range(nbinned):
        movd[t,:int(Ly0/2),:] = (mov[t,0:Ly0:2,:] + mov[t,1:Ly0:2,:])/2
    if Ly%2==1:
        movd[:,-1,:] = mov[:,-1,:]/nu

    # bin along X
    mov2 = np.zeros((nbinned,int(np.ceil(Ly/2)),int(np.ceil(Lx/2))), 'float32')
    Lx0 = 2*int(Lx/2)
    for t in range(nbinned):
        mov2[t,:,:int(Lx0/2)] = (movd[t,:,0:Lx0:2] + movd[t,:,1:Lx0:2])/2
    if Lx%2==1:
        mov2[:,:,-1] = movd[:,:,-1]/nu
    return mov2

def threshold_reduce(movu, Th2):
    """ thresholded stddev of spatially downsampled binned movie
    
    is function faster without loop?

    Parameters
    ----------------

    movu : 3D array
        downsampled binned movie, size [nbinned x Lyp x Lxp]

    Th2 : float
        threshold on pixel intensity

    Returns
    ----------------

    Vt : 2D array
        stddev of pixels across time above threshold Th2

    """
    nbinned, Lyp, Lxp = movu.shape
    Vt = np.zeros((Lyp,Lxp), 'float32')
    for t in range(nbinned):
        Vt += movu[t]**2 * (movu[t]>Th2)
    Vt = Vt**.5
    return Vt

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
    ipix = np.arange(-lhf,-lhf+lx)+ np.zeros(lx, 'int32')[:, np.newaxis]
    x0 = xi + ipix
    y0 = yi + ipix.T
    mask  = np.ones((lx,lx), 'float32')
    ix = np.all((y0>=0, y0<Ly, x0>=0 , x0<Lx), axis=0)
    x0 = x0[ix]
    y0 = y0[ix]
    mask = mask[ix]
    mask = mask / (mask**2).sum()**.5
    return y0.flatten(), x0.flatten(), mask.flatten()

def iter_extend(ypix, xpix, rez, Lyc, Lxc, active_frames):
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
        usub = rez[np.ix_(active_frames, ypix*Lxc+ xpix)]
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


def sparsery(ops):
    """ bin ops['reg_file'] then detect ROIs using correlations in time
    
    Parameters
    ----------------

    ops : dictionary
        'reg_file', 'Ly', 'Lx', 'yrange', 'xrange', 'tau', 'fs', 'nframes', 'high_pass', 'batch_size'


    Returns
    ----------------

    ops : dictionary
        adds 'max_proj', 'Vcorr', 'Vmap', 'Vsplit'
    
    stat : array of dicts
        list of ROIs

    """
    rez, max_proj = utils.bin_movie(ops)
    ops['max_proj'] = max_proj
    nbinned, Lyc, Lxc = rez.shape
    # cropped size
    ops['Lyc'] = Lyc
    ops['Lxc'] = Lxc
    sdmov = utils.get_sdmov(rez, ops)
    rez /= sdmov
    
    # subtract low-pass filtered version of binned movie
    rez = neuropil_subtraction(rez, ops['spatial_hp'])

    LL = np.meshgrid(np.arange(Lxc), np.arange(Lyc))
    gxy = [np.array(LL).astype('float32')]
    dmov = rez
    movu = []

    # downsample movie at various spatial scales
    # downsampled sizes
    Lyp = np.zeros(5, 'int32')
    Lxp = np.zeros(5,'int32')
    for j in range(5):
        # convolve
        movu.append(square_conv2(dmov, 3))
        # downsample
        dmov = 2 * downsample(dmov)
        gxy0 = downsample(gxy[j], False)
        gxy.append(gxy0)
        nbinned, Lyp[j], Lxp[j] = movu[j].shape
        
    # find maximum spatial scale for each pixel
    V0 = []
    ops['Vmap']  = []
    for j in range(len(movu)):
        V0.append(movu[j].max(axis=0))
        ops['Vmap'].append(V0[j].copy())
    # spline over scales
    I = np.zeros((len(gxy), gxy[0].shape[1], gxy[0].shape[2]))
    for t in range(1,len(gxy)-1):
        gmodel = RectBivariateSpline(gxy[t][1,:,0], gxy[t][0, 0,:], ops['Vmap'][t],
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
    vmultiplier = max(1, np.float32(rez.shape[0])/1200)
    print('NOTE: %s spatial scale ~%d pixels, time epochs %2.2f, threshold %2.2f '%(fstr, 3*2**im, vmultiplier, vmultiplier*Th2))
    ops['spatscale_pix'] = 3*2**im

    V0 = []
    ops['Vmap']  = []
    # get standard deviation for pixels for all values > Th2
    for j in range(len(movu)):
        V0.append(threshold_reduce(movu[j], Th2))
        ops['Vmap'].append(V0[j].copy())
        movu[j] = np.reshape(movu[j], (movu[j].shape[0], -1))

    xpix,ypix,lam = [],[],[]
    rez = np.reshape(rez, (-1, Lyc*Lxc))
    lxs = 3 * 2**np.arange(5)
    nscales = len(lxs)

    niter = 250 * ops['max_iterations']
    Vmax = np.zeros((niter))
    ihop = np.zeros((niter))
    vrat = np.zeros((niter))
    Npix = np.zeros((niter))

    t0 = time.time()

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
        tproj = rez[:, ypix0*Lxc + xpix0] @ lam0
        active_frames = np.nonzero(tproj>Th2)[0] # frames with activity > Th2
        
        # extend mask based on activity similarity
        for j in range(3):
            ypix0, xpix0, lam0 = iter_extend(ypix0, xpix0, rez, Lyc, Lxc, active_frames)
            tproj = rez[:, ypix0*Lxc+ xpix0] @ lam0
            active_frames = np.nonzero(tproj>Th2)[0]
            if len(active_frames)<1:
                break
        if len(active_frames)<1:
            break

        # check if ROI should be split
        vrat[tj], ipack = two_comps(rez[:, ypix0*Lxc+ xpix0], lam0, Th2)
        if vrat[tj]>1.25:
            lam0, xp, active_frames = ipack
            tproj[active_frames] = xp
            ix = lam0>lam0.max()/5
            xpix0 = xpix0[ix]
            ypix0 = ypix0[ix]
            lam0 = lam0[ix]

        # update residual on raw movie
        rez[np.ix_(active_frames, ypix0*Lxc+ xpix0)] -= tproj[active_frames][:,np.newaxis] * lam0
        # update filtered movie
        ys, xs, lms = multiscale_mask(ypix0,xpix0,lam0, Lyp, Lxp)
        for j in range(nscales):
            movu[j][np.ix_(active_frames, xs[j]+Lxp[j]*ys[j])] -= np.outer(tproj[active_frames], lms[j])
            Mx = movu[j][:,xs[j]+Lxp[j]*ys[j]]
            V0[j][ys[j], xs[j]] = (Mx**2 * np.float32(Mx>Th2)).sum(axis=0)**.5
            
        xpix.append(xpix0)
        ypix.append(ypix0)
        lam.append(lam0)
        if tj%1000==0:
            print('%d ROIs, score=%2.2f'%(tj, Vmax[tj]))
    
    ops['Vmax'] = Vmax
    ops['ihop'] = ihop
    ops['Vsplit'] = vrat
    stat  = [{'ypix':ypix[n] + ops['yrange'][0], 'lam':lam[n]*sdmov[ypix[n], xpix[n]], 
              'xpix':xpix[n] + ops['xrange'][0], 'footprint': ops['ihop'][n]} for n in range(len(xpix))]

    return ops, stat
