import math
import time

import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from matplotlib.colors import hsv_to_rgb

from .stats import fitMVGaus
from .utils import temporal_high_pass_filter, standard_deviation_over_time

def getSVDdata(mov: np.ndarray, ops):

    mov = temporal_high_pass_filter(mov, width=int(ops['high_pass']))
    ops['max_proj'] = mov.max(axis=0)
    nbins, Lyc, Lxc = np.shape(mov)

    sig = ops['diameter']/10. # PICK UP
    for j in range(nbins):
        mov[j,:,:] = gaussian_filter(mov[j,:,:], sig)

    # compute noise variance across frames
    sdmov = standard_deviation_over_time(mov, batch_size=ops['batch_size'])
    mov /= sdmov
    mov = np.reshape(mov, (-1,Lyc*Lxc))
    # compute covariance of binned frames
    cov = mov @ mov.transpose() / mov.shape[1]
    cov = cov.astype('float32')

    nsvd_for_roi = min(ops['nbinned'], int(cov.shape[0]/2))
    u, s, v = np.linalg.svd(cov)

    u = u[:, :nsvd_for_roi]
    U = u.transpose() @ mov
    U = np.reshape(U, (-1,Lyc,Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()
    return ops, U, sdmov, u

def getSVDproj(mov: np.ndarray, ops, u):

    mov = temporal_high_pass_filter(mov, int(ops['high_pass']))

    nbins, Lyc, Lxc = np.shape(mov)
    if ('smooth_masks' in ops) and ops['smooth_masks']:
        sig = np.maximum([.5, .5], ops['diameter']/20.)
        for j in range(nbins):
            mov[j,:,:] = gaussian_filter(mov[j,:,:], sig)
    if 1:
        sdmov = standard_deviation_over_time(mov, batch_size=ops['batch_size'])
        mov/=sdmov
        mov = np.reshape(mov, (-1,Lyc*Lxc))

        U = u.transpose() @ mov
        U = U.transpose().copy().reshape((Lyc,Lxc,-1))
    else:
        U = np.transpose(mov, (1, 2, 0)).copy()
    return U, sdmov


def getStU(ops, U):
    Lyc, Lxc, nbins = np.shape(U)
    S = create_neuropil_basis(ops, Lyc, Lxc)
    # compute covariance of neuropil masks with spatial masks
    StU = S.reshape((Lyc*Lxc,-1)).transpose() @ U.reshape((Lyc*Lxc,-1))
    StS = S.reshape((Lyc*Lxc,-1)).transpose() @ S.reshape((Lyc*Lxc,-1))
    #U = np.reshape(U, (-1,Lyc,Lxc))
    return S, StU , StS

def drawClusters(stat, ops):
    Ly = ops['Lyc']
    Lx = ops['Lxc']

    ncells = len(stat)
    r=np.random.random((ncells,))
    iclust = -1*np.ones((Ly,Lx),np.int32)
    Lam = np.zeros((Ly,Lx))
    H = np.zeros((Ly,Lx,1))
    for n in range(ncells):
        isingle = Lam[stat[n]['ypix'],stat[n]['xpix']]+1e-4 < stat[n]['lam']
        y = stat[n]['ypix'][isingle]
        x = stat[n]['xpix'][isingle]
        Lam[y,x] = stat[n]['lam'][isingle]
        #iclust[ypix,xpix] = n*np.ones(ypix.shape)
        H[y,x,0] = r[n]*np.ones(y.shape)

    S  = np.ones((Ly,Lx,1))
    V  = np.maximum(0, np.minimum(1, 0.75 * Lam / Lam[Lam>1e-10].mean()))
    V  = np.expand_dims(V,axis=2)
    hsv = np.concatenate((H,S,V),axis=2)
    rgb = hsv_to_rgb(hsv)

    return rgb


def create_neuropil_basis(ops, Ly, Lx):
    ''' computes neuropil basis functions
        inputs:
            ops, Ly, Lx
            from ops: ratio_neuropil, tile_factor, diameter, neuropil_type
        outputs:
            basis functions (pixels x nbasis functions)
    '''
    if 'ratio_neuropil' in ops:
        ratio_neuropil = ops['ratio_neuropil']
    else:
        ratio_neuropil = 6.
    if 'tile_factor' in ops:
        tile_factor    = ops['tile_factor']
    else:
        tile_factor = 1.
    diameter       = ops['diameter']

    ntilesY  = 1+2*int(np.ceil(tile_factor * Ly / (ratio_neuropil * diameter[0]/2))/2)
    ntilesX  = 1+2*int(np.ceil(tile_factor * Lx / (ratio_neuropil * diameter[1]/2))/2)
    ntilesY  = np.maximum(2,ntilesY)
    ntilesX  = np.maximum(2,ntilesX)
    yc = np.linspace(1, Ly, ntilesY)
    xc = np.linspace(1, Lx, ntilesX)
    ys = np.arange(0,Ly)
    xs = np.arange(0,Lx)

    Kx = np.ones((Lx, ntilesX), 'float32')
    Ky = np.ones((Ly, ntilesY), 'float32')
    if 1:
        # basis functions are fourier modes
        for k in range(int((ntilesX-1)/2)):
            Kx[:,2*k+1] = np.sin(2*math.pi * (xs+0.5) *  (1+k)/Lx)
            Kx[:,2*k+2] = np.cos(2*math.pi * (xs+0.5) *  (1+k)/Lx)
        for k in range(int((ntilesY-1)/2)):
            Ky[:,2*k+1] = np.sin(2*math.pi * (ys+0.5) *  (1+k)/Ly)
            Ky[:,2*k+2] = np.cos(2*math.pi * (ys+0.5) *  (1+k)/Ly)
    else:
        for k in range(ntilesX):
            Kx[:,k] = np.cos(math.pi * (xs+0.5) *  k/Lx)
        for k in range(ntilesY):
            Ky[:,k] = np.cos(math.pi * (ys+0.5) *  k/Ly)

    S = np.zeros((ntilesY, ntilesX, Ly, Lx), np.float32)
    for kx in range(ntilesX):
        for ky in range(ntilesY):
            S[ky,kx,:,:] = np.outer(Ky[:,ky], Kx[:,kx])

    S = np.reshape(S,(ntilesY*ntilesX, Ly*Lx))
    S = S / np.reshape(np.sum(S**2,axis=-1)**0.5, (-1,1))
    S = np.transpose(S, (1, 0)).copy()
    S = np.reshape(S, (Ly, Lx, -1))
    return S

def circleMask(d0):
    ''' creates array with indices which are the radius of that x,y point
        inputs:
            d0 (patch of (-d0,d0+1) over which radius computed
        outputs:
            rs: array (2*d0+1,2*d0+1) of radii
            dx,dy: indices in rs where the radius is less than d0
    '''
    dx  = np.tile(np.arange(-d0[1],d0[1]+1)/d0[1], (2*d0[0]+1,1))
    dy  = np.tile(np.arange(-d0[0],d0[0]+1)/d0[0], (2*d0[1]+1,1))
    dy  = dy.transpose()

    rs  = (dy**2 + dx**2) ** 0.5
    dx  = dx[rs<=1.]
    dy  = dy[rs<=1.]
    return rs, dx, dy

def morphOpen(V, footprint):
    ''' computes the morphological opening of V (correlation map) with circular footprint'''
    vrem   = filters.minimum_filter(V, footprint=footprint)
    vrem   = -filters.minimum_filter(-vrem, footprint=footprint)
    return vrem

def localMax(V, footprint, thres):
    ''' find local maxima of V (correlation map) using a filter with (usually circular) footprint
        inputs:
            V, footprint, thres
        outputs:
            i,j: indices of local max greater than thres
    '''
    maxV = filters.maximum_filter(V, footprint=footprint, mode = 'reflect')
    imax = V > np.maximum(thres, maxV - 1e-10)
    i,j  = imax.nonzero()
    i    = i.astype(np.int32)
    j    = j.astype(np.int32)
    return i,j

def localRegion(i,j,dy,dx,Ly,Lx):
    ''' returns valid indices of local region surrounding (i,j) of size (dy.size, dx.size)'''
    xc = dx + j
    yc = dy + i
    goodi = (xc>=0) & (xc<Lx) & (yc>=0) & (yc<Ly)
    xc = xc[goodi]
    yc = yc[goodi]
    yc = yc.astype(np.int32)
    xc = xc.astype(np.int32)
    return yc, xc, goodi

def pairwiseDistance(y,x):
    dists = ((np.expand_dims(y,axis=-1) - np.expand_dims(y,axis=0))**2
         + (np.expand_dims(x,axis=-1) - np.expand_dims(x,axis=0))**2)**0.5
    return dists


def r_squared(yp, xp, ypix, xpix, diam_y, diam_x, estimator=np.median):
    return np.sqrt(((yp - estimator(ypix)) / diam_y) ** 2 + (((xp - estimator(xpix)) / diam_x) ** 2))


# this function needs to be updated with the new stat
def get_stat(ops, stats, Ucell, codes, frac=0.5):
    '''computes statistics of cells found using sourcery
    inputs:
        Ly, Lx, d0, mPix (pixels,ncells), mLam (weights,ncells), codes (ncells,nsvd), Ucell (nsvd,Ly,Lx)
    outputs:
        stat
        assigned to stat: ipix, ypix, xpix, med, npix, lam, footprint, compact, aspect_ratio, ellipse
    '''
    d0, Ly, Lx = ops['diameter'], ops['Lyc'], ops['Lxc']
    rs, dy, dx = circleMask(d0)
    rsort = np.sort(rs.flatten())

    # Remove empty cells
    stats = [stat for stat in stats if len(stat['ypix']) != 0]

    footprints = np.zeros(len(stats))
    for k, (stat, code) in enumerate(zip(stats, codes)):
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']

        # compute footprint of ROI
        yp, xp = extendROI(ypix, xpix, Ly, Lx, int(np.mean(d0)))

        # compute compactness of ROI
        rs = r_squared(yp=yp, xp=xp, ypix=ypix, xpix=xpix, diam_y=d0[0], diam_x=d0[1])
        stat['mrs'] = np.mean(rs)
        stat['mrs0'] = np.mean(rsort[:ypix.size])
        stat['compact'] = stat['mrs'] / (1e-10 + stat['mrs0'])
        stat['ypix'] += ops['yrange'][0]
        stat['xpix'] += ops['xrange'][0]
        stat['med'] = [np.median(stat['ypix']), np.median(stat['xpix'])]
        stat['npix'] = xpix.size
        if 'radius' not in stat:
            ry, rx = fitMVGaus(ypix, xpix, lam, dy=d0[0], dx=d0[1], thres=2).radii
            stat['radius'] = ry * d0.mean()
            stat['aspect_ratio'] = 2 * ry/(.01 + ry + rx)

        proj = (Ucell[yp, xp, :] @ np.expand_dims(code, axis=1)).flatten()
        footprints[k] = np.nanmean(rs[proj > proj.max() * frac])

    mfoot = np.nanmedian(footprints)
    for stat, footprint in zip(stats, footprints):
        stat['footprint'] = footprint / mfoot if not np.isnan(footprint) else 0

    npix = np.array([stat['npix'] for stat in stats], dtype='float32')
    npix /= np.mean(npix[:100])
    for stat, npix0 in zip(stats, npix):
        stat['npix_norm'] = npix0

    return stats


def getVmap(Ucell, sig):
    us = gaussian_filter(Ucell, [sig[0], sig[1], 0.], mode='wrap')
    # compute log variance at each location
    log_variances = (us**2).mean(axis=-1) / gaussian_filter((Ucell**2).mean(axis=-1), sig, mode='wrap')
    return log_variances.astype('float64'), us


def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols


def minDistance(inputs):
    y1, x1, y2, x2 = inputs
    ds = (y1 - np.expand_dims(y2, axis=1))**2 + (x1 - np.expand_dims(x2, axis=1))**2
    return np.amin(ds)**.5

def get_connected(Ly, Lx, stat):
    '''grow i0 until it cannot grow any more
    '''
    ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
    i0   = np.argmax(lam)
    mask = np.zeros((Ly, Lx))
    mask[ypix,xpix] = lam
    ypix, xpix = ypix[i0], xpix[i0]
    nsel = 1
    while 1:
        ypix,xpix = extendROI(ypix, xpix, Ly, Lx)
        ix = mask[ypix,xpix]>1e-10
        ypix,xpix = ypix[ix], xpix[ix]
        if len(ypix)<=nsel:
            break
        nsel = len(ypix)
    lam = mask[ypix, xpix]
    stat['ypix'], stat['xpix'], stat['lam'] = ypix, xpix, lam
    return stat

def connected_region(stat, ops):
    if ('connected' not in ops) or ops['connected']:
        for j in range(len(stat)):
            stat[j] = get_connected(ops['Lyc'], ops['Lxc'], stat[j])
    return stat

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix

def iter_extend(ypix, xpix, Ucell, code, refine=-1, change_codes=False):
    Lyc, Lxc, nsvd = Ucell.shape
    npix = 0
    iter = 0
    while npix<10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix,xpix,Lyc,Lxc, 1)
        usub = Ucell[ypix, xpix, :]
        lam = usub @ np.expand_dims(code, axis=1)
        lam = np.squeeze(lam, axis=1)
        # ix = lam>max(0, np.mean(lam)/3)
        ix = lam>max(0, lam.max()/5.0)
        if ix.sum()==0:
            break;
        ypix, xpix,lam = ypix[ix],xpix[ix], lam[ix]
        lam = lam/np.sum(lam**2+1e-10)**.5
        if refine<0 and change_codes:
            code = lam @ usub[ix, :]
        if iter == 0:
            sgn = 1.
            #sgn = np.sign(ix.sum()-npix)
        if np.sign(sgn * (ix.sum()-npix))<=0:
            break
        else:
            npix = ypix.size
        iter += 1
    return ypix, xpix, lam, ix, code

def sourcery(mov: np.ndarray, ops):
    change_codes = True
    i0 = time.time()
    if isinstance(ops['diameter'], int):
        ops['diameter'] = [ops['diameter'], ops['diameter']]
    ops['diameter'] = np.array(ops['diameter'])
    ops['spatscale_pix'] = ops['diameter'][1]
    ops['aspect'] = ops['diameter'][0] / ops['diameter'][1]
    ops, U,sdmov, u   = getSVDdata(mov=mov, ops=ops) # get SVD components
    S, StU , StS = getStU(ops, U)
    Lyc, Lxc,nsvd = U.shape
    ops['Lyc'] = Lyc
    ops['Lxc'] = Lxc
    d0 = ops['diameter']
    sig = np.ceil(d0 / 4) # smoothing constant
    # make array of radii values of size (2*d0+1,2*d0+1)
    rs,dy,dx     = circleMask(d0)
    nsvd = U.shape[-1]
    nbasis = S.shape[-1]
    codes = np.zeros((0, nsvd), np.float32)
    LtU = np.zeros((0, nsvd), np.float32)
    LtS = np.zeros((0, nbasis), np.float32)
    L   = np.zeros((Lyc, Lxc, 0), np.float32)
    # regress maps onto basis functions and subtract neuropil contribution
    neu   = np.linalg.solve(StS, StU).astype('float32')
    Ucell = U - (S.reshape((-1,nbasis))@neu).reshape(U.shape)

    it = 0
    ncells = 0
    refine = -1

    # initialize
    ypix,xpix,lam = [], [], []

    while 1:
        if refine<0:
            V, us = getVmap(Ucell, sig)
            if it==0:
                vrem   = morphOpen(V, rs<=1.)
            V      = V - vrem # make V more uniform
            if it==0:
                V = V.astype('float64')
                # find indices of all maxima in +/- 1 range
                maxV   = filters.maximum_filter(V, footprint= np.ones((3,3)), mode='reflect')
                imax   = V > (maxV - 1e-10)
                peaks  = V[imax]
                # use the median of these peaks to decide if ROI is accepted
                thres  = ops['threshold_scaling'] * np.median(peaks[peaks>1e-4])
                ops['Vcorr'] = V
            V = np.minimum(V, ops['Vcorr'])

            # add extra ROIs here
            n = ncells
            while n<ncells+200:
                ind = np.argmax(V)
                i,j = np.unravel_index(ind, V.shape)
                if V[i,j] < thres:
                    break;
                yp, xp, la, ix, code = iter_extend(i, j, Ucell, us[i,j,:], change_codes=change_codes)
                codes = np.append(codes, np.expand_dims(code,axis=0), axis=0)
                ypix.append(yp)
                xpix.append(xp)
                lam.append(la)
                Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n,:])

                yp, xp = extendROI(yp,xp,Lyc,Lxc, int(np.mean(d0)))
                V[yp, xp] = 0
                n += 1
            newcells = len(ypix) - ncells
            if it==0:
                Nfirst = newcells
            L   = np.append(L, np.zeros((Lyc, Lxc, newcells), 'float32'), axis =-1)
            LtU = np.append(LtU, np.zeros((newcells, nsvd), 'float32'), axis = 0)
            LtS = np.append(LtS, np.zeros((newcells, nbasis), 'float32'), axis = 0)
            for n in range(ncells, len(ypix)):
                L[ypix[n],xpix[n], n] = lam[n]
                LtU[n,:] = lam[n] @ U[ypix[n],xpix[n],:]
                LtS[n,:] = lam[n] @ S[ypix[n],xpix[n], :]
            ncells +=newcells

            # regression with neuropil
            LtL = L.reshape((-1, ncells)).transpose() @ L.reshape((-1, ncells))
            cellcode = np.concatenate((LtL,LtS), axis=1)
            neucode  = np.concatenate((LtS.transpose(),StS), axis=1)
            codes = np.concatenate((cellcode, neucode), axis=0)
            Ucode = np.concatenate((LtU, StU),axis=0)
            codes = np.linalg.solve(codes + 1e-3*np.eye((codes.shape[0])), Ucode).astype('float32')
            neu   = codes[ncells:,:]
            codes = codes[:ncells,:]

        Ucell = U - (S.reshape((-1,nbasis))@neu + L.reshape((-1,ncells))@codes).reshape(U.shape)
        # reestimate masks
        n,k = 0,0
        while n < len(ypix):
            Ucell[ypix[n], xpix[n], :] += np.outer(lam[n], codes[k,:])
            ypix[n], xpix[n], lam[n], ix, codes[n,:] = iter_extend(ypix[n], xpix[n], Ucell,
                codes[k,:], refine, change_codes=change_codes)
            k+=1
            if ix.sum()==0:
                print('dropped ROI with no pixels')
                del ypix[n], xpix[n], lam[n]
                continue;
            Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n,:])
            n+=1
        codes = codes[:n, :]
        ncells = len(ypix)
        L   = np.zeros((Lyc,Lxc, ncells), 'float32')
        LtU = np.zeros((ncells, nsvd),   'float32')
        LtS = np.zeros((ncells, nbasis), 'float32')
        for n in range(ncells):
            L[ypix[n],xpix[n],n] = lam[n]
            if refine<0:
                LtU[n,:] = lam[n] @ U[ypix[n],xpix[n],:]
                LtS[n,:] = lam[n] @ S[ypix[n],xpix[n],:]
        err = (Ucell**2).mean()
        print('ROIs: %d, cost: %2.4f, time: %2.4f'%(ncells, err, time.time()-i0))

        it += 1
        if refine ==0:
            break
        if refine==2:
            # good place to get connected regions
            stat = [{'ypix':ypix[n], 'lam':lam[n], 'xpix':xpix[n]} for n in range(ncells)]
            stat = connected_region(stat, ops)
            # good place to remove ROIs that overlap, change ncells, codes, ypix, xpix, lam, L
            #stat, ix = remove_overlaps(stat, ops, Lyc, Lxc)
            #print('removed %d overlapping ROIs'%(len(ypix)-len(ix)))
            ypix = [stat[n]['ypix'] for n in range(len(stat))]
            xpix = [stat[n]['xpix'] for n in range(len(stat))]
            lam = [stat[n]['lam'] for n in range(len(stat))]
            #L = L[:,:,ix]
            #codes = codes[ix, :]
            ncells = len(ypix)
        if refine>0:
            Ucell = Ucell + (S.reshape((-1,nbasis))@neu).reshape(U.shape)
        if refine<0 and (newcells<Nfirst/10 or it==ops['max_iterations']):
            refine = 3
            U, sdmov = getSVDproj(mov, ops, u)
            Ucell = U
        if refine>=0:
            StU = S.reshape((Lyc*Lxc,-1)).transpose() @ Ucell.reshape((Lyc*Lxc,-1))
            #StU = np.reshape(S, (Lyc*Lxc,-1)).transpose() @ np.reshape(Ucell, (Lyc*Lxc, -1))
            neu = np.linalg.solve(StS, StU).astype('float32')
        refine -= 1
    Ucell = U - (S.reshape((-1,nbasis))@neu).reshape(U.shape)

    sdmov = np.reshape(sdmov, (Lyc, Lxc))
    ops['sdmov'] = sdmov
    stat  = [{'ypix':ypix[n], 'lam':lam[n]*sdmov[ypix[n], xpix[n]], 'xpix':xpix[n]} for n in range(ncells)]

    stat = postprocess(ops, stat, Ucell, codes)
    return ops, stat

def postprocess(ops, stat, Ucell, codes):
    # this is a good place to merge ROIs
    #mPix, mLam, codes = mergeROIs(ops, Lyc,Lxc,d0,mPix,mLam,codes,Ucell)
    stat = connected_region(stat, ops)
    stat = get_stat(ops, stat, Ucell, codes)
    stat = np.array(stat)
    return stat
