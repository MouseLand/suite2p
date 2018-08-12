import numpy as np
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
from suite2p import utils
import time

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def get_sdmov(mov, ops):
    ix = 0
    batch_size = 500
    nbins, npix = mov.shape
    sdmov = np.zeros(npix, 'float32')
    while 1:
        inds = ix + np.arange(0,batch_size)
        inds = inds[inds<nbins]
        if inds.size==0:
            break
        sdmov += np.sum(np.diff(mov[inds, :], axis = 0)**2, axis = 0)
        ix = ix + batch_size
    sdmov = (sdmov/nbins)**0.5
    sdmov = np.maximum(1e-10,sdmov)
    #sdmov = np.mean(np.diff(mov, axis = 0)**2, axis = 0)**.5
    return sdmov

def getSVDproj(ops, u):
    mov = get_mov(ops)
    nbins, Lyc, Lxc = np.shape(mov)
    mov = np.reshape(mov, (-1,Lyc*Lxc))
    U = u.transpose() @ mov
    U = U.transpose().copy().reshape((Lyc,Lxc,-1))
    return U

def get_mov(ops):
    i0 = tic()

    nframes = ops['nframes']
    bin_min = np.round(nframes / ops['navg_frames_svd']).astype('int32');
    bin_min = max(bin_min, 1)
    bin_tau = np.round(ops['tau'] * ops['fs']).astype('int32');
    nt0 = max(bin_min, bin_tau)
    ops['navg_frames_svd'] = np.floor(nframes/nt0).astype('int32')

    Ly = ops['Ly']
    Lx = ops['Lx']
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    reg_file = open(ops['reg_file'], 'rb')

    nimgbatch = max(500, np.ceil(750 * ops['fs']))
    nimgbatch = min(nframes, nimgbatch)
    nimgbatch = nt0 * np.floor(nimgbatch/nt0)

    nbytesread = np.int64(Ly*Lx*nimgbatch*2)
    mov = np.zeros((ops['navg_frames_svd'], Lyc, Lxc), np.float32)
    ix = 0
    # load and bin data
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        nimgd = int(np.floor(data.size / (Ly*Lx)))
        if nimgd < nt0:
            break
        data = np.reshape(data, (-1, Ly, Lx)).astype(np.float32)
        # bin data
        if nimgd < nimgbatch:
            nmax = int(np.floor(nimgd / nt0) * nt0)
            data = data[:nmax,:,:]

        dbin = np.reshape(data, (-1,nt0,Ly,Lx))
        dbin = np.squeeze(dbin.mean(axis=1))
        dbin -= dbin.mean(axis=0)
        inds = ix + np.arange(0,dbin.shape[0])
        # crop into valid area
        mov[inds,:,:] = dbin[:, ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
        ix += dbin.shape[0]
    reg_file.close()

    return mov

def getSVDdata(ops):
    mov = get_mov(ops)
    nbins, Lyc, Lxc = np.shape(mov)

    sig = ops['diameter']/10.
    for j in range(nbins):
        mov[j,:,:] = ndimage.gaussian_filter(mov[j,:,:], sig)

    mov = np.reshape(mov, (-1,Lyc*Lxc))

    # compute noise variance across frames
    sdmov = get_sdmov(mov, ops)

    # normalize pixels by noise variance
    mov /= sdmov

    # compute covariance of binned frames
    cov = mov @ mov.transpose() / mov.shape[1]
    cov = cov.astype('float32')

    nsvd_for_roi = min(ops['nsvd_for_roi'], int(cov.shape[0]/2))
    u, s, v = np.linalg.svd(cov)

    u = u[:, :nsvd_for_roi]
    U = u.transpose() @ mov
    U = np.reshape(U, (-1,Lyc,Lxc))
    U = np.transpose(U, (1, 2, 0)).copy()
    return U, sdmov, u

def getStU(ops, U):
    Lyc, Lxc, nbins = np.shape(U)
    S = getNeuropilBasis(ops, Lyc, Lxc)
    # compute covariance of neuropil masks with spatial masks
    StU = S.reshape((Lyc*Lxc,-1)).transpose() @ U.reshape((Lyc*Lxc,-1))
    StS = S.reshape((Lyc*Lxc,-1)).transpose() @ S.reshape((Lyc*Lxc,-1))
    U = np.reshape(U, (-1,Lyc,Lxc))
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


def getNeuropilBasis(ops, Ly, Lx):
    ''' computes neuropil basis functions
        inputs:
            ops, Ly, Lx
            from ops: ratio_neuropil, tile_factor, diameter, neuropil_type
        outputs:
            basis functions (pixels x nbasis functions)
    '''
    ratio_neuropil = ops['ratio_neuropil']
    tile_factor    = ops['tile_factor']
    diameter       = ops['diameter']

    ntiles  = int(np.ceil(tile_factor * (Ly+Lx)/2 / (ratio_neuropil * diameter/2)))

    yc = np.linspace(1, Ly, ntiles)
    xc = np.linspace(1, Lx, ntiles)
    ys = np.arange(0,Ly)
    xs = np.arange(0,Lx)

    Kx = np.zeros((Lx, ntiles), 'float32')
    Ky = np.zeros((Ly, ntiles), 'float32')
    for k in range(ntiles):
        Ky[:,k] = np.cos(math.pi * (ys+0.5) *  k/Ly)
        Kx[:,k] = np.cos(math.pi * (xs+0.5) *  k/Lx)

    S = np.zeros((ntiles, ntiles, Ly, Lx), np.float32)
    for kx in range(ntiles):
        for ky in range(ntiles):
            S[ky,kx,:,:] = np.outer(Ky[:,ky], Kx[:,kx])

    S = np.reshape(S,(ntiles*ntiles, Ly*Lx))
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
    dx  = np.tile(np.arange(-d0,d0+1), (2*d0+1,1))
    dy  = dx.transpose()
    rs  = (dy**2 + dx**2) ** 0.5
    dx  = dx[rs<=d0]
    dy  = dy[rs<=d0]
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
    maxV = filters.maximum_filter(V, footprint=footprint)
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

def connectedRegion(mLam, rsmall, d0):
    mLam0 = np.zeros(rsmall.shape)
    mLam1 = np.zeros(rsmall.shape)
    # non-zero lam's
    mLam0[rsmall<=d0] = mLam>0
    mLam1[rsmall<=d0] = mLam

    mmax = mLam1.argmax()
    mask = np.zeros(rsmall.size)
    mask[mmax] = 1
    mask = np.resize(mask, (2*d0+1, 2*d0+1))

    for m in range(int(np.ceil(mask.shape[0]/2))):
        mask = filters.maximum_filter1d(mask, 3, axis=0) * mLam0
        mask = filters.maximum_filter1d(mask, 3, axis=1) * mLam0
        #mask = filters.maximum_filter(mask, footprint=rsmall<=1.5) * mLam0

    mLam *= mask[rsmall<=d0]
    return mLam

def pairwiseDistance(y,x):
    dists = ((np.expand_dims(y,axis=-1) - np.expand_dims(y,axis=0))**2
         + (np.expand_dims(x,axis=-1) - np.expand_dims(x,axis=0))**2)**0.5
    return dists

def convert_to_pix(mPix, mLam):
    stat = []
    ncells = mPix.shape[0]
    for k in range(0,ncells):
        stat0 = {}
        goodi   = np.array(((mPix[k,:]>=0) & (mLam[k,:]>1e-10)).nonzero()).astype(np.int32)
        ipix    = mPix[k,goodi].astype(np.int32)
        ypix,xpix = np.unravel_index(ipix.astype(np.int32), (Ly,Lx))
        stat0['ypix'] = ypix
        stat0['xpix'] = xpix
        stat0['med']  = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        stat0['npix'] = ipix.size
        stat0['lam']  = mLam[k, goodi]
        stat.append(stat0.copy())
    return stat

# this function needs to be updated with the new stat
def getStat(ops, stat, Ucell, codes):
    '''computes statistics of cells found using sourcery
    inputs:
        Ly, Lx, d0, mPix (pixels,ncells), mLam (weights,ncells), codes (ncells,nsvd), Ucell (nsvd,Ly,Lx)
    outputs:
        stat
        assigned to stat: ipix, ypix, xpix, med, npix, lam, footprint, compact, aspect_ratio, ellipse
    '''
    d0 = ops['diameter']
    Ly = ops['Lyc']
    Lx = ops['Lxc']
    rs,dy,dx = circleMask(d0)
    rsort = np.sort(rs.flatten())

    d0 = float(d0)
    rs    = rs[rs<=d0]
    frac = 0.5
    ncells = len(stat)
    footprints = np.zeros((ncells,))
    for k in range(0,ncells):
        stat0 = stat[k]
        ypix = stat0['ypix']
        xpix = stat0['xpix']
        lam = stat0['lam']
        # compute footprint of ROI
        y0 = np.median(ypix)
        x0 = np.median(xpix)
        yp, xp = extendROI(ypix,xpix,Ly,Lx, int(d0))
        rs0 = ((yp-y0)**2 + (xp-x0)**2)**.5

        proj  = Ucell[yp, xp, :] @ np.expand_dims(codes[k,:], axis=1)
        inds  = proj.flatten() > proj.max()*frac
        footprints[k] = np.mean(rs0[inds]) / d0

        # compute compactness of ROI
        r2 = ((ypix-y0)**2 + (xpix-x0)**2)**.5
        stat0['mrs']  = np.mean(r2) / d0
        stat0['mrs0'] = np.mean(rsort[:r2.size]) / d0
        stat0['compact'] = stat0['mrs'] / (1e-10+stat0['mrs0'])
        stat0['ypix'] += ops['yrange'][0]
        stat0['xpix'] += ops['xrange'][0]
        stat0['med']  = [np.median(stat0['ypix']), np.median(stat0['xpix'])]
        stat0['npix'] = xpix.size
    mfoot = np.median(footprints)
    for n in range(len(stat)):
        stat[n]['footprint'] = footprints[n] / mfoot
    return stat

def getOverlaps(stat, ops):
    '''computes overlapping pixels from ROIs in stat
    inputs:
        stat, Ly, Lx
    outputs:
        stat
        assigned to stat: overlap: (npix,1) boolean whether or not pixels also in another cell
    '''
    Ly = ops['Ly']
    Lx = ops['Lx']

    stat2 = []
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    k=0
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += + 1
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        stat[n]['overlap'] = mask[ypix,xpix] > 1
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        xpix = stat[n]['xpix'][~stat[n]['overlap']]
        if np.mean(stat[n]['overlap'])<1.: #ops['max_overlap']:
            stat2.append(stat[n])
            k+=1
    return stat2

def removeOverlaps(stat, ops, Ly, Lx):
    '''removes overlaps iteratively
    '''
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    ix = [k for k in range(ncells)]
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += 1

    while 1:
        O = np.zeros((ncells,1))
        for n in range(ncells):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            O[n] = np.mean(mask[ypix,xpix] > 1)
        i = np.argmax(O)
        if O[i]>ops['max_overlap']:
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            mask[ypix,xpix] -= 1
            ncells -= 1
            del stat[i], ix[i]
        else:
            break
    return stat, ix

def cellMasks(stat, Ly, Lx, allow_overlap):
    '''creates cell masks for ROIs in stat and computes radii
    inputs:
        stat, Ly, Lx, allow_overlap
            from stat: ipix, ypix, xpix, lam
            allow_overlap: boolean whether or not to include overlapping pixels in cell masks (default: False)
    outputs:
        stat, cell_pix (Ly,Lx), cell_masks (ncells,Ly,Lx)
            assigned to stat: radius (minimum of 3 pixels)
    '''
    ncells = len(stat)
    cell_pix = np.zeros((Ly,Lx))
    cell_masks = np.zeros((ncells,Ly,Lx), np.float32)
    for n in range(ncells):
        if allow_overlap:
            overlap = np.zeros((stat[n]['npix'],), bool)
        else:
            overlap = stat[n]['overlap']
        ypix = stat[n]['ypix'][~overlap]
        xpix = stat[n]['xpix'][~overlap]
        lam  = stat[n]['lam'][~overlap]
        if xpix.size:
            # compute radius of neuron (used for neuropil scaling)
            radius = utils.fitMVGaus(ypix,xpix,lam,2)[2]
            stat[n]['radius'] = radius[0]
            stat[n]['aspect_ratio'] = 2 * radius[0]/(.01 + radius[0] + radius[1])
            # add pixels of cell to cell_pix (pixels to exclude in neuropil computation)
            cell_pix[ypix[lam>0],xpix[lam>0]] += 1
            # add pixels to cell masks
            cell_masks[n, ypix, xpix] = lam / lam.sum()
        else:
            stat[n]['radius'] = 0
            stat[n]['aspect_ratio'] = 1
    cell_pix = np.minimum(1, cell_pix)
    return stat, cell_pix, cell_masks


def neuropilMasks(ops, stat, cell_pix):
    '''creates surround neuropil masks for ROIs in stat
    inputs:
        ops, stat, cell_pix
            from ops: inner_neuropil_radius, outer_neuropil_radius, min_neuropil_pixels, ratio_neuropil_to_cell
            from stat: med, radius
            cell_pix: (Ly,Lx) matrix in which non-zero elements indicate cells
    outputs:
        neuropil_masks (ncells,Ly,Lx)
    '''
    inner_radius = int(ops['inner_neuropil_radius'])
    outer_radius = ops['outer_neuropil_radius']
    # if outer_radius is infinite, define outer radius as a multiple of the cell radius
    if np.isinf(ops['outer_neuropil_radius']):
        min_pixels = ops['min_neuropil_pixels']
        ratio      = ops['ratio_neuropil_to_cell']
    # dilate the cell pixels by inner_radius to create ring around cells
    expanded_cell_pix = ndimage.grey_dilation(cell_pix, (inner_radius,inner_radius))

    ncells = len(stat)
    Ly = cell_pix.shape[0]
    Lx = cell_pix.shape[1]
    neuropil_masks = np.zeros((ncells,Ly,Lx),np.float32)
    x,y = np.meshgrid(np.arange(0,Lx),np.arange(0,Ly))
    for n in range(0,ncells):
        cell_center = stat[n]['med']
        if stat[n]['radius'] > 0:
            if np.isinf(ops['outer_neuropil_radius']):
                cell_radius  = stat[n]['radius']
                outer_radius = ratio * cell_radius
                npixels = 0
                # continue increasing outer_radius until minimum pixel value reached
                while npixels < min_pixels:
                    neuropil_on       = (((y - cell_center[1])**2 + (x - cell_center[0])**2)**0.5) <= outer_radius
                    neuropil_no_cells = neuropil_on - expanded_cell_pix > 0
                    npixels = neuropil_no_cells.astype(np.int32).sum()
                    outer_radius *= 1.25
            else:
                neuropil_on       = ((y - cell_center[0])**2 + (x - cell_center[1])**2)**0.5 <= outer_radius
                neuropil_no_cells = neuropil_on - expanded_cell_pix > 0
            npixels = neuropil_no_cells.astype(np.int32).sum()
            neuropil_masks[n,:,:] = neuropil_no_cells.astype(np.float32) / npixels
    return neuropil_masks

def getVmap(Ucell, sig):
    us = gaussian_filter(Ucell, [sig, sig, 0.],  mode='wrap')

    # compute log variance at each location
    V  = (us**2).mean(axis=-1)
    um = (Ucell**2).mean(axis=-1)
    um = gaussian_filter(um, sig,  mode='wrap')
    V  = V / um
    V  = V.astype('float64')
    return V, us

def sub2ind(array_shape, rows, cols):
    inds = rows * array_shape[1] + cols
    return inds


def minDistance(inputs):
    y1, x1, y2, x2 = inputs

    ds = (y1 - np.expand_dims(y2, axis=1))**2 + (x1 - np.expand_dims(x2, axis=1))**2

    return np.amin(ds)**.5

def mergeROIs(ops, Lyc,Lxc,d0,stat,codes,Ucell):
    # ROIs should be input as lists of pixel x and y coordinates + lam
    # how to compute pairwise shortest distance:
    # make a mask of all pixels in ROIs and their x,y coordinates and ROI index
    Lxc = ops['Lxc']
    Lyc = ops['Lyc']
    npix = 0
    for k in range(len(stat)):
        ipix = stat[k]['lam']>np.amax(stat[k]['lam'])/5
        npix += sum(ipix)
    ypix = np.zeros((npix,1), 'float32')
    xpix = np.zeros((npix,1), 'float32')
    cpix = np.zeros((npix,1), 'int32')

    k0 = 0
    for k in range(len(stat)):
        ipix = stat[k]['lam']>np.amax(stat[k]['lam'])/5
        npix = sum(ipix)
        ypix[k0 + np.arange(0,npix)] = stat[k]['ypix'][ipix]
        xpix[k0 + np.arange(0,npix)] = stat[k]['xpix'][ipix]
        cpix[k0 + np.arange(0,npix)] = k
        stat[k]['ix'] = k0 + np.arange(0,npix)
        y0[k], x0[k] = stat[k]['med']
        k0+=npix
    # iterate between finding nearest pixels and computing distances to them
    ds = (x0- xpix.expand_dims(axis=1))**2
    ds += (y0- ypix.expand_dims(axis=1))**2
    ds = ds**.5
    #for k in range(len(stat)):
#        for j in range(k+1,len(stat)):
#        imin = np.argmin(ds[:, stat[k]['ix']], axis=1)
#        x0[k] = 0

    # take minimum of the two distances (not necessarily commutative)
    # compute functional similarity
    # if anatomical distance < N pixels and similarity > 0.5, merge

    # merging combines all pixels, adding weights for duplicated pixels

    return mPix, mLam, codes

def extendROI(ypix, xpix, Ly, Lx,niter=1):
    for k in range(niter):
        yx = ((ypix, ypix, ypix, ypix-1, ypix+1), (xpix, xpix+1,xpix-1,xpix,xpix))
        yx = np.array(yx)
        yx = yx.reshape((2,-1))
        yu = np.unique(yx, axis=1)
        ix = np.all((yu[0]>=0, yu[0]<Ly, yu[1]>=0 , yu[1]<Lx), axis = 0)
        ypix,xpix = yu[:, ix]
    return ypix,xpix


def iter_extend(ypix, xpix, Ucell, code, flag=False):
    Lyc, Lxc, nsvd = Ucell.shape
    npix = 0
    while npix<10000:
        npix = ypix.size
        ypix, xpix = extendROI(ypix,xpix,Lyc,Lxc, 1)
        usub = Ucell[ypix, xpix, :]
        lam = usub @ np.expand_dims(code, axis=1)
        lam = np.squeeze(lam, axis=1)
        if flag:
            ix = lam>max(0, np.mean(lam)/3)
        else:
            ix = lam>max(0, lam.max()/5)

        if ix.sum()==0:
            break;
        ypix, xpix,lam = ypix[ix],xpix[ix], lam[ix]
        lam = lam/np.sum(lam**2+1e-10)**.5
        code = lam @ usub[ix, :]
        if npix>=ix.sum():
            break
        else:
            npix = ypix.size
    return ypix, xpix, lam, ix, code

def sourcery(ops):
    i0 = tic()
    U,sdmov, u      = getSVDdata(ops) # get SVD components
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

    ypix,xpix,lam = [], [], []

    while 1:
        if refine<0:
            V, us = getVmap(Ucell, sig)
            if it==0:
                vrem   = morphOpen(V, rs<=d0)
            V      = V - vrem # make V more uniform
            if it==0:
                V = V.astype('float64')
                # find indices of all maxima in +/- 1 range
                maxV   = filters.maximum_filter(V, footprint= (rs<=1.5))
                imax   = V > (maxV - 1e-10)
                peaks  = V[imax]
                # use the median of these peaks to decide if ROI is accepted
                thres  = ops['threshold_scaling'] * np.median(peaks[peaks>1e-4])
                ops['Vcorr'] = V
            V = np.minimum(V, ops['Vcorr'])

            # add extra ROIs here
            n = ncells
            while n<ncells+100:
                ind = np.argmax(V)
                i,j = np.unravel_index(ind, V.shape)
                if V[i,j] < thres:
                    break;
                yp, xp, la, ix, code = iter_extend(i, j, Ucell, us[i,j,:])
                codes = np.append(codes, np.expand_dims(code,axis=0), axis=0)
                ypix.append(yp)
                xpix.append(xp)
                lam.append(la)
                Ucell[ypix[n], xpix[n], :] -= np.outer(lam[n], codes[n,:])

                yp, xp = extendROI(yp,xp,Lyc,Lxc, int(d0))
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
            ypix[n], xpix[n], lam[n], ix, codes[n,:] = iter_extend(ypix[n], xpix[n], Ucell, codes[k,:])
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
        print('ROIs: %d, cost: %2.4f, time: %2.4f'%(ncells, err, toc(i0)))

        it += 1
        if refine ==0:
            break
        if refine>0:
            Ucell = Ucell + (S.reshape((-1,nbasis))@neu).reshape(U.shape)
        if refine<0 and (newcells<Nfirst/10 or it==ops['max_iterations']):
            refine = 3
            stat = [{'ypix':ypix[n], 'lam':lam[n], 'xpix':xpix[n]} for n in range(ncells)]
            # good place to remove ROIs that overlap, change ncells, codes, ypix, xpix, lam, L
            stat, ix = removeOverlaps(stat, ops, Lyc, Lxc)
            print('removed %d overlapping ROIs'%(len(ypix)-len(ix)))
            ypix = [stat[n]['ypix'] for n in range(len(stat))]
            xpix = [stat[n]['xpix'] for n in range(len(stat))]
            lam = [stat[n]['lam'] for n in range(len(stat))]
            codes = codes[ix, :]
            L = L[:,:,ix]
            ncells = len(ypix)
            U = getSVDproj(ops, u)
            Ucell = U
        if refine>=0:
            StU = S.reshape((Lyc*Lxc,-1)).transpose() @ Ucell.reshape((Lyc*Lxc,-1))
            #StU = np.reshape(S, (Lyc*Lxc,-1)).transpose() @ np.reshape(Ucell, (Lyc*Lxc, -1))
            neu = np.linalg.solve(StS, StU).astype('float32')
        refine -= 1

    Ucell = U - (S.reshape((-1,nbasis))@neu).reshape(U.shape)
    stat = [{'ypix':ypix[n], 'lam':lam[n], 'xpix':xpix[n]} for n in range(ncells)]

    stat = postprocess(ops, stat, Ucell, codes)
    return ops, stat

def postprocess(ops, stat, Ucell, codes):
    # this is a good place to merge ROIs
    #mPix, mLam, codes = mergeROIs(ops, Lyc,Lxc,d0,mPix,mLam,codes,Ucell)
    stat = getStat(ops, stat, Ucell, codes)
    stat = getOverlaps(stat,ops)
    return stat

def extractF(ops, stat):
    nimgbatch = 1000
    nframes = int(ops['nframes'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = len(stat)

    stat,cell_pix,cell_masks = cellMasks(stat,Ly,Lx,False)
    neuropil_masks           = neuropilMasks(ops,stat,cell_pix)
    # add surround neuropil masks to stat
    for n in range(ncells):
        stat[n]['ipix_neuropil'] = neuropil_masks[n,:,:].flatten().nonzero();
    neuropil_masks = np.reshape(neuropil_masks,(-1,Ly*Lx))
    cell_masks     = np.reshape(cell_masks,(-1,Ly*Lx))

    F    = np.zeros((ncells, nframes),np.float32)
    Fneu = np.zeros((ncells, nframes),np.float32)

    reg_file = open(ops['reg_file'], 'rb')
    nimgbatch = int(nimgbatch)
    block_size = Ly*Lx*nimgbatch*2
    ix = 0
    data = 1

    ops['meanImg'] = np.zeros((Ly,Lx))
    k0 = tic()
    while data is not None:
        buff = reg_file.read(block_size)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        nimgd = int(np.floor(data.size / (Ly*Lx)))
        if nimgd == 0:
            break
        data = np.reshape(data, (-1, Ly, Lx)).astype(np.float32)
        ops['meanImg'] += np.sum(data,axis=0)

        # resize data to be Ly*Lx by nimgd
        data = np.reshape(data, (nimgd,-1)).transpose()
        # compute cell activity
        inds = ix + np.arange(0,nimgd)
        F[:, inds]    = cell_masks @ data
        Fneu[:, inds] = neuropil_masks @ data


        if ix%(5*nimgd)==0:
            print('extracted %d/%d frames in %3.2f sec'%(ix,ops['nframes'], toc(k0)))
        ix += nimgd
    print('extracted %d/%d frames in %3.2f sec'%(ix,ops['nframes'], toc(k0)))
    ops['meanImg'] /= ops['nframes']

    reg_file.close()
    return F, Fneu, ops
