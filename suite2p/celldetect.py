import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils
import scipy.sparse as sparse
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

def getSVDdata(ops):
    i0 = tic()
    
    nframes = ops['nframes']
    bin_min = np.round(nframes / ops['navg_frames_svd']).astype('int32');
    bin_tau = np.round(ops['tau'] * ops['fs']).astype('int32');
    nt0 = max(bin_min, bin_tau)
    ops['navg_frames_svd'] = np.floor(nframes/nt0).astype('int32')
    
    Ly = ops['Ly']
    Lx = ops['Lx']
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    reg_file = open(ops['reg_file'], 'rb')
    
    nimgbatch = max(500, np.ceil(750 * ops['fs'])) 
    nimgbatch = nt0 * np.ceil(nimgbatch/nt0)
    
    nbytesread = np.int64(Ly*Lx*nimgbatch*2)
    mov = np.zeros((ops['navg_frames_svd'], Lyc, Lxc), np.float32)
    ix = 0
    # load and bin data
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        nimgd = int(np.floor(data.size / (Ly*Lx)))
        if nimgd == 0:
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

    nbins, Lyc, Lxc = np.shape(mov)
    
    sig = .5
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
    U = np.dot(u.transpose() , mov)        
    U = np.reshape(U, (-1,Lyc,Lxc))    
    
    return U, sdmov
    
def getStU(ops, U):    
    nbins, Lyc, Lxc = np.shape(U)        
    S = getNeuropilBasis(ops, Lyc, Lxc)
    # compute covariance of neuropil masks with spatial masks
    U = np.reshape(U, (-1,Lyc*Lxc))    
    StU = S @ U.transpose()
    StS = S @ S.transpose()
    U = np.reshape(U, (-1,Lyc,Lxc))    
    return S, StU , StS

def drawClusters(r,mPix,mLam,Ly,Lx):
    ncells = mPix.shape[1]
    r=np.random.random((ncells,))
    iclust = -1*np.ones((Ly,Lx),np.int32)
    Lam = np.zeros((Ly,Lx))
    H = np.zeros((Ly,Lx,1))
    for n in range(ncells):
        goodi   = np.array((mPix[:,n]>=0).nonzero()).astype(np.int32)
        goodi   = goodi.flatten()
        n0      = n*np.ones(goodi.shape,np.int32)
        lam     = mLam[goodi,n0]
        ipix    = mPix[mPix[:,n]>=0,n].astype(np.int32)
        if ipix is not None:
            ypix,xpix = np.unravel_index(ipix, (Ly,Lx))
            isingle = Lam[ypix,xpix]+1e-4 < lam
            ypix = ypix[isingle]
            xpix = xpix[isingle]
            Lam[ypix,xpix] = lam[isingle]
            iclust[ypix,xpix] = n*np.ones(ypix.shape)
            H[ypix,xpix,0] = r[n]*np.ones(ypix.shape)
        
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
    
    #sigy = 4*(Ly - 1)/ntiles
    #sigx = 4*(Lx - 1)/ntiles
    
    #for kx in range(ntiles):        
    #    for ky in range(ntiles):        
    #        cosy = 1 + np.cos(2*math.pi*(ys - yc[ky])/sigy)
    #        cosx = 1 + np.cos(2*math.pi*(xs - xc[kx])/sigx)
    #        cosy[abs(ys-yc[ky]) > sigy/2] = 0
    #        cosx[abs(xs-xc[kx]) > sigx/2] = 0
    #        S[ky,kx,:,:] = np.outer(cosy, cosx)
    
    S = np.reshape(S,(ntiles*ntiles, Ly*Lx))
    S = S / np.reshape(np.sum(S**2,axis=-1)**0.5, (-1,1))
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


def getStat(ops, Ly, Lx, d0, mPix, mLam, codes, Ucell):
    '''computes statistics of cells found using sourcery
    inputs:
        Ly, Lx, d0, mPix (pixels,ncells), mLam (weights,ncells), codes (ncells,nsvd), Ucell (nsvd,Ly,Lx)
    outputs:
        stat
        assigned to stat: ipix, ypix, xpix, med, npix, lam, footprint, compact, aspect_ratio, ellipse
    '''
    stat = {}
    rs,dy,dx = circleMask(d0)
    rsort = np.sort(rs.flatten())
    
    d0 = float(d0)
    rs    = rs[rs<=d0]
    frac = 0.5
    ncells = mPix.shape[0]
    footprints = np.zeros((ncells,))
    for n in range(0,ncells):
        stat[n] = {}
        goodi   = np.array(((mPix[n,:]>=0) & (mLam[n,:]>1e-10)).nonzero()).astype(np.int32)
        ipix    = mPix[n,goodi].astype(np.int32)
        ypix,xpix = np.unravel_index(ipix.astype(np.int32), (Ly,Lx))
        # pixels of cell in cropped (Ly,Lx) region of recording
        stat[n]['ypix'] = ypix + ops['yrange'][0]
        stat[n]['xpix'] = xpix + ops['xrange'][0]
        stat[n]['med']  = [np.median(ypix), np.median(xpix)]
        stat[n]['npix'] = ipix.size
        stat[n]['lam']  = mLam[n, goodi]
        # compute footprint of ROI
        y0,x0 = stat[n]['med']
        ypix, xpix, goodi = localRegion(y0,x0,dy,dx,Ly,Lx)
        proj  = codes[n,:] @ Ucell[:,ypix,xpix]
        rs0   = rs[goodi]
        inds  = proj.flatten()>proj.max()*frac
        stat[n]['footprint'] = np.mean(rs0[inds]) / d0
        footprints[n] = stat[n]['footprint']
        # compute compactness of ROI
        lam = mLam[n, :]
        r2 = (stat[n]['ypix']-stat[n]['med'][0])**2 + (stat[n]['xpix']-stat[n]['med'][1])**2        
        stat[n]['mrs']  = np.median(r2**.5) / d0
        stat[n]['mrs0'] = np.median(rsort) / d0
        stat[n]['compact'] = stat[n]['mrs'] / stat[n]['mrs0']
        
    mfoot = np.median(footprints)
    for n in range(ncells):
        stat[n]['footprint'] = stat[n]['footprint'] / mfoot
        
    return stat   


def getOverlaps(stat,Ly,Lx):
    '''computes overlapping pixels from ROIs in stat
    inputs:
        stat, Ly, Lx
    outputs:
        stat
        assigned to stat: overlap: (npix,1) boolean whether or not pixels also in another cell
    '''
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] = mask[ypix,xpix] + 1
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        stat[n]['overlap'] = mask[ypix,xpix] > 1
        
    return stat


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
            stat[n]['aspect_ratio'] = radius[0]/radius[1]
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
    us = gaussian_filter(Ucell, [0., sig, sig],  mode='wrap')
    
    # compute log variance at each location
    V  = (us**2).mean(axis=0)
    um = (Ucell**2).mean(axis=0)
    um = gaussian_filter(um, sig,  mode='wrap')
    V  = V / um
    V  = V.astype('float64')    
    return V, us

def sub2ind(array_shape, rows, cols):
    inds = rows * array_shape[1] + cols
    return inds
    
def sourcery(ops, U, S, StU, StS):           
    nsvd, Lyc, Lxc = U.shape

    ops['Lyc'] = Lyc
    ops['Lxc'] = Lxc

    d0 = ops['diameter']

    sig = np.ceil(d0 / 4) # smoothing constant

    # make array of radii values of size (2*d0+1,2*d0+1)

    rs,dy,dx     = circleMask(d0)

    ncell = int(1e4)
    mPix = -1*np.ones((ncell,  dx.size), np.int32)
    mLam = np.zeros((ncell, dx.size), np.float32)

    ncells = 0    

    nsvd = U.shape[0]
    nbasis = S.shape[0]
    LtU = np.zeros((0, nsvd), np.float32)
    LtS = np.zeros((0, nbasis), np.float32)

    Ucell = U

    # regress maps onto basis functions and subtract neuropil contribution
    neu   = np.linalg.solve(StS, StU).astype('float32')
    Ucell = U -  np.reshape(neu.transpose() @ S, U.shape)

    err = (Ucell**2).mean()    

    it = 0;
    i0 = tic()

    while it<ops['max_iterations']:
        V, us = getVmap(Ucell, sig)    
        # perform morphological opening on V to normalize brightness
        if it==0:        
            vrem   = morphOpen(V, rs<=d0)        
        V      = V - vrem

        if it==0:        
            # find indices of all maxima in +/- 1 range        
            maxV   = filters.maximum_filter(V, footprint= (rs<=1.5))
            imax   = V > (maxV - 1e-10)
            peaks  = V[imax]        
            # use the median of these peaks to decide if ROI is accepted
            thres  = ops['threshold_scaling'] * np.median(peaks[peaks>1e-4])
            ops['Vcorr'] = V

        V = np.minimum(V, ops['Vcorr'])

        # find local maxima in a +/- d0 neighborhood
        i,j  = localMax(V, rs<np.inf, thres)
        if i.size==0:
            break

        # svd values of cell peaks
        new_codes = us[:,i,j]
        new_codes = new_codes / np.sum(new_codes**2, axis=0)**0.5

        newcells = new_codes.shape[1]
        Lnew = sparse.lil_matrix((newcells,Lyc*Lxc),dtype=np.float32)   
        if it==0:
            L = sparse.lil_matrix((newcells,Lyc*Lxc),dtype=np.float32)   
        else:
            L = sparse.vstack([L, Lnew]).tolil()

        LtU = np.append(LtU, np.zeros((newcells, nsvd), 'float32'), axis = 0)
        LtS = np.append(LtS, np.zeros((newcells, nbasis), 'float32'), axis = 0)

        for n in range(ncells, ncells+newcells):
            ypix, xpix, goodi = localRegion(i[n-ncells],j[n-ncells],dy,dx,Lyc,Lxc)        
            usub = Ucell[:, ypix, xpix]
            lam = new_codes[:,n-ncells].transpose() @ usub
            lam[lam<lam.max()/5] = 0
            ipix = sub2ind((Lyc,Lxc), ypix, xpix)            

            mPix[n, goodi] = ipix        
            mLam[n, goodi] = lam
            mLam[n,:] = connectedRegion(mLam[n,:], rs, d0)
            mLam[n,:] = mLam[n,:] / np.sum(mLam[n,:]**2)**0.5

            # save lam in L, LtU, and LtS
            lam  = mLam[n, goodi]
            L[n,ipix] = lam        
            LtU[n,:] = U[:,ypix,xpix] @ lam
            LtS[n,:] = S[:,ipix] @ lam

        ncells += new_codes.shape[1]

        # regression with neuropil
        L = sparse.csr_matrix(L)
        LtL = (L @ L.transpose()).toarray()
        cellcode = np.concatenate((LtL,LtS), axis=1)
        neucode  = np.concatenate((LtS.transpose(),StS), axis=1)
        codes = np.concatenate((cellcode, neucode), axis=0)
        Ucode = np.concatenate((LtU, StU),axis=0)
        codes = np.linalg.solve(codes + 1e-3*np.eye((codes.shape[0])), Ucode).astype('float32')
        neu   = codes[ncells:,:]
        codes = codes[:ncells,:]

        Ucell = U - np.resize(neu.transpose() @ S, U.shape) - np.resize(codes.transpose() @ L, U.shape)

        # reestimate masks
        L = sparse.lil_matrix((ncells,Lyc*Lxc),dtype=np.float32)    
        for n in range(0,ncells):
            goodi   = np.array((mPix[n, :]>=0).nonzero()).astype(np.int32)            
            npix = goodi.size

            ipix    = mPix[n, goodi].astype(np.int32)
            ypix,xpix = np.unravel_index(ipix.astype(np.int32), (Lyc,Lxc))

            usub    = np.squeeze(Ucell[:,ypix,xpix]) + np.outer(codes[n,:], mLam[n, goodi])

            lam = codes[n,:] @ usub
            lam[lam<lam.max()/5] = 0

            mLam[n,goodi] = lam
            mLam[n,:]  = connectedRegion(mLam[n,:], rs, d0)
            mLam[n,:]  = mLam[n,:] / np.sum(mLam[n,:]**2)**0.5
            # save lam in L, LtU, and LtS
            lam = mLam[n, goodi]

            L[n,ipix] = lam            
            LtU[n,:]  = np.sum(U[:,ypix,xpix] * lam, axis = -1).squeeze()
            LtS[n,:]  = np.sum(S[:,ipix] * lam, axis = -1).squeeze()     

            lam0 = np.sum(usub * lam, axis = 1)
            A = usub - np.outer(lam0 , lam)
            Ucell[:,ypix,xpix] = np.reshape(A, (nsvd, -1, npix))

        err = (Ucell**2).mean()    

        print('cells: %d, cost: %2.4f, time: %2.4f'%(ncells, err, toc(i0)))

        if it==0:
            Nfirst = i.size    
        if newcells<Nfirst/10:
            break;
        it += 1
        
    mLam = mLam[:ncells,:]
    mLam = mLam / mLam.sum(axis=1).reshape(ncells,1)
    mPix = mPix[:ncells,:]   

    Ucell = U - np.resize(neu.transpose() @ S, U.shape) 
    # ypix, xpix, goodi = celldetect.localRegion(i[n-ncells],j[n-ncells],dy,dx,Ly,Lx)

    Ly = ops['Ly']
    Lx = ops['Lx']
    stat = getStat(ops, Lyc,Lxc,d0,mPix,mLam,codes,Ucell)    
    stat = getOverlaps(stat,Ly,Lx)
    
    stat,cell_pix,cell_masks = cellMasks(stat,Ly,Lx,False)
    neuropil_masks           = neuropilMasks(ops,stat,cell_pix)
    
    # add surround neuropil masks to stat
    for n in range(ncells):
        stat[n]['ipix_neuropil'] = neuropil_masks[n,:,:].flatten().nonzero();

    neuropil_masks = sparse.csc_matrix(np.resize(neuropil_masks,(-1,Ly*Lx)))
    cell_masks     = sparse.csc_matrix(np.resize(cell_masks,(-1,Ly*Lx)))
    neuropil_masks = neuropil_masks.transpose()
    cell_masks = cell_masks.transpose()

    return ops, stat, cell_masks, neuropil_masks, mPix, mLam

def extractF(ops, stat, cell_masks, neuropil_masks, mPix, mLam):
    nimgbatch = 1000
    nframes = int(ops['nframes'])
    Ly = ops['Ly']
    Lx = ops['Lx']
    ncells = cell_masks.shape[1]
    F    = np.zeros((ncells, nframes),np.float32)
    Fneu = np.zeros((ncells, nframes),np.float32)

    reg_file = open(ops['reg_file'], 'rb')
    nimgbatch = int(nimgbatch)
    block_size = Ly*Lx*nimgbatch*2
    ix = 0
    data = 1 
    
    while data is not None:
        buff = reg_file.read(block_size)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        nimgd = int(np.floor(data.size / (Ly*Lx)))
        if nimgd == 0:
            break
        data = np.reshape(data, (-1, Ly, Lx)).astype(np.float32)        

        # resize data to be Ly*Lx by nimgd
        data = np.reshape(data, (nimgd,-1))
        # compute cell activity
        inds = ix + np.arange(0,nimgd)

        F[:, inds]    = (data @ cell_masks).transpose()

        # compute neuropil activity
        Fneu[:, inds] = (data @ neuropil_masks).transpose()
        ix += nimgd
        if ix%(5*nimgd)==0:
            print('extracted %d/%d frames'%(ix,ops['nframes']))

    reg_file.close()
    return F, Fneu    

