import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
import utils

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
        
    sigy = 4*(Ly - 1)/ntiles
    sigx = 4*(Lx - 1)/ntiles
        
    S = np.zeros((Ly, Lx, ntiles, ntiles), np.float32)
    for kx in range(ntiles):        
        for ky in range(ntiles):        
            cosy = 1 + np.cos(2*math.pi*(ys - yc[ky])/sigy)
            cosx = 1 + np.cos(2*math.pi*(xs - xc[kx])/sigx)
            cosy[abs(ys-yc[ky]) > sigy/2] = 0
            cosx[abs(xs-xc[kx]) > sigx/2] = 0
            S[:,:,ky,kx] = np.expand_dims(cosy,axis=1) @ np.expand_dims(cosx,axis=1).transpose()
    
    S = np.reshape(S,(Ly*Lx, ntiles*ntiles))
    S = S / np.expand_dims(np.sum(np.abs(S)**2,axis=-1)**0.5,axis=1)
    S = S.transpose()
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
    ''' computes the morphological opening of V (correlation map) with (usually circular) footprint'''
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
    imax = (V > (maxV - 1e-10)) & (V > thres)
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
        mask = filters.maximum_filter(mask, footprint=rsmall<=float(d0)/4) * mLam0   
        
    mLam *= mask[rsmall<=d0]
    return mLam 


def pairwiseDistance(y,x):
    dists = ((np.expand_dims(y,axis=-1) - np.expand_dims(y,axis=0))**2
         + (np.expand_dims(x,axis=-1) - np.expand_dims(x,axis=0))**2)**0.5
    return dists


def getStat(Ly, Lx, d0, mPix, mLam, codes, Ucell):
    '''computes statistics of cells found using sourcery
    inputs:
        Ly, Lx, d0, mPix (pixels,ncells), mLam (weights,ncells), codes (ncells,nsvd), Ucell (nsvd,Ly,Lx)
    outputs:
        stat
        assigned to stat: ipix, ypix, xpix, med, npix, lam, footprint, compact, aspect_ratio, ellipse
    '''
    stat = {}
    rs,dy,dx = circleMask(d0)
    dists = pairwiseDistance(dy,dx)
    rsort = np.sort(rs.flatten())
    dists_radius = pairwiseDistance(rs.flatten(), rs.transpose().flatten())
    d0 = float(d0)
    rs    = rs[rs<=d0]
    frac = 0.5
    ncells = mPix.shape[1]
    footprints = np.zeros((ncells,))
    for n in range(0,ncells):
        stat[n] = {}
        goodi   = np.array(((mPix[:,n]>=0) & (mLam[:,n]>1e-10)).nonzero()).astype(np.int32)
        goodi   = goodi.flatten()
        n0      = n*np.ones(goodi.shape,np.int32)
        ipix    = mPix[goodi,n0].astype(np.int32)
        ypix,xpix = np.unravel_index(ipix.astype(np.int32), (Ly,Lx))
        # pixels of cell in cropped (Ly,Lx) region of recording
        stat[n]['ipix'] = ipix
        stat[n]['ypix'] = ypix
        stat[n]['xpix'] = xpix
        stat[n]['med']  = [np.median(ypix), np.median(xpix)]
        stat[n]['npix'] = ipix.size
        stat[n]['lam']  = mLam[goodi,n0]
        # compute footprint of ROI
        y0,x0 = stat[n]['med']
        ypix, xpix, goodi = localRegion(y0,x0,dy,dx,Ly,Lx)
        proj  = codes[n,:] @ Ucell[:,ypix,xpix]
        rs0   = rs[goodi]
        inds  = proj.flatten()>proj.max()*frac
        stat[n]['footprint'] = np.mean(rs0[inds]) / d0
        footprints[n] = stat[n]['footprint']
        # compute compactness of ROI
        lam = mLam[:,n]
        dd = dists[np.ix_(lam>1e-3, lam>1e-3)]
        stat[n]['mrs'] = dd.mean() / d0
        dd = dists_radius[np.ix_(lam>1e-3, lam>1e-3)]
        stat[n]['mrs0'] = dd.mean() / d0
        
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
        ipix = stat[n]['ipix'][~overlap]
        ypix = stat[n]['ypix'][~overlap]
        xpix = stat[n]['xpix'][~overlap]
        lam  = stat[n]['lam'][~overlap]
        if ipix.size:
            # compute radius of neuron (used for neuropil scaling)
            radius = utils.fitMVGaus(ypix,xpix,lam,2)[2]
            stat[n]['radius'] = radius[0]
            # add pixels of cell to cell_pix (pixels to exclude in neuropil computation)
            cell_pix[ypix[lam>0],xpix[lam>0]] += 1
            # add pixels to cell masks
            cell_masks[n*np.ones((lam.size),np.int32), ypix, xpix] = lam / lam.sum()
        else:
            stat[n]['radius'] = 0
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
    if outer_radius is np.inf:
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
            if outer_radius is np.inf:
                cell_radius  = stat[n]['radius']
                outer_radius = ratio * cell_radius
                npixels = 0
                # continue increasing outer_radius until minimum pixel value reached
                while npixels < min_pixels:
                    neuropil_on       = ((y - cell_center[1])**2 + (x - cell_center[0])**2)**0.5 <= outer_radius
                    neuropil_no_cells = neuropil_on - expanded_cell_pix > 0
                    npixels = neuropil_no_cells.astype(np.int32).sum()
                    outer_radius *= 1.25   
                neuropil_masks[n,:,:] = neuropil_no_cells.astype(np.float32) / npixels
            else:
                neuropil_on       = ((y - cell_center[0])**2 + (x - cell_center[1])**2)**0.5 <= outer_radius
                neuropil_no_cells = neuropil_on - expanded_cell_pix > 0
                neuropil_masks[n,:,:] = neuropil_no_cells.astype(np.float32) / npixels
                
    return neuropil_masks
