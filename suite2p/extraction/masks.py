import numpy as np
from .. import utils
from ..detection import sparsedetect

def create_cell_masks(ops, stat):
    '''creates cell masks for ROIs in stat and computes radii
    inputs:
        stat, Ly, Lx, allow_overlap
            from stat: ypix, xpix, lam
            allow_overlap: boolean whether or not to include overlapping pixels in cell masks (default: False)
    outputs:
        stat, cell_pix (Ly,Lx), cell_masks (ncells, Ly, Lx)
            assigned to stat: ipix (non-overlapping if chosen), radius (minimum of 3 pixels)
    '''
    Ly=ops['Ly']
    Lx=ops['Lx']
    ncells = len(stat)
    cell_pix = np.zeros((Ly,Lx))
    cell_masks = np.zeros((ncells,Ly,Lx), np.float32)
    for n in range(ncells):
        #if allow_overlap:
        overlap = np.zeros((stat[n]['npix'],), bool)
        ypix = stat[n]['ypix'][~overlap]
        xpix = stat[n]['xpix'][~overlap]
        lam  = stat[n]['lam'][~overlap]
        ipix = utils.sub2ind((Ly,Lx), stat[n]['ypix'], stat[n]['xpix'])
        stat[n]['ipix'] = ipix
        if xpix.size:
            # compute radius of neuron (used for neuropil scaling)
            if 'aspect' in ops:
                radius = utils.fitMVGaus(ypix/ops['aspect'], xpix, lam, 2)[2]
            else:
                radius = utils.fitMVGaus(ypix, xpix, lam, 2)[2]
            stat[n]['radius'] = radius[0]
            #stat[n]['radius'] = radius[0] * np.mean(ops['diameter'])
            stat[n]['aspect_ratio'] = 2 * radius[0]/(.01 + radius[0] + radius[1])
            # add pixels of cell to cell_pix (pixels to exclude in neuropil computation)
            cell_pix[ypix[lam>0],xpix[lam>0]] += 1
            cell_masks[n, ypix, xpix] = lam / lam.sum()
        else:
            stat[n]['radius'] = 0
            stat[n]['aspect_ratio'] = 1
    cell_pix = np.minimum(1, cell_pix)
    return stat, cell_pix , cell_masks

def circle_neuropil_masks(ops, stat, cell_pix):
    '''creates surround neuropil masks for ROIs in stat using gradually extending circles
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

def create_neuropil_masks(ops, stat, cell_pix):
    '''creates surround neuropil masks for ROIs in stat by EXTENDING ROI (SLOW!!)
    inputs:
        ops, stat, cell_pix
            from ops: inner_neuropil_radius, min_neuropil_pixels
            from stat: ypix, xpix
            cell_pix: (Ly,Lx) matrix in which non-zero elements indicate cells
    outputs:
        neuropil_masks (ncells,Ly,Lx)
    '''
    ncells = len(stat)
    Ly = cell_pix.shape[0]
    Lx = cell_pix.shape[1]
    neuropil_masks = np.zeros((ncells,Ly,Lx), np.float32)
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        # first extend to get ring of dis-allowed pixels
        ypix, xpix = sparsedetect.extendROI(ypix, xpix, Ly, Lx,ops['inner_neuropil_radius'])
        # count how many pixels are valid
        nring = np.sum(cell_pix[ypix,xpix]<.5)
        ypix1,xpix1 = ypix,xpix
        for j in range(0,100):
            ypix1, xpix1 = sparsedetect.extendROI(ypix1, xpix1, Ly, Lx, 5) # keep extending
            if np.sum(cell_pix[ypix1,xpix1]<.5)-nring>ops['min_neuropil_pixels']:
                break # break if there are at least a minimum number of valid pixels

        ix = cell_pix[ypix1,xpix1]<.5
        ypix1, xpix1 = ypix1[ix], xpix1[ix]
        neuropil_masks[n,ypix1,xpix1] = 1.
        neuropil_masks[n,ypix,xpix] = 0
    S = np.sum(neuropil_masks, axis=(1,2))
    neuropil_masks /= S[:, np.newaxis, np.newaxis]
    return neuropil_masks

#from numba import njit, float32, prange, boolean
#@njit((float32[:,:], int32[:], float32[:]), parallel=True)
#def matmul_index(X, inds, Y):
#    for t in prange(inds.shape[0]):
#        Y[n] = X[inds[:], t]
