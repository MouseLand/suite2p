import numpy as np

from suite2p.detection.sparsedetect import extendROI


def get_overlaps(stat, ops):
    """ computes overlapping pixels from ROIs in stat
    
    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx'

    stat : array of dicts 
        'ypix', 'xpix'
        
    Returns
    ----------------

    stat : array of dicts
        adds 'overlap'

    """
    Ly, Lx = ops['Ly'], ops['Lx']
    ncells = len(stat)
    mask = np.zeros((Ly,Lx))
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += 1
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        stat[n]['overlap'] = mask[ypix,xpix] > 1.5
    return stat

def remove_overlappers(stat, ops, Ly, Lx):
    """ removes ROIs that are overlapping more than fraction ops['max_overlap'] with other ROIs
    
    Parameters
    ----------------

    stat : array of dicts
        'ypix', 'xpix'

    ops : dictionary
        'max_overlap'

    Returns
    ----------------

    stat : array of dicts
        ROIs with ops['max_overlap'] overlapped ROIs removed
    
    ix : list
        list of ROIs that were kept

    """
    ncells = len(stat)
    if not isinstance(stat, list):
        stat = list(stat)
    mask = np.zeros((Ly,Lx))
    ix = [k for k in range(ncells)]
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        mask[ypix,xpix] += 1
    while 1:
        O = np.zeros((len(stat),1))
        for n in range(len(stat)):
            ypix = stat[n]['ypix']
            xpix = stat[n]['xpix']
            O[n] = np.mean(mask[ypix,xpix] > 1.5)
        #i = np.argmax(O)
        inds = (O > ops['max_overlap']).nonzero()[0]
        if len(inds) > 0:
            i = np.max(inds)
            ypix = stat[i]['ypix']
            xpix = stat[i]['xpix']
            mask[ypix,xpix] -= 1
            del stat[i], ix[i]
        else:
            break
    return stat, ix

def circle_mask(d0):
    """ creates array (2*d0+1, 2*d0+1) with indices and radii from center point
        
    Parameters
    ----------
    d0 : int
        patch of (-d0,d0+1) over which radius computed

    Returns
    -------
        rs : array, float 
            (2*d0+1,2*d0+1) of radii

        dx,dy: indices in rs where the radius is less than d0
    """
    dx  = np.tile(np.arange(-d0[1],d0[1]+1), (2*d0[0]+1,1))
    dy  = np.tile(np.arange(-d0[0],d0[0]+1), (2*d0[1]+1,1))
    dy  = dy.transpose()

    rs  = (dy**2 + dx**2) ** 0.5
    return rs


def create_cell_masks(stat, Ly, Lx, allow_overlap=False):
    """ creates cell masks for ROIs in stat and computes radii

    Parameters
    ----------

    stat : dictionary
        'ypix', 'xpix', 'lam'

    Ly : float
        y size of frame

    Lx : float
        x size of frame

    allow_overlap : bool (optional, default False)
        whether or not to include overlapping pixels in cell masks

    Returns
    -------
    
    cell_pix : 2D array
        size [Ly x Lx] where 1 if pixel belongs to cell
    
    cell_masks : list 
        len ncells, each has tuple of pixels belonging to each cell and weights

    """

    ncells = len(stat)
    cell_pix = np.zeros((Ly,Lx))
    cell_masks = []

    for n in range(ncells):
        if allow_overlap:
            overlap = np.zeros((stat[n]['npix'],), bool)
        else:
            overlap = stat[n]['overlap']
        ypix = stat[n]['ypix'][~overlap]
        xpix = stat[n]['xpix'][~overlap]
        lam  = stat[n]['lam'][~overlap]
        if xpix.size:
            # add pixels of cell to cell_pix (pixels to exclude in neuropil computation)
            cell_pix[ypix[lam>0],xpix[lam>0]] += 1
            ipix = np.ravel_multi_index((ypix, xpix), (Ly,Lx)).astype('int')
            #utils.sub2ind((Ly,Lx), ypix, xpix)
            cell_masks.append((ipix, lam/lam.sum()))
        else:
            cell_masks.append((np.zeros(0).astype('int'), np.zeros(0)))

    cell_pix = np.minimum(1, cell_pix)
    return cell_pix, cell_masks

def create_neuropil_masks(ops, stat, cell_pix):
    """ creates surround neuropil masks for ROIs in stat by EXTENDING ROI (slow!)
    
    Parameters
    ----------

    ops : dictionary
        'inner_neuropil_radius', 'min_neuropil_pixels'

    stat : dictionary
        'ypix', 'xpix', 'lam'

    cellpix : 2D array
        1 if ROI exists in pixel, 0 if not; 
        pixels ignored for neuropil computation

    Returns
    -------

    neuropil_masks : 3D array
        size [ncells x Ly x Lx] where each pixel is weight of neuropil mask

    """

    ncells = len(stat)
    Ly = cell_pix.shape[0]
    Lx = cell_pix.shape[1]
    neuropil_masks = np.zeros((ncells,Ly,Lx), np.float32)
    for n in range(ncells):
        ypix = stat[n]['ypix']
        xpix = stat[n]['xpix']
        # first extend to get ring of dis-allowed pixels
        ypix, xpix = extendROI(ypix, xpix, Ly, Lx,ops['inner_neuropil_radius'])
        # count how many pixels are valid
        nring = np.sum(cell_pix[ypix,xpix]<.5)
        ypix1,xpix1 = ypix,xpix
        for j in range(0,100):
            ypix1, xpix1 = extendROI(ypix1, xpix1, Ly, Lx, 5) # keep extending
            if (np.sum(cell_pix[ypix1,xpix1]<.5) - nring) > ops['min_neuropil_pixels']:
                break # break if there are at least a minimum number of valid pixels
        ix = cell_pix[ypix1,xpix1]<.5
        ypix1, xpix1 = ypix1[ix], xpix1[ix]
        neuropil_masks[n,ypix1,xpix1] = 1.
        neuropil_masks[n,ypix,xpix] = 0
    S = np.sum(neuropil_masks, axis=(1,2))
    neuropil_masks /= S[:, np.newaxis, np.newaxis]
    return neuropil_masks
