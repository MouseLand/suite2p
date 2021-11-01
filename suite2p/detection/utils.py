import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter

def square_mask(mask, ly, yi, xi):
    """ crop from mask a square of size ly at position yi,xi """
    Lyc, Lxc = mask.shape
    mask0 = np.zeros((2*ly, 2*ly), mask.dtype)
    yinds = [max(0, yi-ly), min(yi+ly, Lyc)]
    xinds = [max(0, xi-ly), min(xi+ly, Lxc)]        
    mask0[max(0, ly-yi) : min(2*ly, Lyc+ly-yi), 
          max(0, ly-xi) : min(2*ly, Lxc+ly-xi)] = mask[yinds[0]:yinds[1], xinds[0]:xinds[1]]
    return mask0

def mask_stats(mask):
    """ median and diameter of mask """
    y,x = np.nonzero(mask)
    y = y.astype(np.int32)
    x = x.astype(np.int32)
    ymed = np.median(y)
    xmed = np.median(x)
    imin = np.argmin((x-xmed)**2 + (y-ymed)**2)
    xmed = x[imin]
    ymed = y[imin]
    diam = len(y)**0.5
    diam /= (np.pi**0.5)/2
    return ymed, xmed, diam

def mask_ious(masks_true, masks_pred):
    """ return best-matched masks 
    
    Parameters
    ------------
    
    masks_true: ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: float, ND-array
        array of IOU pairs
    preds: int, ND-array
        array of matched indices
    iou_all: float, ND-array
        full IOU matrix across all pairs

    """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    iout, preds = match_masks(iou)
    return iout, preds, iou

def match_masks(iou):
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(iou.shape[0])
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(iou.shape[0], 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds
    

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def hp_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array 'mov' using a gaussian kernel.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The kernel width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered video
    """
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def hp_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns a high-pass-filtered copy of the 3D array 'mov' using a non-overlapping rolling mean kernel over time.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames

    """
    mov = mov.copy()
    for i in range(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


def temporal_high_pass_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """
    Returns hp-filtered mov over time, selecting an algorithm for computational performance based on the kernel width.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    width: int
        The filter width

    Returns
    -------
    filtered_mov: nImg x Ly x Lx
        The filtered frames
    """
    
    return hp_gaussian_filter(mov, width) if width < 10 else hp_rolling_mean_filter(mov, width)  # gaussian is slower
    

def standard_deviation_over_time(mov: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Returns standard deviation of difference between pixels across time, computed in batches of batch_size.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to filter
    batch_size: int
        The batch size

    Returns
    -------
    filtered_mov: Ly x Lx
        The statistics for each pixel
    """
    nbins, Ly, Lx = mov.shape
    batch_size = min(batch_size, nbins)
    sdmov = np.zeros((Ly, Lx), 'float32')
    for ix in range(0, nbins, batch_size):
        sdmov += ((np.diff(mov[ix:ix+batch_size, :, :], axis=0) ** 2).sum(axis=0))
    sdmov = np.maximum(1e-10, np.sqrt(sdmov / nbins))
    return sdmov


def downsample(mov: np.ndarray, taper_edge: bool = True) -> np.ndarray:
    """
    Returns a pixel-downsampled movie from 'mov', tapering the edges of 'taper_edge' is True.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to downsample
    taper_edge: bool
        Whether to taper the edges

    Returns
    -------
    filtered_mov:
        The downsampled frames
    """
    n_frames, Ly, Lx = mov.shape

    # bin along Y
    movd = np.zeros((n_frames, int(np.ceil(Ly / 2)), Lx), 'float32')
    movd[:, :Ly//2, :] = np.mean([mov[:, 0:-1:2, :], mov[:, 1::2, :]], axis=0)
    if Ly % 2 == 1:
        movd[:, -1, :] = mov[:, -1, :] / 2 if taper_edge else mov[:, -1, :]

    # bin along X
    mov2 = np.zeros((n_frames, int(np.ceil(Ly / 2)), int(np.ceil(Lx / 2))), 'float32')
    mov2[:, :, :Lx//2] = np.mean([movd[:, :, 0:-1:2], movd[:, :, 1::2]], axis=0)
    if Lx % 2 == 1:
        mov2[:, :, -1] = movd[:, :, -1] / 2 if taper_edge else movd[:, :, -1]

    return mov2


def threshold_reduce(mov: np.ndarray, intensity_threshold: float) -> np.ndarray:
    """
    Returns standard deviation of pixels, thresholded by 'intensity_threshold'.
    Run in a loop to reduce memory footprint.

    Parameters
    ----------
    mov: nImg x Ly x Lx
        The frames to downsample
    intensity_threshold: float
        The threshold to use

    Returns
    -------
    Vt: Ly x Lx
        The standard deviation of the non-thresholded pixels
    """
    nbinned, Lyp, Lxp = mov.shape
    Vt = np.zeros((Lyp,Lxp), 'float32')
    for t in range(nbinned):
        Vt += mov[t]**2 * (mov[t] > intensity_threshold)
    Vt = Vt**.5
    return Vt

