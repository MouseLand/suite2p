from contextlib import ExitStack
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from . import utils, rigid


def compute_motion_and_shift(data, maskMul, maskOffset, cfRefImg, maxregshift, smooth_sigma_time, reg_1p, spatial_hp, pre_smooth, ):
    """ register data matrix to reference image and shift

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    refAndMasks : list
        maskMul, maskOffset and cfRefImg (from prepare_refAndMasks)

    Returns
    -------
    data : int16 (or float32, if ops['nonrigid'])
        registered frames x Ly x Lx
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame
    """

    if smooth_sigma_time > 0:
        data_smooth = gaussian_filter1d(data, sigma=smooth_sigma_time, axis=0)

    # preprocessing for 1P recordings
    if reg_1p:
        if pre_smooth and pre_smooth % 2:
            raise ValueError("if set, pre_smooth must be a positive even integer.")
        if spatial_hp % 2:
            raise ValueError("spatial_hp must be a positive even integer.")
        data = data.astype(np.float32)

        if pre_smooth:
            data = utils.spatial_smooth(data_smooth if smooth_sigma_time > 0 else data, int(pre_smooth))
        data = utils.spatial_high_pass(data_smooth if smooth_sigma_time > 0 else data, int(spatial_hp))

    # rigid registration
    cfRefImg = cfRefImg.squeeze()

    ymax, xmax, cmax = rigid.phasecorr(
        data=data_smooth if smooth_sigma_time > 0 else data,
        maskMul=maskMul,
        maskOffset=maskOffset,
        cfRefImg=cfRefImg,
        maxregshift=maxregshift,
        smooth_sigma_time=smooth_sigma_time,
    )
    rigid.shift_data(data, ymax, xmax)

    return data, ymax, xmax, cmax

def compute_crop(xoff, yoff, corrXY, th_badframes, badframes, maxregshift, Ly, Lx):
    """ determines how much to crop FOV based on motion
    
    determines badframes which are frames with large outlier shifts
    (threshold of outlier is th_badframes) and
    it excludes these badframes when computing valid ranges
    from registration in y and x
    """
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy /= dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, 101)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
    ymax = Ly - ymin
    xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    xmax = Lx - xmin
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange


def pick_initial_reference(frames):
    """ computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg,Ly,Lx = frames.shape
    frames = np.reshape(frames, (nimg,-1)).astype('float32')
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))
    cc = np.matmul(frames, frames.T)
    ndiag = np.sqrt(np.diag(cc))
    cc = cc / np.outer(ndiag, ndiag)
    CCsort = -np.sort(-cc, axis = 1)
    bestCC = np.mean(CCsort[:, 1:20], axis=1);
    imax = np.argmax(bestCC)
    indsort = np.argsort(-cc[imax, :])
    refImg = np.mean(frames[indsort[0:20], :], axis = 0)
    refImg = np.reshape(refImg, (Ly,Lx))
    return refImg

