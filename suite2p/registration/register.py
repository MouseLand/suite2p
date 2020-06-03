from contextlib import ExitStack
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from . import bidiphase, nonrigid, utils, rigid


#HAS_GPU=False
#try:
#    import cupy as cp
#    from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
#    HAS_GPU=True
#except ImportError:
#    HAS_GPU=False


def compute_motion_and_shift(data, refAndMasks, maxregshift, nblocks, xblock, yblock,
                             nr_sm, snr_thresh, smooth_sigma_time, maxregshiftNR,
                             is_nonrigid, reg_1p, spatial_hp, pre_smooth,
                             ):
    """ register data matrix to reference image and shift

    need to run ```refAndMasks = register.prepare_refAndMasks(ops)``` to get fft'ed masks;
    if ```ops['nonrigid']``` need to run ```ops = nonrigid.make_blocks(ops)```

    Parameters
    ----------
    data : int16
        array that's frames x Ly x Lx
    refAndMasks : list
        maskMul, maskOffset and cfRefImg (from prepare_refAndMasks)
    ops : dictionary
        requires 'nonrigid', 'bidiphase', '1Preg'

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
    yxnr : list
        ymax, xmax and cmax from the non-rigid registration

    """

    yxnr = []
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
    maskMul, maskOffset, cfRefImg = refAndMasks[:3]
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

    # non-rigid registration
    if is_nonrigid and len(refAndMasks)>3:
        # preprocessing for 1P recordings
        if reg_1p:
            if pre_smooth and pre_smooth % 2:
                raise ValueError("if set, pre_smooth must be a positive even integer.")
            if spatial_hp % 2:
                raise ValueError("spatial_hp must be a positive even integer.")
            data = data.astype(np.float32)

            if pre_smooth:
                data = utils.spatial_smooth(data, int(pre_smooth))
            data = utils.spatial_high_pass(data, int(spatial_hp))

        ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(
            data=data_smooth if smooth_sigma_time > 0 else data,
            refAndMasks=refAndMasks[3:],
            snr_thresh=snr_thresh,
            NRsm=nr_sm,
            xblock=xblock,
            yblock=yblock,
            maxregshiftNR=maxregshiftNR,
        )
        yxnr = [ymax1, xmax1, cmax1]
        data = nonrigid.transform_data(
            data=data,
            nblocks=nblocks,
            xblock=xblock,
            yblock=yblock,
            ymax1=ymax1,
            xmax1=xmax1
        )
    return data, ymax, xmax, cmax, yxnr

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
    # ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(badframes)])))
    # ymax = Ly + np.minimum(0, np.floor(np.amin(yoff)))
    # xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(badframes)])))
    # xmax = Lx + np.minimum(0, np.floor(np.amin(xoff)))
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

