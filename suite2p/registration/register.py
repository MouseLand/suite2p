from contextlib import ExitStack

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

def prepare_refAndMasks(refImg, ops):
    """ prepares refAndMasks for phasecorr using refImg

    Parameters
    ----------
    refImg : int16
        reference image

    ops : dictionary
        requires 'smooth_sigma'
        (if ```ops['1Preg']```, need 'spatial_taper', 'spatial_hp', 'pre_smooth')

    Returns
    -------
    refAndMasks : list
        maskMul, maskOffset, cfRefImg (see register.prepare_masks for details)

    """
    maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
        refImg0=refImg,
        spatial_taper=ops['spatial_taper'],
        smooth_sigma=ops['smooth_sigma'],
        pad_fft=ops['pad_fft'],
        reg_1p=ops['1Preg'],
        spatial_hp=ops['spatial_hp'],
        pre_smooth=ops['pre_smooth'],
    )
    if 'nonrigid' in ops and ops['nonrigid']:
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]
    return refAndMasks


def compute_motion_and_shift(data, bidiphase, bidi_corrected, refAndMasks, maxregshift, nblocks, xblock, yblock,
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

    if bidiphase and not bidi_corrected:
        bidiphase.shift(data, bidiphase)

    yxnr = []
    if smooth_sigma_time > 0:
        data_smooth = gaussian_filter1d(data, sigma=smooth_sigma_time, axis=0)

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(
        data=data_smooth if smooth_sigma_time > 0 else data,
        refAndMasks=refAndMasks[:3],
        maxregshift=maxregshift,
        reg_1p=reg_1p,
        spatial_hp=spatial_hp,
        pre_smooth=pre_smooth,
        smooth_sigma_time=smooth_sigma_time,
    )
    rigid.shift_data(data, ymax, xmax)

    # non-rigid registration
    if is_nonrigid and len(refAndMasks)>3:
        # preprocessing for 1P recordings
        if reg_1p:
            data = utils.one_photon_preprocess(
                data=data,
                spatial_hp=spatial_hp,
                pre_smooth=pre_smooth,
            )

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

def compute_crop(ops):
    """ determines how much to crop FOV based on motion
    
    determines ops['badframes'] which are frames with large outlier shifts
    (threshold of outlier is ops['th_badframes']) and
    it excludes these ops['badframes'] when computing valid ranges
    from registration in y and x

    Parameters
    ----------
    ops : dictionary
        'yoff', 'xoff', 'corrXY', 'badframes', 'maxregshift'

    Returns
    ----------
    ops : dictionary
        'badframes', 'yrange', 'xrange'


    """
    dx = ops['xoff'] - medfilt(ops['xoff'], 101)
    dy = ops['yoff'] - medfilt(ops['yoff'], 101)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy /= dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = ops['corrXY'] / medfilt(ops['corrXY'], 101)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    ops['badframes'] = np.logical_or(px > ops['th_badframes'] * 100, ops['badframes'])
    ops['badframes'] = np.logical_or(abs(ops['xoff']) > (ops['maxregshift'] * ops['Lx'] * 0.95), ops['badframes'])
    ops['badframes'] = np.logical_or(abs(ops['yoff']) > (ops['maxregshift'] * ops['Ly'] * 0.95), ops['badframes'])
    ymin = np.ceil(np.abs(ops['yoff'][np.logical_not(ops['badframes'])]).max())
    ymax = ops['Ly'] - ymin 
    xmin = np.ceil(np.abs(ops['xoff'][np.logical_not(ops['badframes'])]).max())
    xmax = ops['Lx'] - xmin
    # ymin = np.maximum(0, np.ceil(np.amax(ops['yoff'][np.logical_not(ops['badframes'])])))
    # ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(ops['yoff'])))
    # xmin = np.maximum(0, np.ceil(np.amax(ops['xoff'][np.logical_not(ops['badframes'])])))
    # xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(ops['xoff'])))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    return ops

def register_binary_to_ref(nbatch: int, Ly: int, Lx: int, nframes: int, ops, refAndMasks, reg_file_align, raw_file_align):
    offsets = init_offsets(nonrigid=ops['nonrigid'], n_blocks=ops['nblocks'])

    nbytesread = 2 * Ly * Lx * nbatch
    raw = len(raw_file_align) > 0

    sum_img = np.zeros((Ly, Lx))
    nfr = 0
    with open(reg_file_align, mode='wb' if raw else 'r+b') as reg_file_align, ExitStack() as stack:
        if raw:
            raw_file_align = stack.enter_context(open(raw_file_align, 'rb'))

        while True:
            data = np.frombuffer(
                raw_file_align.read(nbytesread) if raw else reg_file_align.read(nbytesread),
                dtype=np.int16,
                offset=0
            ).copy()
            if (data.size == 0) | (nfr >= nframes):
                break
            data = np.float32(np.reshape(data, (-1, Ly, Lx)))

            dout = compute_motion_and_shift(
                data=data,
                bidiphase=ops['bidiphase'],
                bidi_corrected=ops['bidi_corrected'],
                refAndMasks=refAndMasks,
                maxregshift=ops['maxregshift'],
                nblocks=ops['nblocks'],
                xblock=ops['xblock'],
                yblock=ops['yblock'],
                nr_sm=ops['NRsm'],
                snr_thresh=ops['snr_thresh'],
                smooth_sigma_time=ops['smooth_sigma_time'],
                maxregshiftNR=ops['maxregshiftNR'],
                is_nonrigid=ops['nonrigid'],
                reg_1p=ops['1Preg'],
                spatial_hp=ops['spatial_hp'],
                pre_smooth=ops['pre_smooth'],
            )

            # compile offsets (dout[1:])
            for n in range(len(dout) - 1):
                if n < 3:
                    offsets[n] = np.hstack((offsets[n], dout[n + 1]))
                else:
                    # add on nonrigid stats
                    for m in range(len(dout[-1])):
                        offsets[n + m] = np.vstack((offsets[n + m], dout[-1][m]))

            data = np.minimum(dout[0], 2**15 - 2)
            sum_img += data.sum(axis=0)
            data = data.astype('int16')

            # write to reg_file_align
            if not raw:
                reg_file_align.seek(-2*data.size,1)
            reg_file_align.write(bytearray(data))

            nfr += data.shape[0]

            yield ops, offsets, sum_img, data, nfr


def apply_shifts_to_binary(batch_size: int, Ly: int, Lx: int, nframes: int,
    is_nonrigid: bool, bidiphase_value: int, bidi_corrected, nblocks, xblock, yblock,
    offsets, reg_file_alt, raw_file_alt):
    """ apply registration shifts computed on one binary file to another
    
    Parameters
    ----------

    offsets : list of arrays
        shifts computed from reg_file_align/raw_file_align, 
        rigid shifts in Y are offsets[0] and in X are offsets[1], 
        nonrigid shifts in Y are offsets[3] and in X are offsets[4]

    reg_file_alt : string
        file to (read if raw_file_align empty, and) write registered binary to

    raw_file_align : string
        file to read raw binary from (if not empty)

    Returns
    -------
    ops : dictionary
        sets 'meanImg' or 'meanImg_chan2'
        
    """
    nbytesread = 2 * Ly * Lx * batch_size
    ix = 0
    meanImg = np.zeros((Ly, Lx))
    k=0
    raw = len(raw_file_alt) > 0
    with open(reg_file_alt, mode='wb' if raw else 'r+b') as reg_file_alt, ExitStack() as stack:
        if raw:
            raw_file_alt = stack.enter_context(open(raw_file_alt, 'rb'))
        while True:
            data = np.frombuffer(
                raw_file_alt.read(nbytesread) if raw else reg_file_alt.read(nbytesread),
                dtype=np.int16,
                offset=0,
            ).copy()
            if (data.size == 0) | (ix >= nframes):
                break
            data = np.reshape(data[:int(np.floor(data.shape[0] / Ly / Lx) * Ly * Lx)], (-1, Ly, Lx))
            nframes = data.shape[0]
            iframes = ix + np.arange(0, nframes, 1, int)

            # get shifts
            ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)
            ymax1, xmax1 = [], []
            if is_nonrigid:
                ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]

            # apply shifts
            if bidiphase_value != 0 and not bidi_corrected:
                bidiphase.shift(data, bidiphase_value)
            rigid.shift_data(data, ymax, xmax)
            if is_nonrigid:
                data = nonrigid.transform_data(data, nblocks=nblocks, xblock=xblock, yblock=yblock, ymax1=ymax1, xmax1=xmax1)
            data = np.minimum(data, 2 ** 15 - 2)
            meanImg += data.mean(axis=0)
            data = data.astype('int16')
            # write to binary
            if not raw:
                reg_file_alt.seek(-2 * data.size, 1)
            reg_file_alt.write(bytearray(data))

            ix += nframes
            yield meanImg, data


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


def iterative_alignment(ops, frames, refImg):
    """ iterative alignment of initial frames to compute reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top 20 correlated pairs is the
    inital reference frame returned

    Parameters
    ----------
    ops : dictionary
        requires 'nonrigid', 'smooth_sigma', 'bidiphase', '1Preg'

    frames : int16
        frames from binary (frames x Ly x Lx)

    refImg : int16
        initial reference image (Ly x Lx)

    Returns
    -------
    refImg : int16
        final reference image (Ly x Lx)

    """
    # do not reshift frames by bidiphase during alignment
    ops['bidiphase'] = 0
    niter = 8
    for iter in range(0,niter):
        ops['refImg'] = refImg
        maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
            refImg0=refImg,
            spatial_taper=ops['spatial_taper'],
            smooth_sigma=ops['smooth_sigma'],
            pad_fft=ops['pad_fft'],
            reg_1p=ops['1Preg'],
            spatial_hp=ops['spatial_hp'],
            pre_smooth=ops['pre_smooth'],
        )
        freg, ymax, xmax, cmax, yxnr = compute_motion_and_shift(
            data=frames,
            refAndMasks=[maskMul, maskOffset, cfRefImg],
            maxregshift=ops['maxregshift'],
            bidiphase=ops['bidiphase'],
            bidi_corrected=ops['bidi_corrected'],
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            nr_sm=ops['NRsm'],
            snr_thresh=ops['snr_thresh'],
            smooth_sigma_time=ops['smooth_sigma_time'],
            maxregshiftNR=ops['maxregshiftNR'],
            is_nonrigid=ops['nonrigid'],
            reg_1p=ops['1Preg'],
            spatial_hp=ops['spatial_hp'],
            pre_smooth=ops['pre_smooth'],
        )
        ymax = ymax.astype(np.float32)
        xmax = xmax.astype(np.float32)
        isort = np.argsort(-cmax)
        nmax = int(frames.shape[0] * (1.+iter)/(2*niter))
        refImg = freg[isort[1:nmax], :, :].mean(axis=0).squeeze().astype(np.int16)
        dy, dx = -ymax[isort[1:nmax]].mean(), -xmax[isort[1:nmax]].mean()
        # shift data requires an array of shifts
        dy = np.array([int(np.round(dy))])
        dx = np.array([int(np.round(dx))])
        rigid.shift_data(refImg, dy, dx)
        refImg = refImg.squeeze()
    return refImg


def compute_reference_image(ops, bin_file):
    """ compute the reference image

    computes initial reference image using ops['nimg_init'] frames

    Parameters
    ----------
    ops : dictionary
        requires 'nimg_init', 'nonrigid', 'smooth_sigma', 'bidiphase', '1Preg',
        'reg_file', (optional 'keep_movie_raw', 'raw_movie')

    bin_file : str
        location of binary file with data

    Returns
    -------
    refImg : int16
        initial reference image (Ly x Lx)

    """

    nframes = ops['nframes']
    nsamps = np.minimum(ops['nimg_init'], nframes)
    ix = np.linspace(0, nframes, 1+nsamps).astype('int64')[:-1]
    frames = utils.get_frames(
        Lx=ops['Lx'],
        Ly=ops['Ly'],
        xrange=ops['xrange'],
        yrange=ops['yrange'],
        ix=ix,
        bin_file=bin_file,
        crop=False,
        badframes=False
    )
    #frames = subsample_frames(ops, nFramesInit)
    if ops['do_bidiphase'] and ops['bidiphase']==0:
        bidi = bidiphase.compute(frames)
        print('NOTE: estimated bidiphase offset from data: %d pixels'%bidi)
    else:
        bidi = ops['bidiphase']
    if bidi != 0:
        bidiphase.shift(frames, bidi)
    refImg = pick_initial_reference(frames)
    refImg = iterative_alignment(ops, frames, refImg)
    return refImg, bidi


def init_offsets(nonrigid, n_blocks):
    """ initialize offsets for all frames """
    if nonrigid:
        nb = n_blocks[0] * n_blocks[1]
        return [
            np.zeros((0,), np.float32),  # yoff
            np.zeros((0,), np.float32),  # xoff
            np.zeros((0,), np.float32),  # corrXY
            np.zeros((0, nb), np.float32),  # yoff1
            np.zeros((0, nb), np.float32),  # xoff1
            np.zeros((0, nb), np.float32),  # corrXY1
        ]
    else:
        return [
            np.zeros((0,), np.float32),  # yoff
            np.zeros((0,), np.float32),  # xoff
            np.zeros((0,), np.float32),  # corrXY
        ]
