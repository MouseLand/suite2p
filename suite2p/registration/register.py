import time
from contextlib import ExitStack
from os import path
from typing import Optional, Dict, Any
from warnings import warn

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from tqdm import tqdm

from suite2p import io
from suite2p.registration import bidiphase, utils, rigid, nonrigid


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


def register_binary(ops: Dict[str, Any], refImg=None, raw=True):
    """ main registration function

    if ops is a list of dictionaries, each will be registered separately

    Parameters
    ----------

    ops : dictionary or list of dicts
        'Ly', 'Lx', 'batch_size', 'align_by_chan', 'nonrigid'
        (optional 'keep_movie_raw', 'raw_file')

    refImg : 2D array (optional, default None)

    raw : bool (optional, default True)
        use raw_file for registration if available, if False forces reg_file to be used

    Returns
    --------

    ops : dictionary
        'nframes', 'yoff', 'xoff', 'corrXY', 'yoff1', 'xoff1', 'corrXY1', 'badframes'


    """

    if ops['pre_smooth'] and ops['pre_smooth'] % 2:
        raise ValueError("if set, pre_smooth must be a positive even integer.")
    if ops['spatial_hp_reg'] % 2:
        raise ValueError("spatial_hp must be a positive even integer.")

    # set number of frames and print warnings
    if ops['frames_include'] != -1:
        ops['nframes'] = min((ops['nframes'], ops['frames_include']))
    else:
        nbytes = path.getsize(ops['raw_file'] if ops.get('keep_movie_raw') and path.exists(ops['raw_file']) else ops['reg_file'])
        ops['nframes'] = int(nbytes / (2 * ops['Ly'] * ops['Lx']))

    print('registering %d frames'%ops['nframes'])
    if ops['nframes'] < 50:
        raise ValueError('the total number of frames should be at least 50.')
    if ops['nframes'] < 200:
        warn('number of frames is below 200, unpredictable behaviors may occur.')

    # get binary file paths
    if raw:
        raw = ops.get('keep_movie_raw') and 'raw_file' in ops and path.isfile(ops['raw_file'])
        if raw:
            if ops['nchannels'] > 1:
                if ops['functional_chan'] == ops['align_by_chan']:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = ops['raw_file'], ops['reg_file'], ops['raw_file_chan2'], ops['reg_file_chan2']
                else:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = ops['raw_file_chan2'], ops['reg_file_chan2'], ops['raw_file'], ops['reg_file']
            else:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = ops['raw_file'], ops['reg_file'], [], []
        else:
            if ops['nchannels'] > 1:
                if ops['functional_chan'] == ops['align_by_chan']:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = [], ops['reg_file'], [], ops['reg_file_chan2']
                else:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = [], ops['reg_file_chan2'], [], ops['reg_file']
            else:
                    raw_file_align, reg_file_align, raw_file_alt, reg_file_alt = [], ops['reg_file'], [], []
    bin_file = raw_file_align if raw else reg_file_align


    # compute reference image
    if refImg is not None:
        print('NOTE: user reference frame given')
    else:
        t0 = time.time()
        nframes = ops['nframes']
        nsamps = np.minimum(ops['nimg_init'], nframes)
        ix = np.linspace(0, nframes, 1 + nsamps).astype('int64')[:-1]
        frames = io.get_frames(
            Lx=ops['Lx'],
            Ly=ops['Ly'],
            xrange=ops['xrange'],
            yrange=ops['yrange'],
            ix=ix,
            bin_file=bin_file,
            crop=False
        )
        if ops['do_bidiphase'] and ops['bidiphase'] == 0:
            ops['bidiphase'] = bidiphase.compute(frames)
            print('NOTE: estimated bidiphase offset from data: %d pixels' % ops['bidiphase'])
        if ops['bidiphase'] != 0:
            bidiphase.shift(frames, int(ops['bidiphase']))

        refImg = pick_initial_reference(frames)

        niter = 8
        for iter in range(0, niter):

            if ops['1Preg']:
                refImg = refImg.astype(np.float32)
                refImg = refImg[np.newaxis, :, :]
                if ops['pre_smooth']:
                    refImg = utils.spatial_smooth(refImg, int(ops['pre_smooth']))
                refImg = utils.spatial_high_pass(refImg, int(ops['spatial_hp_reg']))
                refImg = refImg.squeeze()
            refImg = refImg.copy()

            ops['refImg'] = refImg
            maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
                refImg=refImg,
                maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
                smooth_sigma=ops['smooth_sigma'],
                pad_fft=ops['pad_fft'],
            )

            freg = frames
            if ops['smooth_sigma_time'] > 0:
                freg = gaussian_filter1d(freg, sigma=ops['smooth_sigma_time'], axis=0)
                freg = freg.astype(np.float32)

            # preprocessing for 1P recordings
            if ops['1Preg']:
                freg = freg.astype(np.float32)

                if ops['pre_smooth']:
                    freg = utils.spatial_smooth(freg, int(ops['pre_smooth']))
                freg = utils.spatial_high_pass(freg, int(ops['spatial_hp_reg']))

            # rigid registration
            ymax, xmax, cmax = rigid.phasecorr(
                data=freg,
                maskMul=maskMul,
                maskOffset=maskOffset,
                cfRefImg=cfRefImg.squeeze(),
                maxregshift=ops['maxregshift'],
                smooth_sigma_time=ops['smooth_sigma_time'],
            )
            for frame, dy, dx in zip(freg, ymax.flatten(), xmax.flatten()):
                rigid.shift_frame(frame=frame, dy=dy, dx=dx)

            ymax = ymax.astype(np.float32)
            xmax = xmax.astype(np.float32)
            isort = np.argsort(-cmax)
            nmax = int(frames.shape[0] * (1. + iter) / (2 * niter))
            refImg = freg[isort[1:nmax], :, :].mean(axis=0).astype(np.int16)
            dy, dx = -ymax[isort[1:nmax]].mean(), -xmax[isort[1:nmax]].mean()

            # shift data requires an array of shifts
            rigid.shift_frame(frame=refImg, dy=int(np.round(dy)), dx=int(np.round(dx)))

        print('Reference frame, %0.2f sec.'%(time.time()-t0))
    ops['refImg'] = refImg


    # register binary to reference image
    if ops['1Preg']:
        refImg = refImg.astype(np.float32)
        refImg = refImg[np.newaxis, :, :]
        if ops['pre_smooth']:
            refImg = utils.spatial_smooth(refImg, int(ops['pre_smooth']))
        refImg = utils.spatial_high_pass(refImg, int(ops['spatial_hp_reg']))
        refImg = refImg.squeeze()

    refImg = refImg.copy()

    maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
        refImg=refImg,
        maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
        smooth_sigma=ops['smooth_sigma'],
        pad_fft=ops['pad_fft'],
    )
    if ops.get('nonrigid'):
        if 'yblock' not in ops:
            ops['yblock'], ops['xblock'], ops['nblocks'], ops['maxregshiftNR'], ops['block_size'], ops[
                'NRsm'] = nonrigid.make_blocks(
                Ly=ops['Ly'], Lx=ops['Lx'], maxregshiftNR=ops['maxregshiftNR'], block_size=ops['block_size']
            )

        maskSlope = ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma']  # slope of taper mask at the edges
        if ops['1Preg']:
            data = refImg[np.newaxis, :, :]
            data = data.astype(np.float32)

            if ops['pre_smooth']:
                data = utils.spatial_smooth(data, int(ops['pre_smooth']))
            data = utils.spatial_high_pass(data, int(ops['spatial_hp_reg']))
            refImg = data.squeeze()

        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
            refImg0=refImg,
            maskSlope=maskSlope,
            smooth_sigma=ops['smooth_sigma'],
            yblock=ops['yblock'],
            xblock=ops['xblock'],
            pad_fft=ops['pad_fft'],
        )

    mean_img = np.zeros((ops['Ly'], ops['Lx']))
    rigid_offsets, nonrigid_offsets = [], []
    with io.BinaryFile(nbatch=ops['batch_size'], Ly=ops['Ly'], Lx=ops['Lx'], nframes=ops['nframes'],
                             reg_file=reg_file_align, raw_file=raw_file_align) as f:
        for k, data in tqdm(enumerate(f)):

            if ops['bidiphase'] and not ops['bidi_corrected']:
                bidiphase.shift(data, int(ops['bidiphase']))

            if ops['smooth_sigma_time'] > 0:
                data = gaussian_filter1d(data, sigma=ops['smooth_sigma_time'], axis=0)
                data = data.astype(np.float32)

            # preprocessing for 1P recordings
            if ops['1Preg']:
                data = data.astype(np.float32)

                if ops['pre_smooth']:
                    data = utils.spatial_smooth(data, int(ops['pre_smooth']))
                data = utils.spatial_high_pass(data, int(ops['spatial_hp_reg']))

            # rigid registration
            ymax, xmax, cmax = rigid.phasecorr(
                data=data,
                maskMul=maskMul,
                maskOffset=maskOffset,
                cfRefImg=cfRefImg.squeeze(),
                maxregshift=ops['maxregshift'],
                smooth_sigma_time=ops['smooth_sigma_time'],
            )
            for frame, dy, dx in zip(data, ymax.flatten(), xmax.flatten()):
                rigid.shift_frame(frame=frame, dy=dy, dx=dx)

            rigid_offsets.append([ymax, xmax, cmax])

            # non-rigid registration
            if ops['nonrigid']:

                if ops['smooth_sigma_time'] > 0:
                    data = gaussian_filter1d(data, sigma=ops['smooth_sigma_time'], axis=0)

                ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(
                    data=data,
                    maskMul=maskMulNR.squeeze(),
                    maskOffset=maskOffsetNR.squeeze(),
                    cfRefImg=cfRefImgNR.squeeze(),
                    snr_thresh=ops['snr_thresh'],
                    NRsm=ops['NRsm'],
                    xblock=ops['xblock'],
                    yblock=ops['yblock'],
                    maxregshiftNR=ops['maxregshiftNR'],
                )
                data = nonrigid.transform_data(
                    data=data,
                    nblocks=ops['nblocks'],
                    xblock=ops['xblock'],
                    yblock=ops['yblock'],
                    ymax1=ymax1,
                    xmax1=xmax1,
                )

                nonrigid_offsets.append([ymax1, xmax1, cmax1])

            mean_img += data.sum(axis=0) / ops['nframes']

            f.write(data)
            if ops['reg_tif']:
                fname = io.generate_tiff_filename(
                    functional_chan=ops['functional_chan'],
                    align_by_chan=ops['align_by_chan'],
                    save_path=ops['save_path'],
                    k=k,
                    ichan=True
                )
                io.save_tiff(data=data, fname=fname)

    rigid_offsets = list(np.array(rigid_offsets, dtype=np.float32).squeeze())
    nonrigid_offsets = list(np.array(nonrigid_offsets, dtype=np.float32).squeeze())

    mean_img_key = 'meanImg' if ops['nchannels'] == 1 or ops['functional_chan'] == ops['align_by_chan'] else 'meanImage_chan2'
    ops[mean_img_key] = mean_img

    if ops['nchannels'] > 1:
        t0 = time.time()
        mean_img_sum = np.zeros((ops['Ly'], ops['Lx']))
        nfr = 0
        with io.BinaryFile(nbatch=ops['batch_size'], Ly=ops['Ly'], Lx=ops['Lx'], nframes=ops['nframes'],
                        reg_file=reg_file_alt, raw_file=raw_file_alt) as f:

            for data in f:

                # get shifts
                nframes = data.shape[0]
                iframes = nfr + np.arange(0, nframes, 1, int)
                nfr += nframes
                ymax, xmax = rigid_offsets[0][iframes].astype(np.int32), rigid_offsets[1][iframes].astype(np.int32)

                # apply shifts
                if ops['bidiphase'] != 0 and not ops['bidi_corrected']:
                    bidiphase.shift(data, int(ops['bidiphase']))

                for frame, dy, dx in zip(data, ymax.flatten(), xmax.flatten()):
                    rigid.shift_frame(frame=frame, dy=dy, dx=dx)

                if ops['nonrigid']:
                    ymax1, xmax1 = nonrigid_offsets[0][iframes], nonrigid_offsets[1][iframes]
                    data = nonrigid.transform_data(data, nblocks=ops['nblocks'], xblock=ops['xblock'], yblock=ops['yblock'],
                                                   ymax1=ymax1, xmax1=xmax1)

                # write
                f.write(data)
                if ops['reg_tif_chan2']:
                    fname = io.generate_tiff_filename(
                        functional_chan=ops['functional_chan'],
                        align_by_chan=ops['align_by_chan'],
                        save_path=ops['save_path'],
                        k=k,
                        ichan=False
                    )
                    io.save_tiff(data=data, fname=fname)

                mean_img_sum += data.mean(axis=0)

        print('Registered second channel in %0.2f sec.' % (time.time() - t0))
        meanImg_key = 'meanImag' if ops['functional_chan'] != ops['align_by_chan'] else 'meanImg_chan2'
        ops[meanImg_key] = mean_img_sum / (k + 1)

    ops['yoff'], ops['xoff'], ops['corrXY'] = rigid_offsets
    if ops['nonrigid']:
        ops['yoff1'], ops['xoff1'], ops['corrXY1'] = nonrigid_offsets

    # compute valid region
    # ignore user-specified bad_frames.npy
    ops['badframes'] = np.zeros((ops['nframes'],), np.bool)
    if 'data_path' in ops and len(ops['data_path']) > 0:
        badfrfile = path.abspath(path.join(ops['data_path'][0], 'bad_frames.npy'))
        print('bad frames file path: %s'%badfrfile)
        if path.isfile(badfrfile):
            badframes = np.load(badfrfile)
            badframes = badframes.flatten().astype(int)
            ops['badframes'][badframes] = True
            print('number of badframes: %d'%ops['badframes'].sum())

    # return frames which fall outside range
    ops['badframes'], ops['yrange'], ops['xrange'] = compute_crop(
        xoff=ops['xoff'],
        yoff=ops['yoff'],
        corrXY=ops['corrXY'],
        th_badframes=ops['th_badframes'],
        badframes=ops['badframes'],
        maxregshift=ops['maxregshift'],
        Ly=ops['Ly'],
        Lx=ops['Lx'],
    )

    if not raw:
        ops['bidi_corrected'] = True

    return ops