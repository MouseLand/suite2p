import time
from os import path
from typing import Dict, Any
from warnings import warn

import numpy as np
from scipy.signal import medfilt, medfilt2d

from .. import io
from . import bidiphase, utils, rigid, nonrigid


def compute_crop(xoff: int, yoff: int, corrXY, th_badframes, badframes, maxregshift, Ly: int, Lx:int):
    """ determines how much to crop FOV based on motion
    
    determines badframes which are frames with large outlier shifts
    (threshold of outlier is th_badframes) and
    it excludes these badframes when computing valid ranges
    from registration in y and x

    Parameters
    __________
    xoff: int
    yoff: int
    corrXY
    th_badframes
    badframes
    maxregshift
    Ly: int
        Height of a frame
    Lx: int
        Width of a frame

    Returns
    _______
    badframes
    yrange
    xrange
    """
    filter_window = min((len(yoff)//2)*2 - 1, 101)
    dx = xoff - medfilt(xoff, filter_window)
    dy = yoff - medfilt(yoff, filter_window)
    # offset in x and y (normed by mean offset)
    dxy = (dx**2 + dy**2)**.5
    dxy = dxy / dxy.mean()
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, filter_window)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    if badframes.mean() < 0.5:
        ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
        xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    else:
        warn('WARNING: >50% of frames have large movements, registration likely problematic')
        ymin = np.ceil(np.abs(yoff).max())
        xmin = np.ceil(np.abs(xoff).max())
    ymax = Ly - ymin
    xmax = Lx - xmin
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange


def pick_initial_reference(frames: np.ndarray):
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


def compute_reference(ops, frames):
    """ computes the reference image

    picks initial reference then iteratively aligns frames to create reference

    Parameters
    ----------
    
    ops : dictionary
        need registration options

    frames : 3D array, int16
        size [nimg_init x Ly x Lx], frames to use to create initial reference

    Returns
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    
    refImg = pick_initial_reference(frames)
    if ops['1Preg']:
        if ops['pre_smooth']:
            refImg = utils.spatial_smooth(refImg, int(ops['pre_smooth']))
            frames = utils.spatial_smooth(frames, int(ops['pre_smooth']))
        refImg = utils.spatial_high_pass(refImg, int(ops['spatial_hp_reg']))
        frames = utils.spatial_high_pass(frames, int(ops['spatial_hp_reg']))

    niter = 8
    for iter in range(0, niter):
        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(
                frames,
                *rigid.compute_masks(
                    refImg=refImg,
                    maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
                )
            ),
            cfRefImg=rigid.phasecorr_reference(
                refImg=refImg,
                smooth_sigma=ops['smooth_sigma'],
            ),
            maxregshift=ops['maxregshift'],
            smooth_sigma_time=ops['smooth_sigma_time'],
        )
        for frame, dy, dx in zip(frames, ymax, xmax):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

        nmax = int(frames.shape[0] * (1. + iter) / (2 * niter))
        isort = np.argsort(-cmax)[1:nmax]
        # reset reference image
        refImg = frames[isort].mean(axis=0).astype(np.int16)
        # shift reference image to position of mean shifts
        refImg = rigid.shift_frame(
            frame=refImg,
            dy=int(np.round(-ymax[isort].mean())),
            dx=int(np.round(-xmax[isort].mean()))
        )

    return refImg

def compute_reference_masks(refImg, ops=None):
    ### ------------- compute registration masks ----------------- ###

    maskMul, maskOffset = rigid.compute_masks(
        refImg=refImg,
        maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
    )
    cfRefImg = rigid.phasecorr_reference(
        refImg=refImg,
        smooth_sigma=ops['smooth_sigma'],
    )

    if ops.get('nonrigid'):
        if 'yblock' not in ops:
            ops['yblock'], ops['xblock'], ops['nblocks'], ops['block_size'], ops[
                'NRsm'] = nonrigid.make_blocks(Ly=ops['Ly'], Lx=ops['Lx'], block_size=ops['block_size'])

        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
            refImg0=refImg,
            maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'], # slope of taper mask at the edges
            smooth_sigma=ops['smooth_sigma'],
            yblock=ops['yblock'],
            xblock=ops['xblock'],
        )
    else:
        maskMulNR, maskOffsetNR, cfRefImgNR = [], [], []

    return maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR

def register_frames(refAndMasks, frames, ops=None):
    """ register frames to reference image 
    
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
    if len(refAndMasks)==6 or not isinstance(refAndMasks, np.ndarray):
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR = refAndMasks 
    else:
        refImg = refAndMasks
        if ops.get('norm_frames', False) and 'rmin' not in ops:
            ops['rmin'], ops['rmax'] = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
            refImg = np.clip(refImg, ops['rmin'], ops['rmax'])
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR = compute_reference_masks(refImg, ops)
        

    if ops['bidiphase'] and not ops['bidi_corrected']:
        bidiphase.shift(frames, int(ops['bidiphase']))

    fsmooth = frames.copy().astype(np.float32)
    if ops['smooth_sigma_time'] > 0:
        fsmooth = utils.temporal_smooth(data=fsmooth, sigma=ops['smooth_sigma_time'])

    # preprocessing for 1P recordings
    if ops['1Preg']:
        if ops['pre_smooth']:
            fsmooth = utils.spatial_smooth(fsmooth, int(ops['pre_smooth']))
        fsmooth = utils.spatial_high_pass(fsmooth, int(ops['spatial_hp_reg']))

    # rigid registration
    if ops.get('norm_frames', False):
        fsmooth = np.clip(fsmooth, ops['rmin'], ops['rmax'])
    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(data=fsmooth, maskMul=maskMul, maskOffset=maskOffset),
        cfRefImg=cfRefImg,
        maxregshift=ops['maxregshift'],
        smooth_sigma_time=ops['smooth_sigma_time'],
    )
    
    for frame, dy, dx in zip(frames, ymax, xmax):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    # non-rigid registration
    if ops['nonrigid']:
        # need to also shift smoothed data (if smoothing used)
        if ops['smooth_sigma_time'] or ops['1Preg']:
            for fsm, dy, dx in zip(fsmooth, ymax, xmax):
                fsm[:] = rigid.shift_frame(frame=fsm, dy=dy, dx=dx)
        else:
            fsmooth = frames.copy()

        if ops.get('norm_frames', False):
            fsmooth = np.clip(fsmooth, ops['rmin'], ops['rmax'])
            
        ymax1, xmax1, cmax1 = nonrigid.phasecorr(
            data=fsmooth,
            maskMul=maskMulNR.squeeze(),
            maskOffset=maskOffsetNR.squeeze(),
            cfRefImg=cfRefImgNR.squeeze(),
            snr_thresh=ops['snr_thresh'],
            NRsm=ops['NRsm'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            maxregshiftNR=ops['maxregshiftNR'],
        )

        frames = nonrigid.transform_data(
            data=frames,
            nblocks=ops['nblocks'],
            xblock=ops['xblock'],
            yblock=ops['yblock'],
            ymax1=ymax1,
            xmax1=xmax1,
        )
    else:
        ymax1, xmax1, cmax1 = None, None, None 
    
    return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1

def shift_frames(frames, yoff, xoff, yoff1, xoff1, ops=None):
    if ops['bidiphase'] != 0 and not ops['bidi_corrected']:
        bidiphase.shift(frames, int(ops['bidiphase']))
    
    for frame, dy, dx in zip(frames, yoff, xoff):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    if ops['nonrigid']:
        frames = nonrigid.transform_data(frames, nblocks=ops['nblocks'], xblock=ops['xblock'], yblock=ops['yblock'],
                                        ymax1=yoff1, xmax1=xoff1, bilinear=ops.get('bilinear_reg', True))
    return frames

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
    # set number of frames and print warnings
    if ops['frames_include'] != -1:
        ops['nframes'] = min((ops['nframes'], ops['frames_include']))
    else:
        nbytes = path.getsize(ops['raw_file'] if ops.get('keep_movie_raw') and path.exists(ops['raw_file']) else ops['reg_file'])
        ops['nframes'] = int(nbytes / (2 * ops['Ly'] * ops['Lx'])) # this equation is only true with int16 :)

    print('registering %d frames'%ops['nframes'])
    if ops['nframes'] < 50:
        raise ValueError('the total number of frames should be at least 50.')
    if ops['nframes'] < 200:
        print('WARNING: number of frames is below 200, unpredictable behaviors may occur.')

    # get binary file paths
    raw = raw and ops.get('keep_movie_raw') and 'raw_file' in ops and path.isfile(ops['raw_file'])
    reg_file_align = ops['reg_file'] if (ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan']) else ops['reg_file_chan2']
    if raw:
        raw_file_align = ops.get('raw_file') if (ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan']) else ops.get('raw_file_chan2')
    else:
        raw_file_align = None
        if ops['do_bidiphase'] and ops['bidiphase'] != 0:
            ops['bidi_corrected'] = True

    ### ----- compute and use bidiphase shift -------------- ###
    if refImg is None or (ops['do_bidiphase'] and ops['bidiphase'] == 0):
        # grab frames
        with io.BinaryFile(Lx=ops['Lx'], Ly=ops['Ly'], read_filename=raw_file_align if raw else reg_file_align) as f:
            frames = f[np.linspace(0, ops['nframes'], 1 + np.minimum(ops['nimg_init'], ops['nframes']), dtype=int)[:-1]]    
        # compute bidiphase shift
        if ops['do_bidiphase'] and ops['bidiphase'] == 0:
            ops['bidiphase'] = bidiphase.compute(frames)
            print('NOTE: estimated bidiphase offset from data: %d pixels' % ops['bidiphase'])
        # shift frames
        if refImg is None and ops['bidiphase'] != 0:
            bidiphase.shift(frames, int(ops['bidiphase'])) 

    if refImg is not None:
        print('NOTE: user reference frame given')
    else:
        t0 = time.time()
        refImg = compute_reference(ops, frames)
        print('Reference frame, %0.2f sec.'%(time.time()-t0))

    ops['refImg'] = refImg

    # normalize reference image
    refImg = ops['refImg'].copy()
    if ops.get('norm_frames', False):
        ops['rmin'], ops['rmax'] = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
        refImg = np.clip(refImg, ops['rmin'], ops['rmax'])

    refAndMasks = compute_reference_masks(refImg, ops)
    
    ### ------------- register binary to reference image ------------ ###

    mean_img = np.zeros((ops['Ly'], ops['Lx']))
    rigid_offsets, nonrigid_offsets = [], []
    with io.BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'],
                       read_filename=raw_file_align if raw_file_align else reg_file_align,
                       write_filename=reg_file_align) as f:
        t0 = time.time()
        for k, (_, frames) in enumerate(f.iter_frames(batch_size=ops['batch_size'])):
            frames, ymax, xmax, cmax, ymax1, xmax1, cmax1 = register_frames(refAndMasks, frames, ops)
            
            rigid_offsets.append([ymax, xmax, cmax])
            if ops['nonrigid']:
                nonrigid_offsets.append([ymax1, xmax1, cmax1])

            mean_img += frames.sum(axis=0) / ops['nframes']

            f.write(frames)
            if (ops['reg_tif'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_tif_chan2']):
                fname = io.generate_tiff_filename(
                    functional_chan=ops['functional_chan'],
                    align_by_chan=ops['align_by_chan'],
                    save_path=ops['save_path'],
                    k=k,
                    ichan=True
                )
                io.save_tiff(mov=frames, fname=fname)
            if (k+1)%4==0:
                print('Registered %d/%d in %0.2fs'%(min((k+1)*ops['batch_size'], ops['nframes']), ops['nframes'], time.time()-t0))

    ops['yoff'], ops['xoff'], ops['corrXY'] = utils.combine_offsets_across_batches(rigid_offsets, rigid=True)
    if ops['nonrigid']:
        ops['yoff1'], ops['xoff1'], ops['corrXY1'] = utils.combine_offsets_across_batches(nonrigid_offsets, rigid=False)
    mean_img_key = 'meanImg' if ops['nchannels'] == 1 or ops['functional_chan'] == ops['align_by_chan'] else 'meanImg_chan2'
    ops[mean_img_key] = mean_img

    if ops['nchannels'] > 1:
        reg_file_alt = ops['reg_file_chan2'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_file']
        raw_file_alt = ops.get('raw_file_chan2') if ops['functional_chan'] == ops['align_by_chan'] else ops.get('raw_file')
        raw_file_alt = raw_file_alt if raw else []

        t0 = time.time()
        mean_img_sum = np.zeros((ops['Ly'], ops['Lx']))
        with io.BinaryFile(Ly=ops['Ly'], Lx=ops['Lx'],
                           read_filename=raw_file_alt if raw_file_alt else reg_file_alt,
                           write_filename=reg_file_alt) as f:

            for k, (iframes, frames) in enumerate(f.iter_frames(batch_size=ops['batch_size'])):
                # apply shifts
                
                yoff, xoff = ops['yoff'][iframes].astype(int), ops['xoff'][iframes].astype(int)
                yoff1, xoff1 = None, None
                if ops['nonrigid']:
                    yoff1, xoff1 = ops['yoff1'][iframes], ops['xoff1'][iframes]

                frames = shift_frames(frames, yoff, xoff, yoff1, xoff1, ops)
                
                # write
                f.write(frames)
                if (ops['reg_tif_chan2'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_tif']):
                    fname = io.generate_tiff_filename(
                        functional_chan=ops['functional_chan'],
                        align_by_chan=ops['align_by_chan'],
                        save_path=ops['save_path'],
                        k=k,
                        ichan=False
                    )
                    io.save_tiff(mov=frames, fname=fname)

                mean_img_sum += frames.mean(axis=0)

        print('Registered second channel in %0.2f sec.' % (time.time() - t0))
        meanImg_key = 'meanImg' if ops['functional_chan'] != ops['align_by_chan'] else 'meanImg_chan2'
        ops[meanImg_key] = mean_img_sum / (k + 1)

    # compute valid region
    # ignore user-specified bad_frames.npy
    ops['badframes'] = np.zeros((ops['nframes'],), 'bool')
    if 'data_path' in ops and len(ops['data_path']) > 0:
        badfrfile = path.abspath(path.join(ops['data_path'][0], 'bad_frames.npy'))
        if path.isfile(badfrfile):
            print('bad frames file path: %s'%badfrfile)
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
    
    # add enhanced mean image
    ops = enhanced_mean_image(ops)

    return ops


def enhanced_mean_image(ops):
    """ computes enhanced mean image and adds it to ops

    Median filters ops['meanImg'] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops['meanImgE']

    Parameters
    ----------
    ops : dictionary
        uses 'meanImg', 'aspect', 'spatscale_pix', 'yrange' and 'xrange'

    Returns
    -------
        ops : dictionary
            'meanImgE' field added

    """

    I = ops['meanImg'].astype(np.float32)
    if 'spatscale_pix' not in ops:
        if isinstance(ops['diameter'], int):
            diameter = np.array([ops['diameter'], ops['diameter']])
        else:
            diameter = np.array(ops['diameter'])
        if diameter[0]==0:
            diameter[:] = 12
        ops['spatscale_pix'] = diameter[1]
        ops['aspect'] = diameter[0]/diameter[1]

    diameter = 4*np.ceil(np.array([ops['spatscale_pix'] * ops['aspect'], ops['spatscale_pix']])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
        ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg
    print('added enhanced mean image')
    return ops