import time
from os import path
from typing import Dict, Any
from warnings import warn

import numpy as np
from scipy.signal import medfilt, medfilt2d

from .. import io, default_ops
from . import bidiphase as bidi
from . import utils, rigid, nonrigid

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


def compute_reference(frames, ops=default_ops()):
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

        nmax = max(2, int(frames.shape[0] * (1. + iter) / (2 * niter)))
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

def compute_reference_masks(refImg, ops=default_ops()):
    ### ------------- compute registration masks ----------------- ###
    if isinstance(refImg, list):
        refAndMasks_all = []
        for rimg in refImg:
            refAndMasks = compute_reference_masks(rimg)
            refAndMasks_all.append(refAndMasks)
        return refAndMasks_all
    else:
        maskMul, maskOffset = rigid.compute_masks(
            refImg=refImg,
            maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
        )
        cfRefImg = rigid.phasecorr_reference(
            refImg=refImg,
            smooth_sigma=ops['smooth_sigma'],
        )
        Ly, Lx = refImg.shape
        blocks = []
        if ops.get('nonrigid'):
            blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops['block_size'])

            maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
                refImg0=refImg,
                maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'], # slope of taper mask at the edges
                smooth_sigma=ops['smooth_sigma'],
                yblock=blocks[0],
                xblock=blocks[1],
            )
        else:
            maskMulNR, maskOffsetNR, cfRefImgNR = [], [], [] 

        return maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks

def register_frames(refAndMasks, frames, rmin=-np.inf, rmax=np.inf, bidiphase=0, ops=default_ops(), nZ=1):
    """ register frames to reference image 
    
    Parameters
    ----------

    refAndMasks : list of processed reference images and masks, or 2D array of reference image

    frames : np.ndarray, np.int16 or np.float32
        time x Ly x Lx

    rmin : clip frames at rmin

    rmax : clip frames at rmax


    Returns
    --------

    ops : dictionary
        'nframes', 'yoff', 'xoff', 'corrXY', 'yoff1', 'xoff1', 'corrXY1', 'badframes'


    """

    if nZ > 1:
        cmax_best = -np.inf * np.ones(len(frames), 'float32')
        cmax_all = -np.inf * np.ones((len(frames), nZ), 'float32')
        zpos_best = np.zeros(len(frames), 'int')
        run_nonrigid = ops['nonrigid']
        for z in range(nZ):
            ops['nonrigid'] = False
            outputs = register_frames(refAndMasks[z], frames.copy(), rmin=rmin[z], rmax=rmax[z], 
                                        bidiphase=bidiphase, ops=ops, nZ=1)
            cmax_all[:,z] = outputs[3]
            if z==0:
                outputs_best = list(outputs[:-4]).copy() 
            ibest = cmax_best < cmax_all[:,z]
            zpos_best[ibest] = z
            cmax_best[ibest] = cmax_all[ibest, z]
            for i, (output_best, output) in enumerate(zip(outputs_best, outputs[:-4])):
                output_best[ibest] = output[ibest]
        if run_nonrigid:
            ops['nonrigid'] = True
            nfr = frames.shape[0]
            for i,z in enumerate(zpos_best):
                outputs = register_frames(refAndMasks[z], frames[[i]], rmin=rmin[z], rmax=rmax[z],
                                            bidiphase=bidiphase, ops=ops, nZ=1)
                
                if i==0:
                    outputs_best = []
                    for output in outputs[:-1]:
                        outputs_best.append(np.zeros((nfr, *output.shape[1:]), dtype=output.dtype))
                        outputs_best[-1][0] = output[0]
                else:
                    for output, output_best in zip(outputs[:-1], outputs_best):
                        output_best[i] = output[0]
        frames, ymax, xmax, cmax, ymax1, xmax1, cmax1 = outputs_best
        return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, (zpos_best, cmax_all)
    else:    
        if len(refAndMasks)==7 or not isinstance(refAndMasks, np.ndarray):
            maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = refAndMasks 
        else:
            refImg = refAndMasks
            if ops.get('norm_frames', False) and 'rmin' not in ops:
                rmin, rmax = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
                refImg = np.clip(refImg, rmin, rmax)
            maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = compute_reference_masks(refImg, ops)
        
        if bidiphase != 0:
            bidi.shift(frames, bidiphase)

        
        # if smoothing or filtering or clipping to compute registration shifts, make a copy of the frames
        dtype = 'float32' if ops['smooth_sigma_time'] > 0 or ops['1Preg'] else frames.dtype
        fsmooth = frames.copy().astype(dtype) if ops['smooth_sigma_time'] > 0 or ops['1Preg'] else frames
        
        if ops['smooth_sigma_time']:
            fsmooth = utils.temporal_smooth(data=fsmooth, sigma=ops['smooth_sigma_time'])
        else:
            fsmooth = frames

        # preprocessing for 1P recordings
        if ops['1Preg']:
            if ops['pre_smooth']:
                fsmooth = utils.spatial_smooth(fsmooth, int(ops['pre_smooth']))
            fsmooth = utils.spatial_high_pass(fsmooth, int(ops['spatial_hp_reg']))

        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(data=np.clip(fsmooth, rmin, rmax) if rmin>-np.inf else fsmooth, 
                                    maskMul=maskMul, maskOffset=maskOffset),
            cfRefImg=cfRefImg,
            maxregshift=ops['maxregshift'],
            smooth_sigma_time=ops['smooth_sigma_time'],
        )
        
        for frame, dy, dx in zip(frames, ymax, xmax):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)
        
        # non-rigid registration
        if ops['nonrigid']:
            # need to also shift smoothed/filtered data
            if ops['smooth_sigma_time'] or ops['1Preg']:
                for fsm, dy, dx in zip(fsmooth, ymax, xmax):
                    fsm[:] = rigid.shift_frame(frame=fsm, dy=dy, dx=dx)
                    
            ymax1, xmax1, cmax1 = nonrigid.phasecorr(
                data=np.clip(fsmooth, rmin, rmax) if rmin>-np.inf else fsmooth,
                maskMul=maskMulNR.squeeze(),
                maskOffset=maskOffsetNR.squeeze(),
                cfRefImg=cfRefImgNR.squeeze(),
                snr_thresh=ops['snr_thresh'],
                NRsm=blocks[-1],
                xblock=blocks[1],
                yblock=blocks[0],
                maxregshiftNR=ops['maxregshiftNR'],
            )

            frames = nonrigid.transform_data(
                data=frames,
                yblock=blocks[0],
                xblock=blocks[1],
                nblocks=blocks[2],
                ymax1=ymax1,
                xmax1=xmax1,
            )
        else:
            ymax1, xmax1, cmax1 = None, None, None 
        
        return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, None

def shift_frames(frames, yoff, xoff, yoff1, xoff1, blocks=None, ops=default_ops()):
    if ops['bidiphase'] != 0 and not ops['bidi_corrected']:
        bidi.shift(frames, int(ops['bidiphase']))
    
    for frame, dy, dx in zip(frames, yoff, xoff):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    if ops['nonrigid']:
        frames = nonrigid.transform_data(frames, yblock=blocks[0], xblock=blocks[1], nblocks=blocks[2],
                                         ymax1=yoff1, xmax1=xoff1, bilinear=ops.get('bilinear_reg', True))
    return frames

def normalize_reference_image(refImg):
    if isinstance(refImg, list):
        rmins = []
        rmaxs = []
        for rimg in refImg:
            rmin, rmax = np.int16(np.percentile(rimg,1)), np.int16(np.percentile(rimg,99))
            rimg[:] = np.clip(rimg, rmin, rmax)
            rmins.append(rmin)
            rmaxs.append(rmax)
        return refImg, rmins, rmaxs
    else:
        rmin, rmax = np.int16(np.percentile(refImg,1)), np.int16(np.percentile(refImg,99))
        refImg = np.clip(refImg, rmin, rmax)
        return refImg, rmin, rmax

def compute_reference_and_register_frames(f_align_in, f_align_out=None, refImg=None, ops=default_ops()):
    """ compute reference frame, if refImg is None, and align frames in f_align_in to reference 
    
    if f_align_out is not None, registered frames are written to f_align_out

    f_align_in, f_align_out can be a BinaryRWFile or any type of array that can be slice-indexed
    
    """
    
    n_frames, Ly, Lx = f_align_in.shape
    
    batch_size = ops['batch_size']
    ### ----- compute reference image and bidiphase shift -------------- ###
    if refImg is None:
        # grab frames
        frames = f_align_in[np.linspace(0, n_frames, 1 + np.minimum(ops['nimg_init'], n_frames), dtype=int)[:-1]]    
        # compute bidiphase shift
        if ops['do_bidiphase'] and ops['bidiphase'] == 0 and not ops['bidi_corrected']:
            bidiphase = bidi.compute(frames)
            print('NOTE: estimated bidiphase offset from data: %d pixels' % bidiphase)
            ops['bidiphase'] = bidiphase
            # shift frames
            if bidiphase != 0:
                bidi.shift(frames, int(ops['bidiphase']))
        else:
            bidiphase = 0

        if refImg is None:
            t0 = time.time()
            refImg = compute_reference(frames, ops=ops)
            print('Reference frame, %0.2f sec.'%(time.time()-t0))
    
    if isinstance(refImg, list):
        nZ = len(refImg)
        print(f'List of reference frames len = {nZ}')
    else:
        nZ = 1

    # normalize reference image
    refImg_orig = refImg.copy()
    if ops.get('norm_frames', False):
        refImg, rmin, rmax = normalize_reference_image(refImg)
    else:
        if nZ==1:
            rmin, rmax = -np.inf, np.inf
        else:
            rmin = -np.inf * np.ones(nZ)
            rmax = np.inf * np.ones(nZ)

    if ops['bidiphase'] and not ops['bidi_corrected']:
        bidiphase = int(ops['bidiphase'])
    else:
        bidiphase = 0

    refAndMasks = compute_reference_masks(refImg, ops)
    
    ### ------------- register frames to reference image ------------ ###

    mean_img = np.zeros((Ly, Lx), 'float32')
    rigid_offsets, nonrigid_offsets, zpos, cmax_all = [], [], [], []

    if ops['frames_include'] != -1:
        n_frames = min(n_frames, ops['frames_include'])

    t0 = time.time()
    
    for k in np.arange(0, n_frames, batch_size):
        frames = f_align_in[k : min(k + batch_size, n_frames)]
        frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, zest = register_frames(refAndMasks, frames, 
                                                                                rmin=rmin, rmax=rmax, 
                                                                                bidiphase=bidiphase, 
                                                                                ops=ops,
                                                                                nZ=nZ)
        rigid_offsets.append([ymax, xmax, cmax])
        if zest is not None:
            zpos.extend(list(zest[0]))
            cmax_all.extend(list(zest[1]))
        if ops['nonrigid']:
            nonrigid_offsets.append([ymax1, xmax1, cmax1])

        mean_img += frames.sum(axis=0) / n_frames

        if f_align_out is None:
            f_align_in[k : min(k + batch_size, n_frames)] = frames
        else:
            f_align_out[k : min(k + batch_size, n_frames)] = frames
            
        if (ops['reg_tif'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_tif_chan2']):
            fname = io.generate_tiff_filename(
                functional_chan=ops['functional_chan'],
                align_by_chan=ops['align_by_chan'],
                save_path=ops['save_path'],
                k=k,
                ichan=True
            )
            io.save_tiff(mov=frames, fname=fname)
        
        print('Registered %d/%d in %0.2fs'%(k+frames.shape[0], n_frames, time.time()-t0))
    rigid_offsets = utils.combine_offsets_across_batches(rigid_offsets, rigid=True)
    if ops['nonrigid']:
        nonrigid_offsets = utils.combine_offsets_across_batches(nonrigid_offsets, rigid=False)
    
    return refImg_orig, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, (zpos, cmax_all)

def shift_frames_and_write(f_alt_in, f_alt_out=None, yoff=None, xoff=None, yoff1=None, xoff1=None, ops=default_ops()):
    """ shift frames for alternate channel in f_alt_in and write to f_alt_out if not None (else write to f_alt_in) """
    n_frames, Ly, Lx = f_alt_in.shape
    if yoff is None or xoff is None:
        raise ValueError('no rigid registration offsets provided')
    elif yoff.shape[0] != n_frames or xoff.shape[0] != n_frames:
        raise ValueError('rigid registration offsets are not the same size as input frames')
    # Overwrite blocks if nonrigid registration is activated   
    blocks = None
    if ops.get('nonrigid'):
        if yoff1 is None or xoff1 is None:
            raise ValueError('nonrigid registration is activated but no nonrigid shifts provided')
        elif yoff1.shape[0] != n_frames or xoff1.shape[0] != n_frames:
            raise ValueError('nonrigid registration offsets are not the same size as input frames')

        blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops['block_size'])

    if ops['frames_include'] != -1:
        n_frames = min(n_frames, ops['frames_include'])

    mean_img = np.zeros((Ly, Lx), 'float32')
    batch_size = ops['batch_size']
    t0 = time.time()
    for k in np.arange(0, n_frames, batch_size):
        frames = f_alt_in[k : min(k + batch_size, n_frames)].astype('float32')
        yoffk = yoff[k : min(k + batch_size, n_frames)].astype(int)
        xoffk = xoff[k : min(k + batch_size, n_frames)].astype(int)
        if ops.get('nonrigid'):
            yoff1k = yoff1[k : min(k + batch_size, n_frames)]
            xoff1k = xoff1[k : min(k + batch_size, n_frames)]
        else:
            yoff1k, xoff1k = None, None

        frames = shift_frames(frames, yoffk, xoffk, yoff1k, xoff1k, blocks, ops)
        mean_img += frames.sum(axis=0) / n_frames

        if f_alt_out is None:
            f_alt_in[k : min(k + batch_size, n_frames)] = frames
        else:
            f_alt_out[k : min(k + batch_size, n_frames)] = frames
                
        if (ops['reg_tif_chan2'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_tif']):
            fname = io.generate_tiff_filename(
                functional_chan=ops['functional_chan'],
                align_by_chan=ops['align_by_chan'],
                save_path=ops['save_path'],
                k=k,
                ichan=False
            )
            io.save_tiff(mov=frames, fname=fname)

        print('Second channel, Registered %d/%d in %0.2fs'%(k+frames.shape[0], n_frames, time.time()-t0))

    return mean_img  


def registration_wrapper(f_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None, refImg=None, align_by_chan2=False, ops=default_ops()):
    """ main registration function

    if f_raw is not None, f_raw is read and registered and saved to f_reg
    if f_raw_chan2 is not None, f_raw_chan2 is read and registered and saved to f_reg_chan2

    the registration shifts are computed on chan2 if ops['functional_chan'] != ops['align_by_chan']


    Parameters
    ----------------

    f_reg : array of registered functional frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx

    f_raw : array of raw functional frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx

    f_reg_chan2 : array of registered anatomical frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx

    f_raw_chan2 : array of raw anatomical frames, np.ndarray or io.BinaryRWFile
        n_frames x Ly x Lx

    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    align_by_chan2: boolean
        whether you'd like to align by non-functional channel

    ops : dictionary or list of dicts
        dictionary containing input arguments for suite2p pipeline

    Returns
    ----------------

    refImg : 2D array, int16
        size [Ly x Lx], initial reference image (if not registered)

    rmin : int
        clip frames at rmin

    rmax : int
        clip frames at rmax

    meanImg : np.ndarray, 
        size [Ly x Lx], Computed Mean Image for functional channel

    rigid_offsets : Tuple of length 3, 
        Rigid shifts computed between each frame and reference image. Shifts for each frame in x,y, and z directions

    nonrigid_offsets : Tuple of length 3
        Non-rigid shifts computed between each frame and reference image.

    zest : Tuple of length 2
        
    meanImg_chan2: np.ndarray, 
        size [Ly x Lx], Computed Mean Image for non-functional channel

    badframes : np.ndarray,   
        size [n_frames, ] Boolean array of frames that have large outlier shifts that may make registration problematic.

    yrange : list of length 2
        Valid ranges for registration along y-axis of frames

    xrange : list of length 2
        Valid ranges for registration along x-axis of frames

    """
    f_alt_in, f_align_out, f_alt_out = None, None, None
    if f_reg_chan2 is None or not align_by_chan2:
        if f_raw is None:
            f_align_in = f_reg
            f_alt_in = f_reg_chan2
        else:
            f_align_in = f_raw
            f_alt_in = f_raw_chan2
            f_align_out = f_reg
            f_alt_out = f_reg_chan2
    else:
        if f_raw is None:
            f_align_in = f_reg_chan2 
            f_alt_in = f_reg
        else:
            f_align_in = f_raw_chan2
            f_alt_in = f_raw
            f_align_out = f_reg_chan2
            f_alt_out = f_reg


    n_frames, Ly, Lx = f_align_in.shape
    if f_alt_in is not None and f_alt_in.shape[0] == f_align_in.shape[0]:
        nchannels = 2
        print('registering two channels')
    else:
        nchannels = 1

    outputs = compute_reference_and_register_frames(f_align_in, f_align_out=f_align_out, refImg=refImg, ops=ops)
    refImg, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, zest = outputs
    yoff, xoff, corrXY = rigid_offsets

            
    if ops['nonrigid']:
            yoff1, xoff1, corrXY1 = nonrigid_offsets
    else:
        yoff1, xoff1, corryXY1 = None, None, None

    if nchannels > 1:
        mean_img_alt = shift_frames_and_write(f_alt_in, f_alt_out, yoff, xoff, yoff1, xoff1, ops)
    else:
        mean_img_alt = None

    if nchannels==1 or not align_by_chan2:
        meanImg = mean_img
        if nchannels==2:
            meanImg_chan2 = mean_img_alt
        else:
            meanImg_chan2 = None
    elif nchannels == 2:
        meanImg_chan2 = mean_img
        meanImg = mean_img_alt
            
        
    # compute valid region
    badframes = np.zeros(n_frames, 'bool')
    if 'data_path' in ops and len(ops['data_path']) > 0:
        badfrfile = path.abspath(path.join(ops['data_path'][0], 'bad_frames.npy'))
        # Check if badframes file exists
        if path.isfile(badfrfile):
            print('bad frames file path: %s'%badfrfile)
            bf_indices = np.load(badfrfile)
            bf_indices = bf_indices.flatten().astype(int)
            # Set indices of badframes to true
            badframes[bf_indices] = True
            print('number of badframes: %d'%badframes.sum())

    # return frames which fall outside range
    badframes, yrange, xrange = compute_crop(
        xoff=xoff,
        yoff=yoff,
        corrXY=corrXY,
        th_badframes=ops['th_badframes'],
        badframes=badframes,
        maxregshift=ops['maxregshift'],
        Ly=Ly,
        Lx=Lx,
    )

    return refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange

def register_binary(ops: Dict[str, Any], refImg=None, raw=True):
    """ main registration function

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
    Ly, Lx = ops['Ly'], ops['Lx']
    n_frames = ops['nframes']
    print('registering %d frames'%ops['nframes'])
    
    # get binary file paths
    raw = raw and ops.get('keep_movie_raw') and 'raw_file' in ops and path.isfile(ops['raw_file'])
    reg_file_align = ops['reg_file'] if (ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan']) else ops['reg_file_chan2']
    if raw:
        raw_file_align = ops.get('raw_file') if (ops['nchannels'] < 2 or ops['functional_chan'] == ops['align_by_chan']) else ops.get('raw_file_chan2')
    else:
        raw_file_align = None
        if ops['do_bidiphase'] and ops['bidiphase'] != 0:
            ops['bidi_corrected'] = True

    if ops['nchannels'] > 1:
        reg_file_alt = ops['reg_file_chan2'] if ops['functional_chan'] == ops['align_by_chan'] else ops['reg_file']
        raw_file_alt = ops.get('raw_file_chan2') if ops['functional_chan'] == ops['align_by_chan'] else ops.get('raw_file')
        raw_file_alt = raw_file_alt if raw else []
    else:
        reg_file_alt = reg_file_align 
        raw_file_alt = reg_file_align

    with io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=raw_file_align if raw else reg_file_align) as f_align_in, \
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file_align) as f_align_out, \
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=raw_file_alt if raw else reg_file_alt) as f_alt_in,\
         io.BinaryRWFile(Ly=Ly, Lx=Lx, filename=reg_file_alt) as f_alt_out:         
        if not raw:
            f_align_out.close()
            f_align_out = None
            f_alt_out.close()
            f_alt_out = None
        if ops['nchannels'] == 1:
            f_alt_in.close() 
            f_alt_in = None
        
        outputs = registration_wrapper(f_align_out, f_align_in, f_alt_out, f_alt_in, refImg, ops=ops)
        
    # refImg, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, zpos, mean_img_alt, badframes, yrange, xrange = outputs

    # # assign reference image and normalizers
    # ops['refImg'] = refImg 
    # ops['rmin'], ops['rmax'] = rmin, rmax
    # # assign rigid offsets to ops
    # ops['yoff'], ops['xoff'], ops['corrXY'] = rigid_offsets
    # # assign nonrigid offsets to ops
    # ops['yoff1'], ops['xoff1'], ops['corrXY1'] = nonrigid_offsets
    # # assign mean images
    # if ops['nchannels'] == 1 or ops['functional_chan'] == ops['align_by_chan']:
    #     ops['meanImg'] = mean_img 
    # elif ops['nchannels'] == 2:
    #     ops['meanImg_chan2'] = mean_img_alt
    # # assign crop computation and badframes
    # ops['badframes'], ops['yrange'], ops['xrange'] = badframes, yrange, xrange
    
    ops = save_registration_outputs_to_ops(outputs, ops)

    # add enhanced mean image
    ops = enhanced_mean_image(ops)

    return ops


def save_registration_outputs_to_ops(registration_outputs, ops):

    refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange = registration_outputs
    # assign reference image and normalizers
    ops['refImg'] = refImg 
    ops['rmin'], ops['rmax'] = rmin, rmax
    # assign rigid offsets to ops
    ops['yoff'], ops['xoff'], ops['corrXY'] = rigid_offsets
    # assign nonrigid offsets to ops
    if ops['nonrigid']:
        ops['yoff1'], ops['xoff1'], ops['corrXY1'] = nonrigid_offsets
    # assign mean images
    ops['meanImg'] = meanImg
    if meanImg_chan2 is not None:
        ops['meanImg_chan2'] = meanImg_chan2
    # assign crop computation and badframes
    ops['badframes'], ops['yrange'], ops['xrange'] = badframes, yrange, xrange
    if len(zest[0]) > 0:
        ops['zpos_registration'] = np.array(zest[0])
        ops['cmax_registration'] = np.array(zest[1])
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
    mimg0 = compute_enhanced_mean_image(I, ops)
    #mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    #mimg[ops['yrange'][0]:ops['yrange'][1],
    #    ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg0
    print('added enhanced mean image')
    return ops

def compute_enhanced_mean_image(I, ops):
    """ computes enhanced mean image

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

    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    return mimg0