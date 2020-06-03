import os
from os import path
import time

import numpy as np
from tqdm import tqdm

from .pc import pclowhigh, pc_register
from .. import io
from . import nonrigid, register, nonrigid, rigid, utils, bidiphase


def register_binary(ops, refImg=None, raw=True):
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
    if (type(ops) is list) or (type(ops) is np.ndarray):
        for op in ops:
            op = register_binary(op)
        return ops

    # make blocks for nonrigid
    if ops['nonrigid']:
        ops['yblock'], ops['xblock'], ops['nblocks'], ops['maxregshiftNR'], ops['block_size'], ops['NRsm'] = nonrigid.make_blocks(
            Ly=ops['Ly'], Lx=ops['Lx'], maxregshiftNR=ops['maxregshiftNR'], block_size=ops['block_size']
        )

    if not ops['frames_include'] == -1:
        ops['nframes'] = min((ops['nframes'], ops['frames_include']))
    else:
        nbytes = path.getsize(ops['raw_file'] if ops.get('keep_movie_raw') and path.exists(ops['raw_file']) else ops['reg_file'])
        ops['nframes'] = int(nbytes / (2 * ops['Ly'] * ops['Lx']))

    print('registering %d frames'%ops['nframes'])
    # check number of frames and print warnings
    if ops['nframes']<50:
        raise Exception('ERROR: the total number of frames should be at least 50 ')
    if ops['nframes']<200:
        print('WARNING: number of frames is below 200, unpredictable behaviors may occur')

    # get binary file paths
    if raw:
        raw = ('keep_movie_raw' in ops and ops['keep_movie_raw'] and 'raw_file' in ops and path.isfile(ops['raw_file']))
        raw_file_align = []
        raw_file_alt = []
        reg_file_align = []
        reg_file_alt = []
        if raw:
            if ops['nchannels'] > 1:
                if ops['functional_chan'] == ops['align_by_chan']:
                    raw_file_align = ops['raw_file']
                    raw_file_alt = ops['raw_file_chan2']
                    reg_file_align = ops['reg_file']
                    reg_file_alt = ops['reg_file_chan2']
                else:
                    raw_file_align = ops['raw_file_chan2']
                    raw_file_alt = ops['raw_file']
                    reg_file_align = ops['reg_file_chan2']
                    reg_file_alt = ops['reg_file']
            else:
                raw_file_align = ops['raw_file']
                reg_file_align = ops['reg_file']
        else:
            if ops['nchannels'] > 1:
                if ops['functional_chan'] == ops['align_by_chan']:
                    reg_file_align = ops['reg_file']
                    reg_file_alt = ops['reg_file_chan2']
                else:
                    reg_file_align = ops['reg_file_chan2']
                    reg_file_alt = ops['reg_file']
            else:
                reg_file_align = ops['reg_file']



    # compute reference image
    if refImg is not None:
        print('NOTE: user reference frame given')
    else:
        t0 = time.time()
        bin_file = raw_file_align if raw else reg_file_align
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
            bidi = bidiphase.compute(frames)
            print('NOTE: estimated bidiphase offset from data: %d pixels' % bidi)
        else:
            bidi = ops['bidiphase']
        if bidi != 0:
            bidiphase.shift(frames, bidi)
        refImg = register.pick_initial_reference(frames)

        ops['bidiphase'] = 0
        niter = 8


        if 'yblock' not in ops:
            ops['yblock'], ops['xblock'], ops['nblocks'], ops['maxregshiftNR'], ops['block_size'], ops[
                'NRsm'] = nonrigid.make_blocks(
                Ly=ops['Ly'], Lx=ops['Lx'], maxregshiftNR=ops['maxregshiftNR'], block_size=ops['block_size']
            )

        for iter in range(0, niter):
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

            if ops['bidiphase'] and not ops['bidi_corrected']:
                bidiphase.shift(frames, ops['bidiphase'])

            freg, ymax, xmax, cmax, yxnr = register.compute_motion_and_shift(
                data=frames,
                refAndMasks=[maskMul, maskOffset, cfRefImg],
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
            ymax = ymax.astype(np.float32)
            xmax = xmax.astype(np.float32)
            isort = np.argsort(-cmax)
            nmax = int(frames.shape[0] * (1. + iter) / (2 * niter))
            refImg = freg[isort[1:nmax], :, :].mean(axis=0).squeeze().astype(np.int16)
            dy, dx = -ymax[isort[1:nmax]].mean(), -xmax[isort[1:nmax]].mean()

            # shift data requires an array of shifts
            dy = np.array([int(np.round(dy))])
            dx = np.array([int(np.round(dx))])
            rigid.shift_data(refImg, dy, dx)
            refImg = refImg.squeeze()

        ops['bidiphase'] = bidi
        print('Reference frame, %0.2f sec.'%(time.time()-t0))
    ops['refImg'] = refImg


    # register binary to reference image
    maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(
        refImg0=refImg,
        spatial_taper=ops['spatial_taper'],
        smooth_sigma=ops['smooth_sigma'],
        pad_fft=ops['pad_fft'],
        reg_1p=ops['1Preg'],
        spatial_hp=ops['spatial_hp'],
        pre_smooth=ops['pre_smooth'],
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
            if ops['pre_smooth'] and ops['pre_smooth'] % 2:
                raise ValueError("if set, pre_smooth must be a positive even integer.")
            if ops['spatial_hp'] % 2:
                raise ValueError("spatial_hp must be a positive even integer.")
            data = data.astype(np.float32)

            if ops['pre_smooth']:
                data = utils.spatial_smooth(data, int(ops['pre_smooth']))
            data = utils.spatial_high_pass(data, int(ops['spatial_hp']))
            refImg = data

        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
            refImg0=refImg,
            maskSlope=maskSlope,
            smooth_sigma=ops['smooth_sigma'],
            yblock=ops['yblock'],
            xblock=ops['xblock'],
            pad_fft=ops['pad_fft'],
        )
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]

    mean_img = np.zeros((ops['Ly'], ops['Lx']))
    rigid_offsets, nonrigid_offsets = [], []
    with io.BinaryFile(nbatch=ops['batch_size'], Ly=ops['Ly'], Lx=ops['Lx'], nframes=ops['nframes'],
                             reg_file=reg_file_align, raw_file=raw_file_align) as f:
        for k, data in tqdm(enumerate(f)):

            if ops['bidiphase'] and not ops['bidi_corrected']:
                bidiphase.shift(data, ops['bidiphase'])

            data, ymax, xmax, cmax, yxnr = register.compute_motion_and_shift(
                data=data,
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

            # output
            rigid_offsets.append([ymax, xmax, cmax])
            nonrigid_offsets.append(list(yxnr))
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
    offsets = rigid_offsets + nonrigid_offsets

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
                ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)

                # apply shifts
                if ops['bidiphase'] != 0 and not ops['bidi_corrected']:
                    bidiphase.shift(data, ops['bidiphase'])

                rigid.shift_data(data, ymax, xmax)

                if ops['nonrigid']:
                    ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]
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

    if 'yoff' not in ops:
        nframes = ops['nframes']
        ops['yoff'] = np.zeros((nframes,), np.float32)
        ops['xoff'] = np.zeros((nframes,), np.float32)
        ops['corrXY'] = np.zeros((nframes,), np.float32)
        if ops['nonrigid']:
            nb = ops['nblocks'][0] * ops['nblocks'][1]
            ops['yoff1'] = np.zeros((nframes, nb), np.float32)
            ops['xoff1'] = np.zeros((nframes, nb), np.float32)
            ops['corrXY1'] = np.zeros((nframes, nb), np.float32)

    ops['yoff'] += offsets[0]
    ops['xoff'] += offsets[1]
    ops['corrXY'] += offsets[2]
    if ops['nonrigid']:
        ops['yoff1'] += offsets[3]
        ops['xoff1'] += offsets[4]
        ops['corrXY1'] += offsets[5]

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
    ops['badframes'], ops['yrange'], ops['xrange'] = register.compute_crop(
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

    if 'ops_path' in ops:
        np.save(ops['ops_path'], ops)
    return ops


def get_pc_metrics(ops, use_red=False):
    """ computes registration metrics using top PCs of registered movie

        movie saved as binary file ops['reg_file']
        metrics saved to ops['regPC'] and ops['X']
        'regDX' is nPC x 3 where X[:,0] is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts
        'regPC' is average of top and bottom frames for each PC
        'tPC' is PC across time frames

        Parameters
        ----------
        ops : dictionary
            'nframes', 'Ly', 'Lx', 'reg_file' (if use_red=True, 'reg_file_chan2')
            (optional, 'refImg', 'block_size', 'maxregshiftNR', 'smooth_sigma', 'maxregshift', '1Preg')
        use_red : :obj:`bool`, optional
            default False, whether to use 'reg_file' or 'reg_file_chan2'

        Returns
        -------
            ops : dictionary
                adds 'regPC' and 'tPC' and 'regDX'

    """
    nsamp = min(5000, ops['nframes'])  # n frames to pick from full movie
    if ops['nframes'] < 5000:
        nsamp = min(2000, ops['nframes'])
    if ops['Ly'] > 700 or ops['Lx'] > 700:
        nsamp = min(2000, nsamp)
    nPC = 30 # n PCs to compute motion for
    nlowhigh = np.minimum(300,int(ops['nframes']/2))  # n frames to average at ends of PC coefficient sortings
    ix = np.linspace(0,ops['nframes']-1,nsamp).astype('int')

    mov = io.get_frames(
        Lx=ops['Lx'],
        Ly=ops['Ly'],
        xrange=ops['xrange'],
        yrange=ops['yrange'],
        ix=ix,
        bin_file=ops['reg_file_chan2'] if use_red and 'reg_file_chan2' in ops else ops['reg_file'],
        crop=True,
    )

    pclow, pchigh, sv, v = pclowhigh(mov, nlowhigh, nPC)
    if 'block_size' not in ops:
        ops['block_size'] = [128, 128]
    if 'maxregshiftNR' not in ops:
        ops['maxregshiftNR'] = 5
    if 'smooth_sigma' not in ops:
        ops['smooth_sigma'] = 1.15
    if 'maxregshift' not in ops:
        ops['maxregshift'] = 0.1
    if '1Preg' not in ops:
        ops['1Preg'] = False

    X = pc_register(pclow, pchigh, spatial_hp=ops['spatial_hp'], pre_smooth=ops['pre_smooth'],
                       bidi_corrected=ops['bidi_corrected'], smooth_sigma=ops['smooth_sigma'], smooth_sigma_time=ops['smooth_sigma_time'],
                       block_size=ops['block_size'], maxregshift=ops['maxregshift'], maxregshiftNR=ops['maxregshiftNR'],
                       reg_1p=ops['1Preg'], snr_thresh=ops['snr_thresh'], is_nonrigid=ops['nonrigid'], pad_fft=ops['pad_fft'],
                       bidiphase=ops['bidiphase'], spatial_taper=ops['spatial_taper'])
    ops['regPC'] = np.concatenate((pclow[np.newaxis, :,:,:], pchigh[np.newaxis, :,:,:]), axis=0)
    ops['regDX'] = X
    ops['tPC'] = v

    return ops


def compute_zpos(Zreg, ops):
    """ compute z position of frames given z-stack Zreg

    Parameters
    ------------

    Zreg : 3D array
        size [nplanes x Ly x Lx], z-stack

    ops : dictionary
        'reg_file' <- binary to register to z-stack, 'smooth_sigma',
        'Ly', 'Lx', 'batch_size'


    """
    if 'reg_file' not in ops:
        print('ERROR: no binary')
        return

    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch

    ops_orig = ops.copy()
    ops['nonrigid'] = False
    nplanes, zLy, zLx = Zreg.shape
    if Zreg.shape[1] != Ly or Zreg.shape[2] != Lx:
        # padding
        if Zreg.shape[1] > Ly:
            Zreg = Zreg[:, ]
        pad = np.zeros((nplanes, int(N/2), zLx))
        dsmooth = np.concatenate((pad, Zreg, pad), axis=1)
        pad = np.zeros((dsmooth.shape[0], dsmooth.shape[1], int(N/2)))
        dsmooth = np.concatenate((pad, dsmooth, pad), axis=2)

    nbytes = os.path.getsize(ops['reg_file'])
    nFrames = int(nbytes/(2 * Ly * Lx))

    reg_file = open(ops['reg_file'], 'rb')
    refAndMasks = []
    for Z in Zreg:
        refAndMasks.append(
            rigid.phasecorr_reference(
                refImg0=Z,
                spatial_taper=ops['spatial_taper'],
                smooth_sigma=ops['smooth_sigma'],
                pad_fft=ops['pad_fft'],
                reg_1p=ops['1Preg'],
                spatial_hp=ops['spatial_hp'],
                pre_smooth=ops['pre_smooth'],
            )
        )

    zcorr = np.zeros((Zreg.shape[0], nFrames), np.float32)
    t0 = time.time()
    k = 0
    nfr = 0
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        buff = []
        if (data.size==0) | (nfr >= ops['nframes']):
            break
        data = np.float32(np.reshape(data, (-1, Ly, Lx)))
        inds = np.arange(nfr, nfr+data.shape[0], 1, int)
        for z,ref in enumerate(refAndMasks):
            _, _, zcorr[z, inds] = rigid.phasecorr(
                data=data,
                refAndMasks=ref,
                maxregshift=ops['maxregshift'],
                reg_1p=ops['1Preg'],
                spatial_hp=ops['spatial_hp'],
                pre_smooth=ops['pre_smooth'],
                smooth_sigma_time=ops['smooth_sigma_time'],
            )
            if z%10 == 1:
                print('%d planes, %d/%d frames, %0.2f sec.'%(z, nfr, ops['nframes'], time.time()-t0))
        print('%d planes, %d/%d frames, %0.2f sec.'%(z, nfr, ops['nframes'], time.time()-t0))
        nfr += data.shape[0]
        k+=1

    reg_file.close()
    ops_orig['zcorr'] = zcorr
    return ops_orig, zcorr