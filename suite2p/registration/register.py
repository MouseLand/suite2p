import time, os
import numpy as np
from scipy.fftpack import next_fast_len
from numpy import fft
from numba import vectorize, complex64, float32, int16
import math
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d
from suite2p.io import tiff
from mkl_fft import fft2, ifft2
from . import reference, bidiphase, nonrigid, utils, rigid

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
    maskMul, maskOffset, cfRefImg = rigid.phasecorr_reference(refImg, ops)
    if 'nonrigid' in ops and ops['nonrigid']:
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]
    return refAndMasks

def compute_motion_and_shift(data, refAndMasks, ops):
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
    data : int16
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

    if ops['bidiphase']!=0 and not ops['bidi_corrected']:
        bidiphase.shift(data, ops['bidiphase'])
    nr=False
    yxnr = []
    if ops['nonrigid'] and len(refAndMasks)>3:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        nr=True

    # rigid registration
    if ops['smooth_sigma_time'] > 0: # temporal smoothing:
        data_smooth = gaussian_filter1d(data.copy(), sigma=ops['smooth_sigma_time'], axis=0)
        ymax, xmax, cmax = rigid.phasecorr(data_smooth, refAndMasks[:3], ops)
    else:
        ymax, xmax, cmax = rigid.phasecorr(data, refAndMasks[:3], ops)
    rigid.shift_data(data, ymax, xmax)

    # non-rigid registration
    if nr:
        if ops['smooth_sigma_time'] > 0: # temporal smoothing:
            data_smooth = gaussian_filter1d(data.copy(), sigma=ops['smooth_sigma_time'], axis=0)
            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(data_smooth, refAndMasks[3:], ops)
        else:
            ymax1, xmax1, cmax1, _ = nonrigid.phasecorr(data, refAndMasks[3:], ops)
        yxnr = [ymax1,xmax1,cmax1]
        data = nonrigid.transform_data(data, ops, ymax1, xmax1)
    return data, ymax, xmax, cmax, yxnr

def compute_crop(ops):
    ''' determines ops['badframes'] (using ops['th_badframes'])
        and excludes these ops['badframes'] when computing valid ranges
        from registration in y and x
    '''
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
    ymin = np.maximum(0, np.ceil(np.amax(ops['yoff'][np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(ops['yoff'])))
    xmin = np.maximum(0, np.ceil(np.amax(ops['xoff'][np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(ops['xoff'])))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    return ops

def register_binary_to_ref(ops, refImg, reg_file_align, raw_file_align):
    """ register binary data to reference image refImg

    Parameters
    ----------
    ops : dictionary

    refImg : int16
        reference image

    reg_file_align : string
        file to (read if raw_file_align empty, and) write registered binary to

    raw_file_align : string
        file to read raw binary from (if not empty)

    Returns
    -------
    ops : dictionary
        sets 'meanImg' or 'meanImg_chan2'
        maskMul, maskOffset, cfRefImg (see register.prepare_masks for details)

    offsets : list
        [ymax, xmax, cmax, yxnr] <- shifts and correlations
    """
    offsets = utils.init_offsets(ops)
    refAndMasks = prepare_refAndMasks(refImg,ops)

    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch
    if len(raw_file_align) > 0:
        raw = True
    else:
        raw = False
        #raw = 'keep_movie_raw' in ops and ops['keep_movie_raw'] and 'raw_file' in ops and os.path.isfile(ops['raw_file'])
    if raw:
        reg_file_align = open(reg_file_align, 'wb')
        raw_file_align = open(raw_file_align, 'rb')
    else:
        reg_file_align = open(reg_file_align, 'r+b')

    meanImg = np.zeros((Ly, Lx))
    k=0
    nfr=0
    t0 = time.time()
    while True:
        if raw:
            buff = raw_file_align.read(nbytesread)
        else:
            buff = reg_file_align.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        buff = []
        if (data.size==0) | (nfr >= ops['nframes']):
            break
        data = np.float32(np.reshape(data, (-1, Ly, Lx)))

        dout = compute_motion_and_shift(data, refAndMasks, ops)
        data = np.minimum(dout[0], 2**15 - 2)
        meanImg += data.sum(axis=0)
        data = data.astype('int16')

        # write to reg_file_align
        if not raw:
            reg_file_align.seek(-2*data.size,1)
        reg_file_align.write(bytearray(data))

        # compile offsets (dout[1:])
        for n in range(len(dout)-1):
            if n < 3:
                offsets[n] = np.hstack((offsets[n], dout[n+1]))
            else:
                # add on nonrigid stats
                for m in range(len(dout[-1])):
                    offsets[n+m] = np.vstack((offsets[n+m], dout[-1][m]))

        # write registered tiffs
        if ops['reg_tif']:
            tiff.write(data, ops, k, True)

        nfr += data.shape[0]
        k += 1
        if k%5==0:
            print('%d/%d frames, %0.2f sec.'%(nfr, ops['nframes'], time.time()-t0))

    print('%d/%d frames, %0.2f sec.'%(nfr, ops['nframes'], time.time()-t0))

    # mean image across all frames
    if ops['nchannels']==1 or ops['functional_chan']==ops['align_by_chan']:
        ops['meanImg'] = meanImg/ops['nframes']
    else:
        ops['meanImg_chan2'] = meanImg/ops['nframes']

    reg_file_align.close()
    if raw:
        raw_file_align.close()
    return ops, offsets

def apply_shifts(data, ops, ymax, xmax, ymax1, xmax1):
    """ apply rigid and nonrigid shifts to data (for chan that's not 'align_by_chan')

    Parameters
    ----------
    data : int16


    ops : dictionary

    refImg : int16
        reference image

    reg_file_align : string
        file to (read if raw_file_align empty, and) write registered binary to

    raw_file_align : string
        file to read raw binary from (if not empty)

    Returns
    -------
    ops : dictionary
        sets 'meanImg' or 'meanImg_chan2'
        maskMul, maskOffset, cfRefImg (see register.prepare_masks for details)

    offsets : list
        [ymax, xmax, cmax, yxnr] <- shifts and correlations


    """
    if ops['bidiphase']!=0  and not ops['bidi_corrected']:
        bidiphase.shift(data, ops['bidiphase'])
    rigid.shift_data(data, ymax, xmax)
    if ops['nonrigid']==True:
        data = nonrigid.transform_data(data, ops, ymax1, xmax1)
    return data

def apply_shifts_to_binary(ops, offsets, reg_file_alt, raw_file_alt):
    ''' apply registration shifts to binary data'''
    nbatch = ops['batch_size']
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread = 2 * Ly * Lx * nbatch
    ix = 0
    meanImg = np.zeros((Ly, Lx))
    k=0
    t0 = time.time()
    if len(raw_file_alt) > 0:
        reg_file_alt = open(reg_file_alt, 'wb')
        raw_file_alt = open(raw_file_alt, 'rb')
        raw = True
    else:
        reg_file_alt = open(reg_file_alt, 'r+b')
        raw = False
    while True:
        if raw:
            buff = raw_file_alt.read(nbytesread)
        else:
            buff = reg_file_alt.read(nbytesread)

        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        buff = []
        if (data.size==0) | (ix >= ops['nframes']):
            break
        data = np.reshape(data[:int(np.floor(data.shape[0]/Ly/Lx)*Ly*Lx)], (-1, Ly, Lx))
        nframes = data.shape[0]
        iframes = ix + np.arange(0,nframes,1,int)

        # get shifts
        ymax, xmax = offsets[0][iframes].astype(np.int32), offsets[1][iframes].astype(np.int32)
        ymax1,xmax1 = [],[]
        if ops['nonrigid']:
            ymax1, xmax1 = offsets[3][iframes], offsets[4][iframes]

        # apply shifts
        data = apply_shifts(data, ops, ymax, xmax, ymax1, xmax1)
        data = np.minimum(data, 2**15 - 2)
        meanImg += data.mean(axis=0)
        data = data.astype('int16')
        # write to binary
        if not raw:
            reg_file_alt.seek(-2*data.size,1)
        reg_file_alt.write(bytearray(data))

        # write registered tiffs
        if ops['reg_tif_chan2']:
            tiff.write(data, ops, k, False)
        ix += nframes
        k+=1
    if ops['functional_chan']!=ops['align_by_chan']:
        ops['meanImg'] = meanImg/k
    else:
        ops['meanImg_chan2'] = meanImg/k
    print('Registered second channel in %0.2f sec.'%(time.time()-t0))

    reg_file_alt.close()
    if raw:
        raw_file_alt.close()
    return ops

def register_binary(ops, refImg=None, raw=True):
    ''' registration of binary files '''
    # if ops is a list of dictionaries, each will be registered separately
    if (type(ops) is list) or (type(ops) is np.ndarray):
        for op in ops:
            op = register_binary(op)
        return ops

    # make blocks for nonrigid
    if ops['nonrigid']:
        ops = nonrigid.make_blocks(ops)

    ops['nframes'] = utils.get_nFrames(ops)
    if not ops['frames_include'] == -1:
        ops['nframes'] = min((ops['nframes'], ops['frames_include']))

    print('registering %d frames'%ops['nframes'])
    # check number of frames and print warnings
    if ops['nframes']<50:
        raise Exception('ERROR: the total number of frames should be at least 50 ')
    if ops['nframes']<200:
        print('WARNING: number of frames is below 200, unpredictable behaviors may occur')

    # get binary file paths
    if raw:
        raw = ('keep_movie_raw' in ops and ops['keep_movie_raw'] and
                'raw_file' in ops and os.path.isfile(ops['raw_file']))
    reg_file_align, reg_file_alt, raw_file_align, raw_file_alt = utils.bin_paths(ops, raw)

    # compute reference image
    if refImg is not None:
        print('NOTE: user reference frame given')
    else:
        t0 = time.time()
        if raw:
            refImg, bidi = reference.compute_reference_image(ops, raw_file_align)
        else:
            refImg, bidi = reference.compute_reference_image(ops, reg_file_align)
        ops['bidiphase'] = bidi
        print('Reference frame, %0.2f sec.'%(time.time()-t0))
    ops['refImg'] = refImg

    k = 0
    nfr = 0

    # register binary to reference image
    ops, offsets = register_binary_to_ref(ops, refImg, reg_file_align, raw_file_align)

    if ops['nchannels']>1:
        ops = apply_shifts_to_binary(ops, offsets, reg_file_alt, raw_file_alt)

    if 'yoff' not in ops:
        nframes = ops['nframes']
        ops['yoff'] = np.zeros((nframes,),np.float32)
        ops['xoff'] = np.zeros((nframes,),np.float32)
        ops['corrXY'] = np.zeros((nframes,),np.float32)
        if ops['nonrigid']:
            nb = ops['nblocks'][0] * ops['nblocks'][1]
            ops['yoff1'] = np.zeros((nframes,nb),np.float32)
            ops['xoff1'] = np.zeros((nframes,nb),np.float32)
            ops['corrXY1'] = np.zeros((nframes,nb),np.float32)

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
        badfrfile = os.path.abspath(os.path.join(ops['data_path'][0], 'bad_frames.npy'))
        print('bad frames file path: %s'%badfrfile)
        if os.path.isfile(badfrfile):
            badframes = np.load(badfrfile)
            badframes = badframes.flatten().astype(int)
            #print('badframes[0]=%d, badframes[-1]=%d'%(badframes[0],badframes[-1]))
            ops['badframes'][badframes] = True
            print('number of badframes: %d'%ops['badframes'].sum())

    # return frames which fall outside range
    ops = compute_crop(ops)

    if not raw:
        ops['bidi_corrected'] = True

    if 'ops_path' in ops:
        np.save(ops['ops_path'], ops)
    return ops

def register_npy(Z, ops):
    # if ops does not have refImg, get a new refImg
    if 'refImg' not in ops:
        ops['refImg'] = Z.mean(axis=0)
    ops['nframes'], ops['Ly'], ops['Lx'] = Z.shape

    if ops['nonrigid']:
        ops = nonrigid.make_blocks(ops)

    Ly = ops['Ly']
    Lx = ops['Lx']

    nbatch = ops['batch_size']
    meanImg = np.zeros((Ly, Lx)) # mean of this stack

    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)

    maskMul, maskOffset, cfRefImg = prepare_masks(refImg, ops) # prepare masks for rigid registration
    if ops['nonrigid']:
        # prepare masks for non- rigid registration
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, ops)
        refAndMasks = [maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR]
        nb = ops['nblocks'][0] * ops['nblocks'][1]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]

    k = 0
    nfr = 0
    Zreg = np.zeros((nframes, Ly, Lx,), 'int16')
    while True:
        irange = np.arange(nfr, nfr+nbatch)
        data = Z[irange, :,:]
        if data.size==0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax, yxnr = phasecorr(data, refAndMasks, ops)
        dwrite = dwrite.astype('int16') # need to hold on to this
        meanImg += dwrite.sum(axis=0)
        yoff = np.hstack((yoff, ymax))
        xoff = np.hstack((xoff, xmax))
        corrXY = np.hstack((corrXY, cmax))
        if ops['nonrigid']:
            yoff1 = np.vstack((yoff1, yxnr[0]))
            xoff1 = np.vstack((xoff1, yxnr[1]))
            corrXY1 = np.vstack((corrXY1, yxnr[2]))
        nfr += dwrite.shape[0]
        Zreg[irange] = dwrite

        k += 1
        if k%5==0:
            print('%d/%d frames %4.2f sec'%(nfr, ops['nframes'], time.time()-k0))

    # compute some potentially useful info
    ops['th_badframes'] = 100
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    dxy = (dx**2 + dy**2)**.5
    cXY = corrXY / medfilt(corrXY, 101)
    px = dxy/np.mean(dxy) / np.maximum(0, cXY)
    ops['badframes'] = px > ops['th_badframes']
    ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(ops['badframes'])])))
    ymax = ops['Ly'] + np.minimum(0, np.floor(np.amin(yoff)))
    xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(ops['badframes'])])))
    xmax = ops['Lx'] + np.minimum(0, np.floor(np.amin(xoff)))
    ops['yrange'] = [int(ymin), int(ymax)]
    ops['xrange'] = [int(xmin), int(xmax)]
    ops['corrXY'] = corrXY

    ops['yoff'] = yoff
    ops['xoff'] = xoff

    if ops['nonrigid']:
        ops['yoff1'] = yoff1
        ops['xoff1'] = xoff1
        ops['corrXY1'] = corrXY1

    ops['meanImg'] = meanImg/ops['nframes']

    return Zreg, ops
