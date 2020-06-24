import os
import time

import numpy as np
from scipy.signal import medfilt

from . import nonrigid, rigid, utils

# This function doesn't work. Has a bunch of name errors. 
def register_stack(Z, ops):
    if 'refImg' not in ops:
        ops['refImg'] = Z.mean(axis=0)
    ops['nframes'], ops['Ly'], ops['Lx'] = Z.shape

    if ops['nonrigid']:
        ops['yblock'], ops['xblock'], ops['nblocks'], ops['block_size'], ops['NRsm'] = nonrigid.make_blocks(
            Ly=ops['Ly'], Lx=ops['Lx'], block_size=ops['block_size']
        )

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

    maskMul, maskOffset, cfRefImg = rigid.prepare_masks(refImg, ops) # prepare masks for rigid registration
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
        dwrite, ymax, xmax, cmax, yxnr = phasecorr(data, refAndMasks, ops) # not here
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
            print('%d/%d frames %4.2f sec'%(nfr, ops['nframes'], time.time()-k0)) # where is this timer set?

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
        raise IOError('no binary specified')

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

        half_pad = N // 2
        dsmooth = np.pad(Zreg, ((0, 0), (half_pad, half_pad), (half_pad, half_pad)), mode='constant', constant_values=0)

    nbytes = os.path.getsize(ops['reg_file'])
    nFrames = int(nbytes/(2 * Ly * Lx))

    reg_file = open(ops['reg_file'], 'rb')
    refAndMasks = []
    for Z in Zreg:
        if ops['1Preg']:
            Z = Z.astype(np.float32)
            Z = Z[np.newaxis, :, :]
            if ops['pre_smooth']:
                Z = utils.spatial_smooth(Z, int(ops['pre_smooth']))
            Z = utils.spatial_high_pass(Z, int(ops['spatial_hp_reg']))
            Z = Z.squeeze()

        maskMul, maskOffset = rigid.compute_masks(
            refImg=Z,
            maskSlope=ops['spatial_taper'] if ops['1Preg'] else 3 * ops['smooth_sigma'],
        )
        cfRefImag = rigid.phasecorr_reference(
            refImg=Z,
            smooth_sigma=ops['smooth_sigma'],
            pad_fft=ops['pad_fft'],
        )
        cfRefImag = cfRefImag[np.newaxis, :, :]
        refAndMasks.append((maskMul, maskOffset, cfRefImag))

    zcorr = np.zeros((Zreg.shape[0], nFrames), np.float32)
    t0 = time.time()
    k = 0
    nfr = 0
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        if (data.size==0) | (nfr >= ops['nframes']):
            break
        data = np.float32(np.reshape(data, (-1, Ly, Lx)))
        inds = np.arange(nfr, nfr+data.shape[0], 1, int)
        for z,ref in enumerate(refAndMasks):

            # preprocessing for 1P recordings
            if ops['1Preg']:
                data = data.astype(np.float32)

                if ops['pre_smooth']:
                    data = utils.spatial_smooth(data, int(ops['pre_smooth']))
                data = utils.spatial_high_pass(data, int(ops['spatial_hp_reg']))

            maskMul, maskOffset, cfRefImg = ref
            cfRefImg = cfRefImg.squeeze()

            _, _, zcorr[z, inds] = rigid.phasecorr(
                data=rigid.apply_masks(data=data, maskMul=maskMul, maskOffset=maskOffset),
                cfRefImg=cfRefImg,
                maxregshift=ops['maxregshift'],
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
