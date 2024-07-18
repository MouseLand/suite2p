"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time

import numpy as np
from scipy.signal import medfilt
import torch

import logging 
logger = logging.getLogger(__name__)

from . import nonrigid, rigid, utils
from .. import default_settings


def register_frames_zstack(refAndMasks, frames, rmin=-np.inf, rmax=np.inf, bidiphase=0,
                    settings=default_settings(), nZ=1):
    """ register frames to zstack of reference images """
    cmax_best = -np.inf * np.ones(len(frames), "float32")
    cmax_all = -np.inf * np.ones((len(frames), nZ), "float32")
    zpos_best = np.zeros(len(frames), "int")
    run_nonrigid = settings["nonrigid"]
    for z in range(nZ):
        settings["nonrigid"] = False
        outputs = register_frames(refAndMasks[z], frames.copy(), rmin=rmin[z],
                                    rmax=rmax[z], bidiphase=bidiphase, settings=settings, nZ=1)
        cmax_all[:, z] = outputs[3]
        if z == 0:
            outputs_best = list(outputs[:-4]).copy()
        ibest = cmax_best < cmax_all[:, z]
        zpos_best[ibest] = z
        cmax_best[ibest] = cmax_all[ibest, z]
        for i, (output_best, output) in enumerate(zip(outputs_best, outputs[:-4])):
            output_best[ibest] = output[ibest]
    if run_nonrigid:
        settings["nonrigid"] = True
        nfr = frames.shape[0]
        for i, z in enumerate(zpos_best):
            outputs = register_frames(refAndMasks[z], frames[[i]], rmin=rmin[z],
                                        rmax=rmax[z], bidiphase=bidiphase, settings=settings,
                                        nZ=1)
            if i == 0:
                outputs_best = []
                for output in outputs[:-1]:
                    outputs_best.append(
                        np.zeros((nfr, *output.shape[1:]), dtype=output.dtype))
                    outputs_best[-1][0] = output[0]
            else:
                for output, output_best in zip(outputs[:-1], outputs_best):
                    output_best[i] = output[0]
    if len(outputs_best)==7:
        frames, ymax, xmax, cmax, ymax1, xmax1, cmax1 = outputs_best
    else:
        frames, ymax, xmax, cmax = outputs_best
        ymax1, xmax1, cmax1 = None, None, None
    return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, zpos_best, cmax_all

    
def register_frames(refAndMasks, frames, rmin=-np.inf, rmax=np.inf, bidiphase=0,
                    settings=default_settings(), nZ=1):
    """ register frames to reference image 
    
    Parameters
    ----------

    refAndMasks : list of processed reference images and masks

    frames : np.ndarray, np.int16 or np.float32
        time x Ly x Lx

    rmin : clip frames at rmin

    rmax : clip frames at rmax


    Returns
    --------

    settings : dictionary
        "nframes", "yoff", "xoff", "corrXY", "yoff1", "xoff1", "corrXY1", "badframes"


    """
    maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = refAndMasks
    device = maskMul.device
    
    if bidiphase != 0:
        bidi.shift(frames, bidiphase)

    if device.type == "cuda":
        fr_torch = torch.from_numpy(frames).pin_memory().to(device)
    else:
        fr_torch = torch.from_numpy(frames)
    
    fr_reg = fr_torch.clone()
    

    ymax, xmax, cmax, ymax1, xmax1, cmax1 = compute_shifts(refAndMasks, fr_reg, rmin, rmax, bidiphase, settings)

    frames_out = shift_frames2(fr_torch, ymax, xmax, ymax1, xmax1, blocks, settings)
    
    ymax = ymax.cpu().numpy()
    xmax = xmax.cpu().numpy()
    cmax = cmax.cpu().numpy()
    if ymax1 is not None:
        ymax1 = ymax1.cpu().numpy()
        xmax1 = xmax1.cpu().numpy()
        cmax1 = cmax1.cpu().numpy()
        

    if device.type == "cuda":
        torch.cuda.synchronize()
        
    return frames_out, ymax, xmax, cmax, ymax1, xmax1, cmax1, None, None


# This function doesn"t work. Has a bunch of name errors.
def register_stack(Z, settings):
    """

    Parameters
    ----------
    Z
    settings: dict

    Returns
    -------
    Zreg: nplanes x Ly x Lx
        Z-stack
    settings: dict
    """

    if "refImg" not in settings:
        settings["refImg"] = Z.mean(axis=0)
    settings["nframes"], settings["Ly"], settings["Lx"] = Z.shape

    if settings["nonrigid"]:
        settings["yblock"], settings["xblock"], settings["nblocks"], settings["block_size"], settings[
            "NRsm"] = nonrigid.make_blocks(Ly=settings["Ly"], Lx=settings["Lx"],
                                           block_size=settings["block_size"])

    Ly = settings["Ly"]
    Lx = settings["Lx"]

    nbatch = settings["batch_size"]
    meanImg = np.zeros((Ly, Lx))  # mean of this stack

    yoff = np.zeros((0,), np.float32)
    xoff = np.zeros((0,), np.float32)
    corrXY = np.zeros((0,), np.float32)
    if settings["nonrigid"]:
        yoff1 = np.zeros((0, nb), np.float32)
        xoff1 = np.zeros((0, nb), np.float32)
        corrXY1 = np.zeros((0, nb), np.float32)

    maskMul, maskOffset, cfRefImg = rigid.prepare_masks(
        refImg, settings)  # prepare masks for rigid registration
    if settings["nonrigid"]:
        # prepare masks for non- rigid registration
        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.prepare_masks(refImg, settings)
        refAndMasks = [
            maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR
        ]
        nb = settings["nblocks"][0] * settings["nblocks"][1]
    else:
        refAndMasks = [maskMul, maskOffset, cfRefImg]

    k = 0
    nfr = 0
    Zreg = np.zeros((
        nframes,
        Ly,
        Lx,
    ), "int16")
    while True:
        irange = np.arange(nfr, nfr + nbatch)
        data = Z[irange, :, :]
        if data.size == 0:
            break
        data = np.reshape(data, (-1, Ly, Lx))
        dwrite, ymax, xmax, cmax, yxnr = rigid.phasecorr(data, refAndMasks,
                                                         settings)  # not here
        dwrite = dwrite.astype("int16")  # need to hold on to this
        meanImg += dwrite.sum(axis=0)
        yoff = np.hstack((yoff, ymax))
        xoff = np.hstack((xoff, xmax))
        corrXY = np.hstack((corrXY, cmax))
        if settings["nonrigid"]:
            yoff1 = np.vstack((yoff1, yxnr[0]))
            xoff1 = np.vstack((xoff1, yxnr[1]))
            corrXY1 = np.vstack((corrXY1, yxnr[2]))
        nfr += dwrite.shape[0]
        Zreg[irange] = dwrite

        k += 1
        if k % 5 == 0:
            logger.info("%d/%d frames %4.2f sec" %
                  (nfr, settings["nframes"], time.time() - k0))  # where is this timer set?

    # compute some potentially useful info
    settings["th_badframes"] = 100
    dx = xoff - medfilt(xoff, 101)
    dy = yoff - medfilt(yoff, 101)
    dxy = (dx**2 + dy**2)**.5
    cXY = corrXY / medfilt(corrXY, 101)
    px = dxy / np.mean(dxy) / np.maximum(0, cXY)
    settings["badframes"] = px > settings["th_badframes"]
    ymin = np.maximum(0, np.ceil(np.amax(yoff[np.logical_not(settings["badframes"])])))
    ymax = settings["Ly"] + np.minimum(0, np.floor(np.amin(yoff)))
    xmin = np.maximum(0, np.ceil(np.amax(xoff[np.logical_not(settings["badframes"])])))
    xmax = settings["Lx"] + np.minimum(0, np.floor(np.amin(xoff)))
    settings["yrange"] = [int(ymin), int(ymax)]
    settings["xrange"] = [int(xmin), int(xmax)]
    settings["corrXY"] = corrXY

    settings["yoff"] = yoff
    settings["xoff"] = xoff

    if settings["nonrigid"]:
        settings["yoff1"] = yoff1
        settings["xoff1"] = xoff1
        settings["corrXY1"] = corrXY1

    settings["meanImg"] = meanImg / settings["nframes"]

    return Zreg, settings


def compute_zpos(Zreg, settings, reg_file=None):
    """ compute z position of frames given z-stack Zreg

    Parameters
    ----------

    Zreg : 3D array
        size [nplanes x Ly x Lx], z-stack

    settings : dictionary
        "reg_file" <- binary to register to z-stack, "smooth_sigma",
        "Ly", "Lx", "batch_size"

    Returns
    -------
    settings_orig
    zcorr
    """
    if "reg_file" not in settings:
        raise IOError("no binary specified")

    nbatch = settings["batch_size"]
    Ly = settings["Ly"]
    Lx = settings["Lx"]
    nbytesread = 2 * Ly * Lx * nbatch

    settings_orig = settings.copy()
    settings["nonrigid"] = False
    nplanes, zLy, zLx = Zreg.shape
    if Zreg.shape[1] > Ly or Zreg.shape[2] != Lx:
        Zreg = Zreg[
            :,
        ]

    reg_file = settings["reg_file"] if reg_file is None else reg_file
    nbytes = os.path.getsize(reg_file)
    nFrames = int(nbytes / (2 * Ly * Lx))

    reg_file = open(reg_file, "rb")
    refAndMasks = []
    for Z in Zreg:
        if settings["1Preg"]:
            Z = Z.astype(np.float32)
            Z = Z[np.newaxis, :, :]
            if settings["pre_smooth"]:
                Z = utils.spatial_smooth(Z, int(settings["pre_smooth"]))
            Z = utils.spatial_high_pass(Z, int(settings["spatial_hp_reg"]))
            Z = Z.squeeze()

        maskMul, maskOffset = rigid.compute_masks(
            refImg=Z,
            maskSlope=settings["spatial_taper"] if settings["1Preg"] else 3 * settings["smooth_sigma"],
        )
        cfRefImag = rigid.phasecorr_reference(refImg=Z,
                                              smooth_sigma=settings["smooth_sigma"])
        cfRefImag = cfRefImag[np.newaxis, :, :]
        refAndMasks.append((maskMul, maskOffset, cfRefImag))

    zcorr = np.zeros((Zreg.shape[0], nFrames), np.float32)
    t0 = time.time()
    k = 0
    nfr = 0
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        if (data.size == 0) | (nfr >= settings["nframes"]):
            break
        data = np.float32(np.reshape(data, (-1, Ly, Lx)))
        inds = np.arange(nfr, nfr + data.shape[0], 1, int)
        for z, ref in enumerate(refAndMasks):

            # preprocessing for 1P recordings
            if settings["1Preg"]:
                data = data.astype(np.float32)

                if settings["pre_smooth"]:
                    data = utils.spatial_smooth(data, int(settings["pre_smooth"]))
                data = utils.spatial_high_pass(data, int(settings["spatial_hp_reg"]))

            maskMul, maskOffset, cfRefImg = ref
            cfRefImg = cfRefImg.squeeze()

            _, _, zcorr[z, inds] = rigid.phasecorr(
                data=rigid.apply_masks(data=data, maskMul=maskMul,
                                       maskOffset=maskOffset),
                cfRefImg=cfRefImg,
                maxregshift=settings["maxregshift"],
                smooth_sigma_time=settings["smooth_sigma_time"],
            )
            if z % 10 == 1:
                logger.info("%d planes, %d/%d frames, %0.2f sec." %
                      (z, nfr, settings["nframes"], time.time() - t0))
        logger.info("%d planes, %d/%d frames, %0.2f sec." %
              (z, nfr, settings["nframes"], time.time() - t0))
        nfr += data.shape[0]
        k += 1

    reg_file.close()
    settings_orig["zcorr"] = zcorr
    return settings_orig, zcorr
