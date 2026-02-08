"""
Copyright © 2024 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import shutil
import time
from natsort import natsorted
from datetime import datetime
from getpass import getpass
import pathlib
import contextlib
import numpy as np
from scipy.stats import skew
import torch
import logging 
logger = logging.getLogger(__name__)


from . import extraction, registration, detection, classification, default_settings, default_db
from .registration import zalign

def pipeline(save_path, f_reg, f_raw=None, f_reg_chan2=None, f_raw_chan2=None,
             run_registration=True, settings=default_settings(), badframes=None, stat=None,
             device=torch.device("cuda"), Zstack=None):
    """
    Run suite2p processing pipeline on an array or BinaryFile.

    Runs registration, ROI detection, signal extraction, spike deconvolution,
    and classification sequentially on a single plane.

    Parameters
    ----------
    save_path : str
        Path to save results.
    f_reg : numpy.ndarray or BinaryFile
        Registered or unregistered frames, shape (n_frames, Ly, Lx).
    f_raw : numpy.ndarray or BinaryFile, optional (default None)
        Unregistered frames that will not be overwritten during registration,
        shape (n_frames, Ly, Lx).
    f_reg_chan2 : numpy.ndarray or BinaryFile, optional (default None)
        Non-functional registered or unregistered frames,
        shape (n_frames, Ly, Lx).
    f_raw_chan2 : numpy.ndarray or BinaryFile, optional (default None)
        Non-functional unregistered frames that will not be overwritten
        during registration, shape (n_frames, Ly, Lx).
    run_registration : bool, optional (default True)
        Whether to run registration.
    settings : dict, optional
        Dictionary of pipeline settings from default_settings().
    badframes : numpy.ndarray, optional (default None)
        Boolean array of bad frames (e.g. photostim times),
        shape (n_frames,).
    stat : numpy.ndarray, optional (default None)
        Pre-defined ROI masks. If provided, detection is skipped.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for performing operations.
    Zstack : list, optional (default None)
        Dense Z-stack used for Z-position estimation, via 
        correlation with f_reg per frame.

    Returns
    -------
    reg_outputs : dict
        Registration outputs including shifts and reference images.
    detect_outputs : numpy.ndarray or None
        Detection outputs including correlation and projection maps.
    stat : numpy.ndarray or None
        Array of ROI statistics dictionaries.
    F : numpy.ndarray or None
        ROI fluorescence traces, shape (n_rois, n_frames).
    Fneu : numpy.ndarray or None
        Neuropil fluorescence traces, shape (n_rois, n_frames).
    F_chan2 : numpy.ndarray or None
        ROI fluorescence traces for anatomical channel.
    Fneu_chan2 : numpy.ndarray or None
        Neuropil fluorescence traces for anatomical channel.
    spks : numpy.ndarray or None
        Deconvolved spike traces, shape (n_rois, n_frames).
    iscell : numpy.ndarray or None
        Classification results, shape (n_rois, 2).
    redcell : numpy.ndarray or None
        Red channel cell overlap scores.
    zcorr : numpy.ndarray
        Correlations of f_reg with Z-stack (n_frames, nZ).
    plane_times : dict
        Timing information for each step of the pipeline.
    """

    plane_times = {}
    t1 = time.time()

    # Select file for classification
    settings_classfile = settings["classification"].get("classifier_path", None)
    if settings_classfile is not None and os.path.exists(settings_classfile):
        classfile, ctype = settings_classfile, "settings"
    elif settings["classification"]["use_builtin_classifier"] or not classification.user_classfile.is_file():
        classfile, ctype = classification.builtin_classfile, "builtin"
    else:
        classfile, ctype = classification.user_classfile, "default"
    logger.info(f"NOTE: applying {ctype} classifier: {classfile}")
    
    if run_registration:
        t11 = time.time()
        logger.info("----------- REGISTRATION")
        align_by_chan2 = settings["registration"]["align_by_chan2"]
        reg_outputs = registration.registration_wrapper(
            f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2, f_raw_chan2=f_raw_chan2,
            align_by_chan2=align_by_chan2, save_path=save_path,
            badframes=badframes, settings=settings["registration"], device=device)
        np.save(os.path.join(save_path, "reg_outputs.npy"), reg_outputs)
        plane_times["registration"] = time.time() - t11
        logger.info("----------- Total %0.2f sec" % plane_times["registration"])
        
        n_frames, Ly, Lx = f_reg.shape
        if settings["run"]["do_regmetrics"] and n_frames >= 1500:
            yrange, xrange = reg_outputs["yrange"], reg_outputs["xrange"]
            t0 = time.time()
            out = registration.get_pc_metrics(f_reg, yrange=yrange, xrange=xrange,
                                                settings=settings["registration"])
            reg_outputs["tPC"], reg_outputs["regPC"], reg_outputs["regDX"] = out
            plane_times["registration_metrics"] = time.time() - t0
            logger.info("Registration metrics, %0.2f sec." %
                  plane_times["registration_metrics"])
            np.save(os.path.join(save_path, "reg_outputs.npy"), reg_outputs)
    else:
        try:
            reg_outputs = np.load(os.path.join(save_path, "reg_outputs.npy"), allow_pickle=True).item()
        except:
            reg_outputs = {}
            n_frames, Ly, Lx = f_reg.shape
            reg_outputs["yrange"], reg_outputs["xrange"] = [0, Ly], [0, Lx]
            reg_outputs["badframes"] = np.zeros(n_frames, dtype="bool")
            np.save(os.path.join(save_path, "reg_outputs.npy"), reg_outputs)

    if not settings["run"]["do_detection"]:
        logger.warn("WARNING: skipping cell detection (settings['run']['do_detection']=False)")
        plane_times["total_plane_runtime"] = time.time() - t1
        return reg_outputs, None, None, None, None, None, None, None, None, None, None, plane_times
    
    yrange, xrange = reg_outputs["yrange"], reg_outputs["xrange"]
    meanImg_chan2 = reg_outputs.get("meanImg_chan2", None)
    
    logger.info("----------- ROI DETECTION")
    t11 = time.time()
    if stat is None:
        bad_frames = reg_outputs["badframes"]
        bad_frames[badframes] = True
        logger.info(f"Excluding {bad_frames.sum()} bad frames from detection")
        if not isinstance(settings["diameter"], (list, tuple, np.ndarray)):
            settings["diameter"] = np.array([settings["diameter"], settings["diameter"]])
        elif isinstance(settings["diameter"], (list, tuple)):
            settings["diameter"] = np.array(settings["diameter"])
        if settings["diameter"].size == 1:
            settings["diameter"] = np.array([settings["diameter"], settings["diameter"]])
        detect_outputs, stat, redcell = detection.detection_wrapper(f_reg, 
                                                                    meanImg_chan2=meanImg_chan2,
                                                    yrange=yrange, xrange=xrange,
                                                    tau=settings["tau"], fs=settings["fs"],
                                                    diameter=settings["diameter"],
                                                settings=settings["detection"], 
                                                classifier_path=classfile,
                                                badframes=bad_frames,
                                                preclassify=settings["classification"]["preclassify"],
                                                device=device)
        np.save(os.path.join(save_path, "stat.npy"), stat)
        np.save(os.path.join(save_path, "detect_outputs.npy"), detect_outputs)
        if redcell is not None:
            np.save(os.path.join(save_path, "redcell.npy"), redcell)
        
    plane_times["detection"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["detection"])

    if len(stat) == 0:
        logger.info("no ROIs found")
        plane_times["total_plane_runtime"] = time.time() - t1
        return reg_outputs, detect_outputs, stat, None, None, None, None, None, None, None, None, plane_times

    logger.info("----------- EXTRACTION")
    t11 = time.time()
    snr_threshold = settings["extraction"]["snr_threshold"]
    for step in range(1 + (snr_threshold > 0)):
        F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(
            stat, f_reg, f_reg_chan2=f_reg_chan2, settings=settings["extraction"],
            device=device)
        # subtract neuropil
        dF = F.copy() - settings["extraction"]["neuropil_coefficient"] * Fneu
        # remove ROIs with low SNR and recompute overlapping pixels
        snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)
        keep_rois = snr > snr_threshold
        nremove = (~keep_rois).sum()
        if step==0 and snr_threshold > 0 and nremove > 0:
            logger.info(f"Removing {nremove} ROIs with snr < {snr_threshold}")
            stat = stat[keep_rois]
            redcell = redcell[keep_rois] if redcell is not None else None
            if redcell is not None:
                np.save(os.path.join(save_path, "redcell.npy"), redcell)
            Ly, Lx = f_reg.shape[-2:]
            stat = detection.assign_overlaps(stat, Ly, Lx)
            logger.info("Running second extraction with updated overlap pixels")
        else:
            # do not run second extraction step
            break
    # compute activity statistics for classifier
    sk, sd = skew(dF, axis=1), np.std(dF, axis=1)
    for k, s in enumerate(stat):
        s["snr"], s["skew"], s["std"] = snr[k], sk[k], sd[k]
    np.save(os.path.join(save_path, "stat.npy"), stat)
    plane_times["extraction"] = time.time() - t11
    logger.info("----------- Total %0.2f sec." % plane_times["extraction"])

    if settings["run"]["do_deconvolution"]:
        logger.info("----------- SPIKE DECONVOLUTION")
        t11 = time.time()
        dF = F.copy() - settings["extraction"]["neuropil_coefficient"] * Fneu
        dF = extraction.preprocess(F=dF, fs=settings["fs"], batch_size=settings["extraction"]["batch_size"],
                                   device=device, **settings["dcnv_preprocess"])
        spks = extraction.oasis(F=dF, batch_size=settings["extraction"]["batch_size"],
                                tau=settings["tau"], fs=settings["fs"])
        plane_times["deconvolution"] = time.time() - t11
        logger.info("----------- Total %0.2f sec." % plane_times["deconvolution"])
    else:
        logger.warn("WARNING: skipping spike detection (settings['do_deconvolution']=False)")
        spks = np.zeros_like(F)
    
    # save results
    np.save(os.path.join(save_path, "stat.npy"), stat)
    np.save(os.path.join(save_path, "F.npy"), F)
    np.save(os.path.join(save_path, "Fneu.npy"), Fneu)
    if F_chan2 is not None:
        np.save(os.path.join(save_path, "F_chan2.npy"), F_chan2)
        np.save(os.path.join(save_path, "Fneu_chan2.npy"), Fneu_chan2)
    np.save(os.path.join(save_path, "spks.npy"), spks)

    logger.info("----------- ROI CLASSIFICATION")
    t11 = time.time()
    if len(stat):
        iscell = classification.classify(stat=stat, classfile=classfile)
    else:
        iscell = np.zeros((0, 2))
    np.save(os.path.join(save_path, "iscell.npy"), iscell)
    plane_times["classification"] = time.time() - t11

    plane_runtime = time.time() - t1
    plane_times["total_plane_runtime"] = plane_runtime
    # np.save(os.path.join(save_path, "timings.npy"), plane_times)

    if Zstack is not None:
        logger.info("----------- ZSTACK ALIGN")
        zcorr = zalign.register_to_zstack(f_reg, list(Zstack), nonrigid=False, 
                                          settings=settings["registration"],
                                          bidiphase=reg_outputs.get("bidiphase", 0), device=device)
    else:
        zcorr = np.zeros((0,))
    np.save(os.path.join(save_path, "zcorr.npy"), zcorr)

    logger.info(f"Plane processed in {plane_runtime:0.2f} sec (can open in GUI).")
    
    return (reg_outputs, detect_outputs, stat, F, Fneu, F_chan2, Fneu_chan2, 
            spks, iscell, redcell, zcorr, plane_times)
