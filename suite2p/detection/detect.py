"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
<<<<<<< HEAD

from . import sourcery, sparsedetect, chan2detect, utils
from .stats import roi_stats
from .denoise import pca_denoise
from ..io.binary import BinaryFile
from ..classification import classify, user_classfile
from .. import default_ops


def detect(ops, classfile=None):

    t0 = time.time()
    bin_size = int(
        max(1, ops["nframes"] // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))
    print("Binning movie in chunks of length %2.2d" % bin_size)
    with BinaryFile(filename=ops["reg_file"], Ly=ops["Ly"], Lx=ops["Lx"]) as f:
        mov = f.bin_movie(
            bin_size=bin_size,
            bad_frames=ops.get("badframes"),
            y_range=ops["yrange"],
            x_range=ops["xrange"],
        )
        print("Binned movie [%d,%d,%d] in %0.2f sec." %
              (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))

        ops, stat = detection_wrapper(f, mov=mov, ops=ops, classfile=classfile)

    return ops, stat


def bin_movie(f_reg, bin_size, yrange=None, xrange=None, badframes=None):
    """ bin registered movie """
=======
from tqdm import trange
import torch
import logging 
logger = logging.getLogger(__name__)

from . import sourcery, sparsedetect, chan2detect, utils, anatomical
from .stats import roi_stats
from .denoise import pca_denoise
from ..classification import classify, user_classfile
from .. import default_settings 
from ..logger import TqdmToLogger

cellpose_options_num = {'max_proj / meanImg': 1, 'meanImg':2, 'enhanced_meanImg': 3 ,'max_proj': 4}

def bin_movie(f_reg, bin_size, yrange=None, xrange=None, badframes=None, nbins=5000):
    """
    Temporally bin the registered movie.

    Reads batches of frames, excludes bad frames, crops to the valid
    region, and averages groups of ``bin_size`` frames to produce a
    downsampled movie.

    Parameters
    ----------
    f_reg : numpy.ndarray or BinaryFile
        Registered movie of shape (n_frames, Ly, Lx).
    bin_size : int
        Number of frames to average per bin.
    yrange : list of int, optional
        Two-element list [y_start, y_end] defining the Y crop range.
    xrange : list of int, optional
        Two-element list [x_start, x_end] defining the X crop range.
    badframes : numpy.ndarray, optional
        Boolean array of shape (n_frames,) where True marks frames to exclude.
    nbins : int, optional (default 5000)
        Maximum number of output binned frames.

    Returns
    -------
    mov : numpy.ndarray
        Binned movie of shape (num_binned_frames, Lyc, Lxc), dtype float32.
    """
>>>>>>> suite2p_dev/tomerge
    n_frames = f_reg.shape[0]
    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    batch_size = min(good_frames.sum(), 500)
    Lyc = yrange[1] - yrange[0]
    Lxc = xrange[1] - xrange[0]

    # Number of binned frames is rounded down when binning frames
<<<<<<< HEAD
    num_binned_frames = n_frames // bin_size
=======
    num_binned_frames = min(nbins, n_frames // bin_size)
>>>>>>> suite2p_dev/tomerge
    mov = np.zeros((num_binned_frames, Lyc, Lxc), np.float32)
    curr_bin_number = 0
    t0 = time.time()

    # Iterate over n_frames to maintain binning over TIME
<<<<<<< HEAD
    for k in np.arange(0, n_frames, batch_size):
        data = f_reg[k:min(k + batch_size, n_frames)]

        # exclude badframes
        good_indices = good_frames[k:min(k + batch_size, n_frames)]
=======
    tstarts = np.arange(0, n_frames, batch_size)
    n_batches = min(nbins // (batch_size // bin_size), len(tstarts))
    tstarts = tstarts[np.linspace(0, len(tstarts) - 1, n_batches, dtype="int")]
    
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        tstart = tstarts[n]
        tend = min(tstart + batch_size, n_frames)
        data = f_reg[tstart : tend]
        
        # exclude badframes
        good_indices = good_frames[tstart : tend]
>>>>>>> suite2p_dev/tomerge
        if good_indices.mean() > 0.5:
            data = data[good_indices]

        # crop to valid region
        if yrange is not None and xrange is not None:
            data = data[:, slice(*yrange), slice(*xrange)]

        # bin in time
        if data.shape[0] > bin_size:
            # Downsample by binning via reshaping and taking mean of each bin
            # only if current batch size exceeds or matches bin_size
            n_d = data.shape[0]
            data = data[:(n_d // bin_size) * bin_size]
            data = data.reshape(-1, bin_size, Lyc, Lxc).astype(np.float32).mean(axis=1)
        else:
            # Current batch size is below bin_size (could have many bad frames in this batch)
            # Downsample taking the mean of batch to get a single bin
            data = data.mean(axis=0)[np.newaxis, :, :]
        # Only fill in binned data if not exceeding the number of bins mov has
        if mov.shape[0] > curr_bin_number:
            # Fill in binned data
<<<<<<< HEAD
            n_bins = data.shape[0]
            mov[curr_bin_number:curr_bin_number + n_bins] = data
            curr_bin_number += n_bins

    print("Binned movie of size [%d,%d,%d] created in %0.2f sec." %
          (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    return mov


def detection_wrapper(f_reg, mov=None, yrange=None, xrange=None, ops=default_ops(),
                      classfile=None):
    """
	Main detection function. 

	Identifies ROIs. 

	Parameters
	----------------

	f_reg : np.ndarray or io.BinaryWFile,
		n_frames x Ly x Lx

	mov : ndarray (t x Lyc x Lxc)
			binned movie

	yrange : list of length 2
		Range of pixels along the y-axis of mov the detection module will be run on 
	
	xrange : list of length 2
		Range of pixels along the x-axis of mov the detection module will be run on 

	ops : dictionary or list of dicts

	classfile: string (optional, default None)
		path to saved classifier

	Returns
	----------------

	ops : dictionary or list of dicts
		
	stat : dictionary "ypix", "xpix", "lam"
		Dictionary containing statistics for ROIs


	"""
    n_frames, Ly, Lx = f_reg.shape
    yrange = ops.get("yrange", [0, Ly]) if yrange is None else yrange
    xrange = ops.get("xrange", [0, Lx]) if xrange is None else xrange
    ops["yrange"] = yrange
    ops["xrange"] = xrange

    if mov is None:
        bin_size = int(
            max(1, n_frames // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))
        print("Binning movie in chunks of length %2.2d" % bin_size)
        mov = bin_movie(f_reg, bin_size, yrange=yrange, xrange=xrange,
                        badframes=ops.get("badframes", None))
=======
            nb = data.shape[0]
            mov[curr_bin_number:curr_bin_number + nb] = data
            curr_bin_number += nb
    mov = mov[:curr_bin_number]
    logger.info("Binned movie of size [%d,%d,%d] created in %0.2f sec." %
          (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    return mov

def detection_wrapper(f_reg, diameter=[12., 12.], tau=1., fs=30, meanImg_chan2=None,
                      yrange=None, xrange=None, badframes=None, mov=None, 
                      preclassify=0., classifier_path=None, 
                      settings=default_settings()["detection"],
                      device=torch.device("cuda")):
    """
    Run the full ROI detection pipeline on a registered movie.

    Bins the movie in time, optionally denoises and high-pass filters,
    detects ROIs using the selected algorithm (sparsery, sourcery, or
    cellpose), computes ROI statistics, and optionally preclassifies and
    detects red cells in a second channel.

    Parameters
    ----------
    f_reg : numpy.ndarray or BinaryFile
        Registered movie of shape (n_frames, Ly, Lx).
    diameter : list of float, optional (default [12., 12.])
        Expected cell diameter [dy, dx] in pixels.
    tau : float, optional (default 1.)
        Timescale of the indicator in seconds, used to set bin size.
    fs : float, optional (default 30)
        Sampling rate in Hz, used with tau to set bin size.
    meanImg_chan2 : numpy.ndarray, optional
        Mean image of the second channel, shape (Ly, Lx). If provided,
        red cell detection is performed.
    yrange : list of int, optional
        Two-element list [y_start, y_end] defining the Y crop range.
    xrange : list of int, optional
        Two-element list [x_start, x_end] defining the X crop range.
    badframes : numpy.ndarray, optional
        Boolean array of shape (n_frames,) marking frames to exclude.
    mov : numpy.ndarray, optional
        Pre-binned movie of shape (nbinned, Lyc, Lxc). If provided,
        skips the binning step.
    preclassify : float, optional (default 0.)
        If positive, apply a classifier and remove ROIs with probability
        below this threshold before final statistics.
    classifier_path : str, optional
        Path to a saved classifier file. If None and preclassify > 0,
        uses the default user classifier.
    settings : dict, optional
        Detection settings dictionary.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for cellpose-based detection.

    Returns
    -------
    new_settings : dict
        Dictionary with detection metadata including "meanImg_crop",
        "max_proj", "diameter", and algorithm-specific keys.
    stat : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix",
        "xpix", "lam", "med", and other computed statistics.
    redcell : numpy.ndarray or None
        Array of shape (n_cells, 2) with red cell labels and probabilities,
        or None if no second channel was provided.
    """
    
    n_frames, Ly, Lx = f_reg.shape
    yrange = [0, Ly] if yrange is None else yrange
    xrange = [0, Lx] if xrange is None else xrange
    
    if mov is None:
        nbins = settings["nbins"]
        bin_size = int(max(1, n_frames // nbins, np.round(tau * fs)))
        #bin_size = int(max(1, np.round(tau * fs)))
        logger.info("Binning movie in chunks of %2.2d frames" % bin_size)
        mov = bin_movie(f_reg, bin_size, yrange=yrange, xrange=xrange,
                        badframes=badframes, nbins=nbins)
>>>>>>> suite2p_dev/tomerge
    else:
        if mov.shape[1] != yrange[-1] - yrange[0]:
            raise ValueError("mov.shape[1] is not same size as yrange")
        elif mov.shape[2] != xrange[-1] - xrange[0]:
            raise ValueError("mov.shape[2] is not same size as xrange")

<<<<<<< HEAD
    if "meanImg" not in ops:
        ops["meanImg"] = mov.mean(axis=0)
        ops["max_proj"] = mov.max(axis=0)

    if ops.get("inverted_activity", False):
=======
    if settings.get("inverted_activity", False):
>>>>>>> suite2p_dev/tomerge
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()

<<<<<<< HEAD
    if ops.get("denoise", 1):
        mov = pca_denoise(
            mov, block_size=[ops["block_size"][0] // 2, ops["block_size"][1] // 2],
            n_comps_frac=0.5)

    if ops.get("anatomical_only", 0):
        try:
            from . import anatomical
            CELLPOSE_INSTALLED = True
        except Exception as e:
            print("Warning: cellpose did not import")
            print(e)
            print("cannot use anatomical mode, but otherwise suite2p will run normally")
            CELLPOSE_INSTALLED = False
        if not CELLPOSE_INSTALLED:
            print(
                "~~~ tried to import cellpose to run anatomical but failed, install with: ~~~"
            )
            print("$ pip install cellpose")
        else:
            print(">>>> CELLPOSE finding masks in " +
                  ["max_proj / mean_img", "mean_img", "enhanced_mean_img", "max_proj"][
                      int(ops["anatomical_only"]) - 1])
            stat = anatomical.select_rois(ops=ops, mov=mov,
                                          diameter=ops.get("diameter", None))
    else:
        stat = select_rois(
            ops=ops,
            mov=mov,
            sparse_mode=ops["sparse_mode"],
            classfile=classfile,
        )

    ymin = int(yrange[0])
    xmin = int(xrange[0])
    if len(stat) > 0:
        for s in stat:
            s["ypix"] += ymin
            s["xpix"] += xmin
            s["med"][0] += ymin
            s["med"][1] += xmin

        if ops["preclassify"] > 0:
            if classfile is None:
                print(f"NOTE: Applying user classifier at {str(user_classfile)}")
                classfile = user_classfile

            stat = roi_stats(stat, Ly, Lx, aspect=ops.get("aspect", None),
                             diameter=ops.get("diameter",
                                              None), do_crop=ops.get("soma_crop", 1))
            if len(stat) == 0:
                iscell = np.zeros((0, 2))
            else:
                iscell = classify(stat=stat, classfile=classfile)
            np.save(Path(ops["save_path"]).joinpath("iscell.npy"), iscell)
            ic = (iscell[:, 0] > ops["preclassify"]).flatten().astype("bool")
            stat = stat[ic]
            print("Preclassify threshold %0.2f, %d ROIs removed" % (ops["preclassify"],
                                                                    (~ic).sum()))

        stat = roi_stats(stat, Ly, Lx, aspect=ops.get("aspect", None),
                         diameter=ops.get("diameter",
                                          None), max_overlap=ops["max_overlap"],
                         do_crop=ops.get("soma_crop", 1))
        print("After removing overlaps, %d ROIs remain" % (len(stat)))

    # if second channel, detect bright cells in second channel
    if "meanImg_chan2" in ops:
        if "chan2_thres" not in ops:
            ops["chan2_thres"] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops["save_path"]).joinpath("redcell.npy"), redcell)

    return ops, stat


def select_rois(ops: Dict[str, Any], mov: np.ndarray, sparse_mode: bool = True,
                classfile: Path = None):

    t0 = time.time()
    if sparse_mode:
        ops.update({"Lyc": mov.shape[1], "Lxc": mov.shape[2]})
        new_ops, stat = sparsedetect.sparsery(
            mov=mov,
            high_pass=ops["high_pass"],
            neuropil_high_pass=ops["spatial_hp_detect"],
            batch_size=ops["batch_size"],
            spatial_scale=ops["spatial_scale"],
            threshold_scaling=ops["threshold_scaling"],
            max_iterations=250 * ops["max_iterations"],
            percentile=ops.get("active_percentile", 0.0),
        )
        ops.update(new_ops)
    else:
        ops, stat = sourcery.sourcery(mov=mov, ops=ops)

    print("Detected %d ROIs, %0.2f sec" % (len(stat), time.time() - t0))
=======
    if settings.get("denoise", False):
        mov = pca_denoise(
            mov, block_size=settings["block_size"],
            n_comps_frac=0.5)

    meanImg = mov.mean(axis=0) 

    mov = utils.temporal_high_pass_filter(mov=mov, width=settings["highpass_time"])
    max_proj = mov.max(axis=0) 
    
    t0 = time.time()
    if settings["algorithm"] == "cellpose":
        if anatomical.CELLPOSE_INSTALLED:
            logger.info(f">>>> CELLPOSE finding masks in {settings['cellpose_settings']['img']}")
            new_settings, stat = anatomical.select_rois(meanImg, max_proj, settings=settings["cellpose_settings"],
                                          diameter=diameter,
                                          device=device)
        else:
            logger.info("Warning: cellpose did not import ", anatomical.cellpose_error)
            logger.info("cannot use anatomical mode, will use functional detection instead")

    if settings["algorithm"] != "cellpose" or not anatomical.CELLPOSE_INSTALLED:
        settings["algorithm"] = "sparsery" if settings["algorithm"] == "cellpose" else settings["algorithm"]
        sdmov = utils.standard_deviation_over_time(mov, batch_size=1000)
        if settings["algorithm"] == "sparsery":
            new_settings, stat = sparsedetect.sparsery(
                mov=mov, sdmov=sdmov,
                threshold_scaling=settings["threshold_scaling"],
                **settings["sparsery_settings"]
            )
        else:
            new_settings, stat = sourcery.sourcery(mov=mov, sdmov=sdmov, diameter=diameter,
                                              threshold_scaling=settings["threshold_scaling"],
                                              **settings["sourcery_settings"])
    logger.info("Detected %d ROIs, %0.2f sec" % (len(stat), time.time() - t0))
>>>>>>> suite2p_dev/tomerge
    stat = np.array(stat)

    if len(stat) == 0:
        raise ValueError(
<<<<<<< HEAD
            "no ROIs were found -- check registered binary and maybe change spatial scale"
        )

    # add ROI stat to stat
    #stat = roi_stats(stat, dy, dx, Ly, Lx, max_overlap=max_overlap, do_crop=do_crop)

    return stat
=======
            "no ROIs were found -- check registered binary and maybe try changing spatial scale / diameter / threshold_scaling"
        )

    # move ROIs to original coordinates
    ymin, xmin = int(yrange[0]), int(xrange[0])
    for s in stat:
        s["ypix"] += ymin; s["xpix"] += xmin; 
        if "med" in s:
            s["med"][0] += ymin; s["med"][1] += xmin

    if preclassify > 0.:
        if classifier_path is None:
            logger.info(f"NOTE: Applying user classifier at {str(user_classfile)}")
            classifier_path = user_classfile

        n0 = len(stat)
        stat = roi_stats(stat, Ly, Lx, diameter=diameter, 
                        max_overlap=None,
                        do_soma_crop=settings.get("soma_crop", 1), 
                        npix_norm_min=settings.get("npix_norm_min", 0.),
                        npix_norm_max=settings.get("npix_norm_max", 100.),
                        median=settings["algorithm"]=="cellpose")
        #import pdb; pdb.set_trace()
        iscell = classify(stat=stat, classfile=classifier_path)
        ic = (iscell[:, 1] > preclassify).flatten().astype("bool")
        stat = stat[ic]
        
        if len(stat) == 0:
            raise ValueError("no ROIs passed preclassify -- turn off preclassify or lower threshold_scaling")

        logger.info(f"preclassify threshold {preclassify}, {(~ic).sum()} ROIs removed")
        
    stat = roi_stats(stat, Ly, Lx, diameter=diameter, 
                        max_overlap=settings.get("max_overlap", 0.75),
                        do_soma_crop=settings.get("soma_crop", 1), 
                        npix_norm_min=settings.get("npix_norm_min", 0.),
                        npix_norm_max=settings.get("npix_norm_max", 100.),
                        median=settings["algorithm"]=="cellpose")
    logger.info("After removing by overlaps and npix_norm, %d ROIs remain" % (len(stat)))

    # if second channel, detect bright cells in second channel
    if meanImg_chan2 is not None:
        extraction_defaults = default_settings()["extraction"]
        redmasks, redcell = chan2detect.detect(meanImg, meanImg_chan2, stat, diameter=diameter,
                                    cellpose_chan2=settings.get("cellpose_chan2", False),
                                    chan2_threshold=settings.get("chan2_threshold", 0.65),
                                    settings=settings['cellpose_settings'],
                                    inner_neuropil_radius=extraction_defaults["inner_neuropil_radius"],
                                    min_neuropil_pixels=extraction_defaults["min_neuropil_pixels"],
                                    device=device)
        new_settings["chan2_masks"] = redmasks
    else:
        redcell = None

    new_settings["meanImg_crop"] = meanImg
    new_settings["max_proj"] = max_proj
    new_settings["diameter"] = diameter

    return new_settings, stat, redcell

 
>>>>>>> suite2p_dev/tomerge
