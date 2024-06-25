"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any
from tqdm import trange
import torch

from . import sourcery, sparsedetect, chan2detect, utils, anatomical
from .stats import roi_stats
from .denoise import pca_denoise
from ..classification import classify, user_classfile
from .. import default_ops

def bin_movie(f_reg, bin_size, yrange=None, xrange=None, badframes=None, nbins=5000):
    """ bin registered movie """
    n_frames = f_reg.shape[0]
    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    batch_size = min(good_frames.sum(), 500)
    Lyc = yrange[1] - yrange[0]
    Lxc = xrange[1] - xrange[0]

    # Number of binned frames is rounded down when binning frames
    num_binned_frames = min(nbins, n_frames // bin_size)
    mov = np.zeros((num_binned_frames, Lyc, Lxc), np.float32)
    curr_bin_number = 0
    t0 = time.time()

    # Iterate over n_frames to maintain binning over TIME
    tstarts = np.arange(0, n_frames, batch_size)
    n_batches = min(nbins // (batch_size // bin_size), len(tstarts))
    tstarts = tstarts[np.linspace(0, len(tstarts) - 1, n_batches, dtype="int")]
    for n in trange(n_batches, mininterval=10):
        tstart = tstarts[n]
        tend = min(tstart + batch_size, n_frames)
        data = f_reg[tstart : tend]

        # exclude badframes
        good_indices = good_frames[tstart : tend]
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
            nb = data.shape[0]
            mov[curr_bin_number:curr_bin_number + nb] = data
            curr_bin_number += nb
    mov = mov[:curr_bin_number]
    print("Binned movie of size [%d,%d,%d] created in %0.2f sec." %
          (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
    return mov


def detection_wrapper(f_reg, tau=1., fs=30, meanImg_chan2=None,
                      yrange=None, xrange=None, badframes=None, mov=None, 
                      classfile=None, ops=default_ops()["detection"],
                      device=torch.device("cuda")):
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
    yrange = [0, Ly] if yrange is None else yrange
    xrange = [0, Lx] if xrange is None else xrange
    
    if mov is None:
        bin_size = int(max(1, np.round(tau * fs)))
        print("Binning movie in chunks of %2.2d frames" % bin_size)
        mov = bin_movie(f_reg, bin_size, yrange=yrange, xrange=xrange,
                        badframes=badframes, nbins=ops["nbins"])
    else:
        if mov.shape[1] != yrange[-1] - yrange[0]:
            raise ValueError("mov.shape[1] is not same size as yrange")
        elif mov.shape[2] != xrange[-1] - xrange[0]:
            raise ValueError("mov.shape[2] is not same size as xrange")

    if ops.get("inverted_activity", False):
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()

    if ops.get("denoise", 1):
        mov = pca_denoise(
            mov, block_size=ops["block_size"],
            n_comps_frac=0.5)

    meanImg = mov.mean(axis=0) 

    mov = utils.temporal_high_pass_filter(mov=mov, width=ops["highpass_time"])
    max_proj = mov.max(axis=0) 
    
    t0 = time.time()
    if ops["algorithm"] == "cellpose":
        if anatomical.CELLPOSE_INSTALLED:
            print(">>>> CELLPOSE finding masks in " +
                  ["max_proj / mean_img", "mean_img", "enhanced_mean_img", "max_proj"][
                      int(ops["cellpose_img"]) - 1])
            new_ops, stat = anatomical.select_rois(meanImg, max_proj, ops=ops,
                                          yrange=yrange, xrange=xrange,
                                          diameter=ops.get("diameter", None), 
                                          device=device)
        else:
            print("Warning: cellpose did not import ", anatomical.cellpose_error)
            print("cannot use anatomical mode, will use functional detection instead")
            
    if ops["algorithm"] != "cellpose" or not anatomical.CELLPOSE_INSTALLED:
        ops["algorithm"] = "sparsery" if ops["algorithm"] == "cellpose" else ops["algorithm"]
        sdmov = utils.standard_deviation_over_time(mov, batch_size=ops["batch_size"])
        if ops["algorithm"] == "sparsery":
            new_ops, stat = sparsedetect.sparsery(
                mov=mov, sdmov=sdmov,
                threshold_scaling=ops["threshold_scaling"],
                **ops["sparsery_settings"]
            )
        else:
            new_ops, stat = sourcery.sourcery(mov=mov, sdmov=sdmov, diameter=ops["diameter"],
                                              threshold_scaling=ops["threshold_scaling"],
                                              **ops["sourcery_settings"])
    print("Detected %d ROIs, %0.2f sec" % (len(stat), time.time() - t0))
    stat = np.array(stat)

    if len(stat) == 0:
        raise ValueError(
            "no ROIs were found -- check registered binary and maybe try changing spatial scale / diameter / threshold_scaling"
        )

    # move ROIs to original coordinates
    ymin, xmin = int(yrange[0]), int(xrange[0])
    for s in stat:
        s["ypix"] += ymin; s["xpix"] += xmin; 
        if "med" in s:
            s["med"][0] += ymin; s["med"][1] += xmin

    stat = roi_stats(stat, Ly, Lx, aspect=ops.get("aspect", None),
                        diameter=ops.get("diameter", None), 
                        max_overlap=ops.get("max_overlap", 0.75),
                        do_soma_crop=ops.get("soma_crop", 1), 
                        npix_norm_min=ops.get("npix_norm_min", 0.5),
                        npix_norm_max=ops.get("npix_norm_max", 2.0),
                        median=ops["algorithm"]=="cellpose")
    print("After removing by size and overlaps, %d ROIs remain" % (len(stat)))

    # if second channel, detect bright cells in second channel
    if meanImg_chan2 is not None:
        redmasks, redcell = chan2detect.detect(meanImg, meanImg_chan2, stat, 
                                    cellpose_chan2=ops.get("cellpose_chan2", True),
                                    chan2_threshold=ops.get("chan2_threshold", 0.65),
                                    device=device)
        new_ops["chan2_masks"] = redmasks
    else:
        redcell = None

    new_ops["meanImg"] = meanImg
    new_ops["max_proj"] = max_proj

    return new_ops, stat, redcell

    # if ops["preclassify"] > 0:
    #     if classfile is None:
    #         print(f"NOTE: Applying user classifier at {str(user_classfile)}")
    #         classfile = user_classfile

    #     stat = roi_stats(stat, Ly, Lx, aspect=ops.get("aspect", None),
    #                         diameter=ops.get("diameter",
    #                                         None), do_soma_crop=ops.get("soma_crop", 1))
    #     if len(stat) == 0:
    #         raise ValueError("no ROIs passed preclassify -- turn off preclassify or lower threshold_scaling")

    #     iscell = classify(stat=stat, classfile=classfile)
    #     ic = (iscell[:, 0] > ops["preclassify"]).flatten().astype("bool")
    #     stat = stat[ic]
    #     print(f"preclassify threshold {ops['preclassify']}, {(~ic).sum()} ROIs removed")
