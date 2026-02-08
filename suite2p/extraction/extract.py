"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time

from tqdm import trange
import numpy as np
import torch
from scipy import stats

import logging 
logger = logging.getLogger(__name__)


from .masks import create_masks
from .. import default_settings
from ..logger import TqdmToLogger

def extract_traces(f_in, cell_masks, neuropil_masks, batch_size=500, 
                    device = torch.device("cuda")):
    """
    Extract fluorescence traces using cell and neuropil masks.

    Performs sparse matrix multiplication on the GPU to extract ROI and
    neuropil traces from registered frames.

    Parameters
    ----------
    f_in : numpy.ndarray or BinaryFile
        Registered frames, shape (n_frames, Ly, Lx).
    cell_masks : list of tuple
        Each element is a tuple (pixel_indices, weights) for one ROI, where
        pixel_indices are flattened and weights are normalized to sum to 1.
    neuropil_masks : list of numpy.ndarray
        Each element is an array of flattened pixel indices for the neuropil
        mask of one ROI.
    batch_size : int, optional (default 500)
        Number of frames processed per batch.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for performing operations.

    Returns
    -------
    F : numpy.ndarray
        ROI fluorescence traces, shape (n_rois, n_frames).
    Fneu : numpy.ndarray
        Neuropil fluorescence traces, shape (n_rois, n_frames).
    """
    n_frames, Ly, Lx = f_in.shape
    t0 = time.time()
    batch_size = min(batch_size, 1000)
    ncells = len(cell_masks)

    # Check if mps and force to be on the CPU so that extraction can be run
    # TODO: Once, sparse_coo_tensor works on sparseMPS backend, we should remove this check
    if device.type == 'mps':
        device = torch.device('cpu')

    npix_neuropil = torch.Tensor([len(nm) for nm in neuropil_masks]).to(device)
    # create coo tensor of neuropil and cell masks 
    ccol_indices = [m for nm in neuropil_masks for m in nm]
    row_indices = [k for k in range(len(neuropil_masks)) for m in neuropil_masks[k]]
    inds = torch.Tensor([ccol_indices, row_indices]).to(device)
    # convert to csc (tried creating csc directly but it was slow)
    nmasks = torch.sparse_coo_tensor(inds, torch.ones(len(row_indices), device=device), 
                                     size=(Ly*Lx, ncells))
    nmasks = nmasks.to_sparse_csc()

    ccol_indices = [m for cm in cell_masks for m in cm[0]]
    row_indices = [k for k in range(len(cell_masks)) for m in cell_masks[k][0]]
    cell_lam = torch.Tensor([l for cm in cell_masks for l in cm[1]]).to(device)
    inds = torch.Tensor([ccol_indices, row_indices]).to(device)
    cmasks = torch.sparse_coo_tensor(inds, cell_lam, 
                                     size=(Ly*Lx, ncells))
    cmasks = cmasks.to_sparse_csc()

    F = np.zeros((ncells, n_frames), np.float32)
    Fneu = np.zeros((ncells, n_frames), np.float32)

    batch_size = int(batch_size)

    n_batches = int(np.ceil(n_frames / batch_size))
    logger.info(f"Extracting fluorescence from {n_frames} frames in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for n in trange(n_batches, mininterval=10, file=tqdm_out):
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        data = torch.from_numpy(f_in[tstart : tend]).to(device)
        data = data.reshape(-1, Ly*Lx).float()
        
        Fneu_batch = (data @ nmasks) / npix_neuropil
        Fneu[:, tstart : tend] = Fneu_batch.T.cpu().numpy()
        
        F_batch = data @ cmasks 
        F[:, tstart : tend] = F_batch.T.cpu().numpy()
        
    return F, Fneu

def extraction_wrapper(stat, f_reg, f_reg_chan2=None, cell_masks=None,
                       neuropil_masks=None, settings=default_settings()["extraction"],
                        device = torch.device("cuda")):
    """
    Main fluorescence extraction function.

    Creates cell and neuropil masks (if not provided) and extracts
    fluorescence traces for both functional and anatomical channels.

    Parameters
    ----------
    stat : numpy.ndarray
        Array of ROI statistics dictionaries, each containing "ypix", "xpix",
        "lam", "radius", and "overlap".
    f_reg : numpy.ndarray or BinaryFile
        Registered functional frames, shape (n_frames, Ly, Lx).
    f_reg_chan2 : numpy.ndarray or BinaryFile, optional (default None)
        Registered anatomical channel frames, shape (n_frames, Ly, Lx).
    cell_masks : list of tuple, optional (default None)
        Pre-computed cell masks. If None, masks are created from stat.
    neuropil_masks : list of numpy.ndarray, optional (default None)
        Pre-computed neuropil masks. If None, masks are created from stat.
    settings : dict
        Extraction settings dictionary.
    device : torch.device, optional (default torch.device("cuda"))
        Torch device for performing operations.

    Returns
    -------
    F : numpy.ndarray
        ROI fluorescence traces for functional channel, shape (n_rois, n_frames).
    Fneu : numpy.ndarray
        Neuropil fluorescence traces for functional channel, shape (n_rois, n_frames).
    F_chan2 : numpy.ndarray or None
        ROI fluorescence traces for anatomical channel, or None if f_reg_chan2
        is not provided.
    Fneu_chan2 : numpy.ndarray or None
        Neuropil fluorescence traces for anatomical channel, or None if
        f_reg_chan2 is not provided.
    """
    n_frames, Ly, Lx = f_reg.shape
    batch_size = settings["batch_size"]
    if cell_masks is None:
        t10 = time.time()
        cell_masks, neuropil_masks0 = create_masks(stat, Ly, Lx, 
                                                   lam_percentile=settings["lam_percentile"], 
                                                allow_overlap=settings["allow_overlap"], 
                                                neuropil_extract=settings["neuropil_extract"], 
                                                inner_neuropil_radius=settings["inner_neuropil_radius"],
                                                min_neuropil_pixels=settings["min_neuropil_pixels"], 
                                                circular_neuropil=settings["circular_neuropil"])
        if neuropil_masks is None:
            neuropil_masks = neuropil_masks0
        logger.info("Masks created, %0.2f sec." % (time.time() - t10))

    t0 = time.time()
    ncells = len(cell_masks)
    logger.info("functional channel:")
    F, Fneu = extract_traces(f_reg, cell_masks, neuropil_masks, 
                             batch_size=batch_size, device=device)
    if f_reg_chan2 is not None:
        logger.info("anatomical channel:")
        F_chan2, Fneu_chan2 = extract_traces(f_reg_chan2, cell_masks, neuropil_masks,
                                             batch_size=batch_size, device=device)
    else:
        F_chan2, Fneu_chan2 = None, None

    logger.info("Extracted fluorescence from %d ROIs in %d frames, %0.2f sec." %
          (ncells, n_frames, time.time() - t0))

    
    return F, Fneu, F_chan2, Fneu_chan2
