"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time

from tqdm import trange
import numpy as np
import torch
from scipy import stats

from .masks import create_masks
from .. import default_ops

def extract_traces(f_in, cell_masks, neuropil_masks, batch_size=500):
    """
    Extracts activity from f_in using cell_masks and neuropil_masks.

    Parameters:
        f_in : np.ndarray or io.BinaryFile object
            Size n_frames, Ly, Lx.
        cell_masks : list
            List of tuples where the first element is cell pixels (flattened) and the second element is pixel weights normalized to sum 1 (lam).
        neuropil_masks : list
            List of neuropil pixels in (Ly*Lx) coordinates.
        batch_size : int
            Maximum batch size (default: 1000).

    Returns:
        F : float, 2D array
            Size [ROIs x time].
        Fneu : float, 2D array
            Size [ROIs x time].
        ops : dictionary
    """
    n_frames, Ly, Lx = f_in.shape
    t0 = time.time()
    batch_size = min(batch_size, 1000)
    ncells = len(cell_masks)

    device = torch.device("cuda")

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
    print(f"Extracting fluorescence from {n_frames} frames in {n_batches} batches")
    for n in trange(n_batches, mininterval=10):
        tstart, tend = n * batch_size, min((n+1) * batch_size, n_frames)
        data = torch.from_numpy(f_in[tstart : tend]).to(device)
        data = data.reshape(-1, Ly*Lx).float()
        
        Fneu_batch = (data @ nmasks) / npix_neuropil
        Fneu[:, tstart : tend] = Fneu_batch.T.cpu().numpy()
        
        F_batch = data @ cmasks 
        F[:, tstart : tend] = F_batch.T.cpu().numpy()
        
    return F, Fneu

def extraction_wrapper(stat, f_reg, f_reg_chan2=None, cell_masks=None,
                       neuropil_masks=None, ops=default_ops()["extraction"]):
    """
    Main fluorescence extraction function.

    Parameters:
        stat: array of dicts
        f_reg: array of functional frames, np.ndarray or io.BinaryFile
        f_reg_chan2: array of anatomical frames, np.ndarray or io.BinaryFile

    Returns:
        stat: list of dictionaries with added keys "skew" and "std"
        F: fluorescence of functional channel
        F_neu: neuropil of functional channel
        F_chan2: fluorescence of anatomical channel
        F_neu_chan2: neuropil of anatomical channel
    """
    n_frames, Ly, Lx = f_reg.shape
    batch_size = ops["batch_size"]
    if cell_masks is None:
        t10 = time.time()
        cell_masks, neuropil_masks0 = create_masks(stat, Ly, Lx, 
                                                   lam_percentile=ops["lam_percentile"], 
                                                allow_overlap=ops["allow_overlap"], 
                                                neuropil_extract=ops["neuropil_extract"], 
                                                inner_neuropil_radius=ops["inner_neuropil_radius"],
                                                min_neuropil_pixels=ops["min_neuropil_pixels"], 
                                                circular_neuropil=ops["circular_neuropil"])
        if neuropil_masks is None:
            neuropil_masks = neuropil_masks0
        print("Masks created, %0.2f sec." % (time.time() - t10))

    t0 = time.time()
    ncells = len(cell_masks)
    print("functional channel:")
    F, Fneu = extract_traces(f_reg, cell_masks, neuropil_masks, batch_size=batch_size)
    if f_reg_chan2 is not None:
        print("anatomical channel:")
        F_chan2, Fneu_chan2 = extract_traces(f_reg_chan2, cell_masks, neuropil_masks,
                                             batch_size=batch_size)
    else:
        F_chan2, Fneu_chan2 = [], []

    print("Extracted fluorescence from %d ROIs in %d frames, %0.2f sec." %
          (ncells, n_frames, time.time() - t0))

    
    return F, Fneu, F_chan2, Fneu_chan2