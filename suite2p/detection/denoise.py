"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
import time
from sklearn.decomposition import PCA
import logging 
logger = logging.getLogger(__name__)

from ..registration.nonrigid import make_blocks, spatial_taper


def pca_denoise(mov, block_size, n_comps_frac):
    """
    Denoise a movie using block-wise PCA reconstruction.

    Splits the movie into spatial blocks, projects each block onto its
    top PCA components, reconstructs, and blends the blocks with spatial
    tapering.

    Parameters
    ----------
    mov : numpy.ndarray
        Movie of shape (nframes, Ly, Lx). Modified in-place (mean
        subtracted then restored).
    block_size : list of int
        Block size [Lyb, Lxb] for spatial tiling.
    n_comps_frac : float
        Fraction of the smaller block dimension used to set the number
        of PCA components (number of PCs n_comps = min(Lyb, Lxb) * n_comps_frac).

    Returns
    -------
    reconstruction : numpy.ndarray
        Denoised movie of shape (nframes, Ly, Lx).
    """
    t0 = time.time()
    nframes, Ly, Lx = mov.shape
    yblock, xblock, nb, block_size = make_blocks(Ly, Lx, block_size=block_size)[:4]

    mov_mean = mov.mean(axis=0)
    mov -= mov_mean

    nblocks = len(yblock)
    Lyb, Lxb = block_size
    n_comps = int(min(min(Lyb * Lxb, nframes), min(Lyb, Lxb) * n_comps_frac))
    maskMul = spatial_taper(Lyb // 4, Lyb, Lxb).numpy()
    norm = np.zeros((Ly, Lx), np.float32)
    reconstruction = np.zeros_like(mov)
    block_re = np.zeros((nblocks, nframes, Lyb * Lxb))
    for i in range(nblocks):
        block = mov[:, yblock[i][0]:yblock[i][-1],
                    xblock[i][0]:xblock[i][-1]].reshape(-1, Lyb * Lxb)
        model = PCA(n_components=n_comps, random_state=0).fit(block)
        block_re[i] = (block @ model.components_.T) @ model.components_
        norm[yblock[i][0]:yblock[i][-1], xblock[i][0]:xblock[i][-1]] += maskMul

    block_re = block_re.reshape(nblocks, nframes, Lyb, Lxb)
    block_re *= maskMul
    for i in range(nblocks):
        reconstruction[:, yblock[i][0]:yblock[i][-1],
                       xblock[i][0]:xblock[i][-1]] += block_re[i]
    reconstruction /= norm
    logger.info("Binned movie denoised (for cell detection only) in %0.2f sec." %
          (time.time() - t0))
    reconstruction += mov_mean
    return reconstruction
