"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from tqdm import trange
from numba import njit, prange
from scipy.ndimage import maximum_filter1d, minimum_filter1d, gaussian_filter
import torch
from torch.nn.functional import conv1d, max_pool1d, pad
from ..logger import TqdmToLogger
import logging 
logger = logging.getLogger(__name__)

@njit([
    "float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32"
], cache=True)
def oasis_trace(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on a single neuron """
    NT = F.shape[0]
    g = -1. / (tau * fs)

    it = 0
    ip = 0

    while it < NT:
        v[ip], w[ip], t[ip], l[ip] = F[it], 1, it, 1
        while ip > 0:
            if v[ip - 1] * np.exp(g * l[ip - 1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * l[ip - 1])
                f2 = np.exp(2 * g * l[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                l[ip - 1] = l[ip - 1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1

    s[t[1:ip]] = v[1:ip] - v[:ip - 1] * np.exp(g * l[:ip - 1])


@njit([
    "float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32"
], parallel=True, cache=True)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on many neurons parallelized with prange  """
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)


def oasis(F: np.ndarray, batch_size: int, tau: float, fs: float) -> np.ndarray:
    """
    Computes non-negative deconvolution.

    No sparsity constraints.

    Parameters
    ----------
    F : ndarray
        Size [neurons x time], in pipeline uses neuropil-subtracted fluorescence.
    batch_size : int
        Number of neurons processed per batch.
    tau : float
        Timescale of the sensor, used for the deconvolution kernel.
    fs : float
        Sampling rate per plane.

    Returns
    -------
    S : ndarray
        Size [neurons x time], deconvolved fluorescence.
    """

    NN, NT = F.shape
    F = F.astype(np.float32)
    S = np.zeros((NN, NT), dtype=np.float32)
    n_batches = int(np.ceil(NN / batch_size))
    logger.info(f"Deconvolving {NN} neurons in {n_batches} batches")
    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for n in trange(n_batches, file=tqdm_out):
        i = n * batch_size
        f = F[i:i + batch_size]
        v = np.zeros((f.shape[0], NT), dtype=np.float32)
        w = np.zeros((f.shape[0], NT), dtype=np.float32)
        t = np.zeros((f.shape[0], NT), dtype=np.int64)
        l = np.zeros((f.shape[0], NT), dtype=np.float32)
        s = np.zeros((f.shape[0], NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, tau, fs)
        S[i:i + batch_size] = s
    return S

def preprocess(F: np.ndarray, baseline: str, win_baseline: float, sig_baseline: float,
               fs: float, prctile_baseline: float = 8) -> np.ndarray:
    """ preprocesses fluorescence traces for spike deconvolution

    baseline-subtraction with window "win_baseline"
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    baseline : str
        setting that describes how to compute the baseline of each trace

    win_baseline : float
        window (in seconds) for max filter

    sig_baseline : float
        width of Gaussian filter in frames

    fs : float
        sampling rate per plane

    prctile_baseline : float
        percentile of trace to use as baseline if using `prctile` for baseline
    
    Returns
    ----------------

    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence

    """
    win = int(win_baseline * fs)
    if baseline == "maximin":
        Flow = gaussian_filter(F, [0., sig_baseline])
        Flow = minimum_filter1d(Flow, win)
        Flow = maximum_filter1d(Flow, win)
    elif baseline == "constant":
        Flow = gaussian_filter(F, [0., sig_baseline])
        Flow = np.amin(Flow)
    elif baseline == "prctile":
        Flow = np.percentile(F, prctile_baseline, axis=1)
        Flow = np.expand_dims(Flow, axis=1)
    else:
        Flow = 0.

    F = F - Flow

    return F


def baseline_maximin(F: np.ndarray, win_baseline: float, sig_baseline: float,
                     fs: float, batch_size=100, device=torch.device('cuda')) -> np.ndarray:

    win = int(win_baseline * fs)
    win += 1 if win%2==0 else 0
    ncells, n_frames = F.shape
    Flow = np.zeros((ncells, n_frames), 'float32')
    batch_size = int(batch_size)
    n_batches = int(np.ceil(ncells / batch_size))
    gwid = int(np.round(sig_baseline * 3))
    gaussian = torch.exp(- torch.arange(-gwid, gwid + 1, 1, device=device)**2 /
                          (2 * sig_baseline**2))
    gaussian /= gaussian.sum()

    for n in range(n_batches):
        nstart, nend = n * batch_size, min((n+1) * batch_size, ncells)
        data = torch.from_numpy(F[nstart : nend]).to(device, dtype=torch.float)      

        data = pad(data, (gwid, gwid), 'replicate')
        data = conv1d(data.unsqueeze(1), gaussian.unsqueeze(0).unsqueeze(0), padding=0)

        data = pad(data, (win//2, win//2), 'replicate')
        data = -max_pool1d(-data, kernel_size=win, stride=1, padding= 0)

        data = pad(data, (win//2, win//2), 'replicate')
        data = max_pool1d(data, kernel_size=win, stride=1, padding= 0)

        Flow[nstart : nend] = data.squeeze().cpu().numpy()

    F = F - Flow

    return F 