import numpy as np
from numba import njit, prange
from scipy.ndimage import filters


@njit(['float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32'], cache=True)
def oasis_trace(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on a single neuron """
    NT = F.shape[0]
    g = -1./(tau * fs)

    it = 0
    ip = 0

    while it<NT:
        v[ip], w[ip],t[ip],l[ip] = F[it],1,it,1
        while ip>0:
            if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * l[ip-1])
                f2 = np.exp(2 * g * l[ip-1])
                wnew = w[ip-1] + w[ip] * f2
                v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip]* f1) / wnew
                w[ip-1] = wnew
                l[ip-1] = l[ip-1] + l[ip]
                ip -= 1
            else:
                break
        it += 1
        ip += 1

    s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])

@njit(['float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32'], parallel=True, cache=True)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    """ spike deconvolution on many neurons parallelized with prange  """
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)


def oasis(F: np.ndarray, batch_size: int, tau: float, fs: float) -> np.ndarray:
    """ computes non-negative deconvolution

    no sparsity constraints
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    batch_size : int
        number of frames processed per batch

    tau : float
        timescale of the sensor, used for the deconvolution kernel

    fs : float
        sampling rate per plane


    Returns
    ----------------

    S : float, 2D array
        size [neurons x time], deconvolved fluorescence

    """
    NN,NT = F.shape
    F = F.astype(np.float32)
    S = np.zeros((NN,NT), dtype=np.float32)
    for i in range(0, NN, batch_size):
        f = F[i:i+batch_size]
        v = np.zeros((f.shape[0],NT), dtype=np.float32)
        w = np.zeros((f.shape[0],NT), dtype=np.float32)
        t = np.zeros((f.shape[0],NT), dtype=np.int64)
        l = np.zeros((f.shape[0],NT), dtype=np.float32)
        s = np.zeros((f.shape[0],NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, tau, fs)
        S[i:i+batch_size] = s
    return S


def preprocess(F: np.ndarray, ops):
    """ preprocesses fluorescence traces for spike deconvolution

    baseline-subtraction with window 'win_baseline'
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    ops : dictionary
        'baseline', 'win_baseline', 'sig_baseline', 'fs',
        (optional 'prctile_baseline' needed if ops['baseline']=='constant_prctile')
    
    Returns
    ----------------

    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence

    """
    sig = ops['sig_baseline']
    win = int(ops['win_baseline']*ops['fs'])
    if ops['baseline']=='maximin':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = filters.minimum_filter1d(Flow,    win)
        Flow = filters.maximum_filter1d(Flow,    win)
    elif ops['baseline']=='constant':
        Flow = filters.gaussian_filter(F,    [0., sig])
        Flow = np.amin(Flow)
    elif ops['baseline']=='constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    F = F - Flow

    return F
