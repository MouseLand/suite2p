import numpy as np
import multiprocessing
from scipy.ndimage import filters
from multiprocessing import Pool
from numba import vectorize,float32,int32,int16,jit,njit,prange, complex64

@njit(['float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32'])
def oasis_trace(F, v, w, t, l, s, tau, fs):
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

@njit(['float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32'], parallel=True)
def oasis_matrix(F, v, w, t, l, s, tau, fs):
    for n in prange(F.shape[0]):
        oasis_trace(F[n], v[n], w[n], t[n], l[n], s[n], tau, fs)

def oasis(F, ops):
    NN,NT = F.shape
    F = F.astype(np.float32)
    batch_size = ops['batch_size']
    S = np.zeros((NN,NT), dtype=np.float32)
    for i in range(0, NN, batch_size):
        f = F[i:i+batch_size]
        v = np.zeros((f.shape[0],NT), dtype=np.float32)
        w = np.zeros((f.shape[0],NT), dtype=np.float32)
        t = np.zeros((f.shape[0],NT), dtype=np.int64)
        l = np.zeros((f.shape[0],NT), dtype=np.float32)
        s = np.zeros((f.shape[0],NT), dtype=np.float32)
        oasis_matrix(f, v, w, t, l, s, ops['tau'], ops['fs'])
        S[i:i+batch_size] = s
    return S

def preprocess(F,ops):
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
