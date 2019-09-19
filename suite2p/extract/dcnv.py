import numpy as np
import multiprocessing
from scipy.ndimage import filters
from multiprocessing import Pool

def oasis1t(inputs):
    F, ops = inputs
    ca = F
    NT = F.shape[0]

    v = np.zeros((NT,))
    w = np.zeros((NT,))
    t = np.zeros((NT,), dtype=int)
    l = np.zeros((NT,))
    s = np.zeros((NT,))

    g = -1./(ops['tau'] * ops['fs'])

    for i in range(0,10):
        it = 0
        ip = 0

        while it<NT:
            v[ip], w[ip],t[ip],l[ip] = ca[it],1,it,1

            while ip>0:
                if v[ip-1] * np.exp(g * l[ip-1]) > v[ip]:
                    # violation of the constraint means merging pools
                    f1 = np.exp(g * l[ip-1])
                    f2 = np.exp(2 * g * l[ip-1])
                    wnew = w[ip-1] + w[ip] * f2
                    v[ip-1] = (v[ip-1] * w[ip-1] + v[ip] * w[ip]* f1) / wnew
                    w[ip-1] = wnew
                    l[ip-1] += l[ip]

                    ip += -1
                else:
                    break
            it += 1
            ip += 1

        s[t[1:ip]] = v[1:ip] - v[:ip-1] * np.exp(g * l[:ip-1])

        return s

def oasis(F, ops):
    num_cores = multiprocessing.cpu_count()
    F = preprocess(F,ops)
    inputs = range(F.shape[0])
    Fsplit = []
    for i in inputs:
        Fsplit.append((F[i,:], ops))
    with Pool(num_cores) as p:
        results = p.map(oasis1t, Fsplit)
    #results = Parallel(n_jobs=num_cores)(delayed(oasis1t)(F[i, :], ops) for i in inputs)
    # collect results as numpy array
    sp = np.zeros_like(F)
    for i in inputs:
        sp[i,:] = results[i]
    return sp


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
