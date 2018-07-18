import numpy as np
from scipy import signal

def conv2(s1, sig, axes=None):
    if axes is None:
        axes = np.arange(0,len(sig))
    
    axes = np.array(axes)
    if isinstance(sig, (int, float)) or len(sig)==1:
        sig = sig*np.ones((len(axes),1),np.float32)
    elif len(sig)>1 and len(sig) != len(axes):
        raise ValueError('number of axes different from number of smoothing constants')
    
    s1 = np.array(s1).astype(np.float32)
    sdim = s1.ndim
    sig = np.array(sig).astype(np.float32)
    
    sfilt = s1
    for i in range(0,len(axes)):
        dims = np.arange(-1,sdim)
        dims[0] = axes[i]
        dims = np.delete(dims, [axes[i]+1])
        sfilt = np.transpose(sfilt, dims)
        print(dims)
        ns = sfilt.shape[0]
        
        tmax = np.ceil(4*sig[i])
        dt = np.arange(-tmax,tmax)
        gaus = np.exp(-dt**2 / (2*sig[i]**2))
        gaus /= gaus.sum()
        
        flat = np.ones((ns,),np.float32)
        snorm = signal.convolve(flat, gaus, mode='same')
        for j in range(0,sdim-1):
            gaus = np.expand_dims(gaus,axis=-1)
            snorm = np.expand_dims(snorm,axis=-1)
        print(gaus.shape)
        print(snorm.shape)
        t0 = time.time()
        sfilt = signal.convolve(sfilt, gaus, mode='same')
        print(time.time()-t0)
        
        #if sfilt.shape[0] > ns:
        #    icent = np.floor(sfilt.shape[0]/2) - np.floor(ns/2)
        #    inds  = (icent + np.arange(0,ns)).astype(np.int32)
        #    sfilt = sfilt[inds,:]
        #    snorm = snorm[inds]
        #sfilt = sfilt / snorm
        print(sfilt.shape)
        dims = np.arange(1,sdim)
        dims = np.insert(dims, axes[i], 0)
        sfilt = np.transpose(sfilt, dims)
        print(sfilt.shape)
    return sfilt

def conv_circ(s1, sig, axes=None):
    if axes is None:
        axes = np.arange(0,1)
        
    return s1
    