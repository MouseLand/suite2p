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
        #sfilt = s1
        dims = np.arange(-1,sdim)
        dims[0] = axes[i]
        dims = np.delete(dims, [axes[i]+1])
        
        sfilt = np.transpose(sfilt, dims)
        print(sfilt.shape)
        
        ns = sfilt.shape[0]
        flat = np.ones((ns,),np.float32)
        for j in range(0,sdim-1):
            flat = np.expand_dims(flat,axis=j+1)
        
        tmax = np.ceil(4*sig[i])
        dt = np.arange(-tmax,tmax)
        gaus = np.exp(-dt**2 / (2*sig[i]**2))
        gaus /= gaus.sum()
        for j in range(0,sdim-1):
            gaus = np.expand_dims(gaus,axis=j+1)
        print(gaus.shape)
        sfilt = signal.convolve(sfilt, gaus, mode='full')
        snorm = signal.convolve(flat, gaus, mode='full')
        print(sfilt.shape)
        if sfilt.shape[0] > ns:
            icent = int(np.floor(sfilt.shape[0]/2) - np.floor(ns/2))
            inds  = (icent + np.arange(0,ns)).astype(np.int32)
            if i==0:
                sout = sfilt[inds,:,:]
                snorm = snorm[inds[0]:inds[-1]+1]
            else:
                sout = sfilt
        #sout = sout / snorm
        dims = np.arange(1,sdim)
        dims = np.insert(dims, axes[i], 0)
        print(dims)
        sfilt = np.transpose(sout, dims)
        
    return sfilt

def conv_circ(s1, sig, axes=None):
    return s1
    