import numpy as np
HAS_GPU=False
try:
    import cupy as cp
    from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
    HAS_GPU=True
except ImportError:
    HAS_GPU=False

def phasecorr_gpu(X, cfRefImg, lcorr):
    ''' not being used - no speed up - may be faster with cuda.jit'''
    nimg,Ly,Lx = X.shape
    ly,lx = cfRefImg.shape[-2:]
    lyhalf = int(np.floor(ly/2))
    lxhalf = int(np.floor(lx/2))

    # put on GPU
    ref_gpu = cp.asarray(cfRefImg)
    x_gpu = cp.asarray(X)

    # phasecorrelation
    x_gpu = fftn(x_gpu, axes=(1,2), overwrite_x=True) * np.sqrt(Ly-1) * np.sqrt(Lx-1)
    for t in range(x_gpu.shape[0]):
        tmp = x_gpu[t,:,:]
        tmp = cp.multiply(tmp, ref_gpu)
        tmp = cp.divide(tmp, cp.absolute(tmp) + 1e-5)
        x_gpu[t,:,:] = tmp
    x_gpu = ifftn(x_gpu, axes=(1,2), overwrite_x=True)  * np.sqrt(Ly-1) * np.sqrt(Lx-1)
    x_gpu = cp.fft.fftshift(cp.real(x_gpu), axes=(1,2))

    # get max index
    x_gpu = x_gpu[cp.ix_(np.arange(0,nimg,1,int),
                    np.arange(lyhalf-lcorr,lyhalf+lcorr+1,1,int),
                    np.arange(lxhalf-lcorr,lxhalf+lcorr+1,1,int))]
    ix = cp.argmax(cp.reshape(x_gpu, (nimg, -1)), axis=1)
    cmax = x_gpu[np.arange(0,nimg,1,int), ix]
    ymax,xmax = cp.unravel_index(ix, (2*lcorr+1,2*lcorr+1))
    cmax = cp.asnumpy(cmax).flatten()
    ymax = cp.asnumpy(ymax)
    xmax = cp.asnumpy(xmax)
    ymax,xmax = ymax-lcorr, xmax-lcorr
    return ymax, xmax, cmax
