"""
Utilities to use skcuda.fft as a drop in replacement for
np.fft.fft2
2018-09-01, CSH
"""
import numpy as np
from numpy import fft

try:
    import pycuda.gpuarray as gpuarray
    import pycuda.driver as cudadrv
    import atexit
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# Set this to True to compare cu(i)fft2 results with
# np.fft.(i)fft2 results
TEST_CUDA = False

def load_cuda():
    from skcuda.fft import Plan
    from skcuda.fft import fft as cudafft
    from skcuda.fft import ifft as cudaifft

    global cudactx

    cudadrv.init()
    dev = cudadrv.Device(0)
    cudactx = dev.make_context()
    atexit.register(cudactx.pop)


def init_cuda_process(n=0):
    """
    Initialize a PyCUDA context at global scope so that it can be accessed
    from processes when using multithreading
    """
    # Initialization has to happen here to work with multiprocessing
    try:
        cudactx
        return
    except NameError:
        load_cuda()


def close_cuda_process(n=0):
    """
    Cleanup cuda process
    """
    import skcuda.misc as cudamisc
    try:
        cudamisc.done_context(cudactx)
        del(cudactx)
    except:
        pass


def cufft2(x):
    """
    Replacement of numpy.fft.fft2 using skcuda.fft
    Code based on
    https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
    """
    from skcuda.fft import Plan
    from skcuda.fft import fft as cudafft

    d1, d2 = x.shape[-2:]
    if x.ndim > 2:
        nbatch = x.shape[0]
        y_gpu = gpuarray.empty((nbatch, d1, int(d2/2)+1), np.complex64)
    else:
        nbatch = 1
        y_gpu = gpuarray.empty((d1, int(d2/2)+1), np.complex64)
    x_gpu = gpuarray.to_gpu(x.astype(np.float32))
    try:
        plan_forward = Plan((d1, d2), np.float32, np.complex64, nbatch)
    except Exception as e:
        raise RuntimeError("Likely out of memory on GPU; try reducing nimg_init and batch_size")

    cudafft(x_gpu, y_gpu, plan_forward)

    if d2%2 == 0:
        off = 1
    else:
        off = 0
    y = y_gpu.get()
    yout = np.empty(x.shape, dtype=y.dtype)
    if x.ndim > 2:
        yout[:, :, :int(d2/2)+1] = y
        yout[:, :, int(d2/2)+1:] = np.roll(
            y.conjugate()[:, :, off:-1][:, ::-1, ::-1], 1, axis=1)
    else:
        yout[:, :int(d2/2)+1] = y
        yout[:, int(d2/2)+1:] = np.roll(
            y.conjugate()[:, off:-1][::-1, ::-1], 1, axis=0)

    if TEST_CUDA:
        npfft = fft.fft2(x)
        print(x.shape)
        np.testing.assert_allclose(
            npfft, yout.astype('complex128'),
            rtol=1e-1, atol=np.abs(yout).max()*1e-6)
        print("Test succeeded", x.shape)
    return yout.astype('complex128')


def cuifft2(y):
    """
    Replacement of numpy.fft.ifft2 using skcuda.ifft
    Code based on
    https://www.idtools.com.au/gpu-accelerated-fft-compatible-numpy/
    """
    from skcuda.fft import Plan
    from skcuda.fft import ifft as cudaifft

    # Get the shape of the initial numpy array
    d1, d2 = y.shape[-2:]
    if y.ndim > 2:
        nbatch = y.shape[0]
    else:
        nbatch = 1

    # From numpy array to GPUarray. Take only the first d2/2+1 non redundant FFT coefficients
    y_gpu = gpuarray.to_gpu(y.astype(np.complex64))

    # Initialise empty output GPUarray 
    x_gpu = gpuarray.empty(y.shape, np.complex64)

    # Inverse FFT
    plan_backward = Plan((d1, d2), np.complex64, np.complex64, nbatch)
    cudaifft(y_gpu, x_gpu, plan_backward, scale=True)

    xout = x_gpu.get()

    if TEST_CUDA:
        np.testing.assert_allclose(fft.ifft2(y), xout, rtol=1e-1, atol=1e-1)
        print("Test succeeded")

    return xout
