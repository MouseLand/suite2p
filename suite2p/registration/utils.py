import os
import warnings

import numpy as np
from numba import vectorize, complex64
from numpy import fft

try:
    from mkl_fft import fft2, ifft2
except ModuleNotFoundError:
    warnings.warn("mkl_fft not installed.  Install it with conda: conda install mkl_fft", ImportWarning)

def one_photon_preprocess(data, ops):
    ''' pre filtering for one-photon data '''
    if ops['pre_smooth'] > 0:
        ops['pre_smooth'] = int(np.ceil(ops['pre_smooth']/2) * 2)
        data = spatial_smooth(data, ops['pre_smooth'])
    else:
        data = data.astype(np.float32)

    #for n in range(data.shape[0]):
    #    data[n,:,:] = laplace(data[n,:,:])
    ops['spatial_hp'] = int(np.ceil(ops['spatial_hp']/2) * 2)
    data = spatial_high_pass(data, ops['spatial_hp'])
    return data

@vectorize([complex64(complex64, complex64)], nopython=True, target = 'parallel')
def apply_dotnorm(Y, cfRefImg):
    eps0 = np.complex64(1e-5)
    x = Y / (eps0 + np.abs(Y))
    x = x*cfRefImg
    return x

def init_offsets(ops):
    """ initialize offsets for all frames """
    yoff = np.zeros((0,),np.float32)
    xoff = np.zeros((0,),np.float32)
    corrXY = np.zeros((0,),np.float32)
    if ops['nonrigid']:
        nb = ops['nblocks'][0] * ops['nblocks'][1]
        yoff1 = np.zeros((0,nb),np.float32)
        xoff1 = np.zeros((0,nb),np.float32)
        corrXY1 = np.zeros((0,nb),np.float32)
        offsets = [yoff,xoff,corrXY,yoff1,xoff1,corrXY1]
    else:
        offsets = [yoff,xoff,corrXY]
    return offsets

def bin_paths(ops, raw):
    """ set which binary is being aligned to """
    raw_file_align = []
    raw_file_alt = []
    reg_file_align = []
    reg_file_alt = []
    if raw:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                raw_file_align = ops['raw_file']
                raw_file_alt = ops['raw_file_chan2']
                reg_file_align = ops['reg_file']
                reg_file_alt = ops['reg_file_chan2']
            else:
                raw_file_align = ops['raw_file_chan2']
                raw_file_alt = ops['raw_file']
                reg_file_align = ops['reg_file_chan2']
                reg_file_alt = ops['reg_file']
        else:
            raw_file_align = ops['raw_file']
            reg_file_align = ops['reg_file']
    else:
        if ops['nchannels']>1:
            if ops['functional_chan'] == ops['align_by_chan']:
                reg_file_align = ops['reg_file']
                reg_file_alt = ops['reg_file_chan2']
            else:
                reg_file_align = ops['reg_file_chan2']
                reg_file_alt = ops['reg_file']
        else:
            reg_file_align = ops['reg_file']
    return reg_file_align, reg_file_alt, raw_file_align, raw_file_alt


def gaussian_fft(sig, Ly, Lx):
    ''' gaussian filter in the fft domain with std sig and size Ly,Lx '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    hgx = np.exp(-np.square(xx/sig) / 2)
    hgy = np.exp(-np.square(yy/sig) / 2)
    hgg = hgy * hgx
    hgg /= hgg.sum()
    fhg = np.real(fft2(fft.ifftshift(hgg))); # smoothing filter in Fourier domain
    return fhg

def spatial_taper(sig, Ly, Lx):
    ''' spatial taper  on edges with gaussian of std sig '''
    x = np.arange(0, Lx)
    y = np.arange(0, Ly)
    x = np.abs(x - x.mean())
    y = np.abs(y - y.mean())
    xx, yy = np.meshgrid(x, y)
    mY = y.max() - 2*sig
    mX = x.max() - 2*sig
    maskY = 1./(1.+np.exp((yy-mY)/sig))
    maskX = 1./(1.+np.exp((xx-mX)/sig))
    maskMul = maskY * maskX
    return maskMul

def spatial_smooth(data,N):
    ''' spatially smooth data using cumsum over axis=1,2 with window N'''
    pad = np.zeros((data.shape[0], int(N/2), data.shape[2]))
    dsmooth = np.concatenate((pad, data, pad), axis=1)
    pad = np.zeros((dsmooth.shape[0], dsmooth.shape[1], int(N/2)))
    dsmooth = np.concatenate((pad, dsmooth, pad), axis=2)
    # in X
    cumsum = np.cumsum(dsmooth, axis=1).astype(np.float32)
    dsmooth = (cumsum[:, N:, :] - cumsum[:, :-N, :]) / float(N)
    # in Y
    cumsum = np.cumsum(dsmooth, axis=2)
    dsmooth = (cumsum[:, :, N:] - cumsum[:, :, :-N]) / float(N)
    return dsmooth

def spatial_high_pass(data, N):
    ''' high pass filters data over axis=1,2 with window N'''
    norm = spatial_smooth(np.ones((1, data.shape[1], data.shape[2])), N).squeeze()
    data -= spatial_smooth(data, N) / norm
    return data

def get_nFrames(ops):
    """ get number of frames in binary file

    Parameters
    ----------
    ops : dictionary
        requires 'Ly', 'Lx', 'reg_file' (optional 'keep_movie_raw' and 'raw_file')

    Returns
    -------
    nFrames : int
        number of frames in the binary

    """

    if 'keep_movie_raw' in ops and ops['keep_movie_raw']:
        try:
            nbytes = os.path.getsize(ops['raw_file'])
        except:
            print('no raw')
            nbytes = os.path.getsize(ops['reg_file'])
    else:
        nbytes = os.path.getsize(ops['reg_file'])
    nFrames = int(nbytes/(2* ops['Ly'] *  ops['Lx']))
    return nFrames


def get_frames(ops, ix, bin_file, crop=False, badframes=False):
    """ get frames ix from bin_file
        frames are cropped by ops['yrange'] and ops['xrange']

    Parameters
    ----------
    ops : dict
        requires 'Ly', 'Lx'
    ix : int, array
        frames to take
    bin_file : str
        location of binary file to read (frames x Ly x Lx)
    crop : bool
        whether or not to crop by 'yrange' and 'xrange' - if True, needed in ops

    Returns
    -------
        mov : int16, array
            frames x Ly x Lx
    """
    if badframes and 'badframes' in ops:
        bad_frames = ops['badframes']
        try:
            ixx = ix[bad_frames[ix]==0].copy()
            ix = ixx
        except:
            notbad=True
    Ly = ops['Ly']
    Lx = ops['Lx']
    nbytesread =  np.int64(Ly*Lx*2)
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]
    if crop:
        mov = np.zeros((len(ix), Lyc, Lxc), np.int16)
    else:
        mov = np.zeros((len(ix), Ly, Lx), np.int16)
    # load and bin data
    with open(bin_file, 'rb') as bfile:
        for i in range(len(ix)):
            bfile.seek(nbytesread*ix[i], 0)
            buff = bfile.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            data = np.reshape(data, (Ly, Lx))
            if crop:
                mov[i,:,:] = data[ops['yrange'][0]:ops['yrange'][-1], ops['xrange'][0]:ops['xrange'][-1]]
            else:
                mov[i,:,:] = data
    return mov


def sampled_mean(ops):
    nframes = ops['nframes']
    nsamps = min(nframes, 1000)
    ix = np.linspace(0, nframes, 1+nsamps).astype('int64')[:-1]
    bin_file = ops['reg_file']
    if ops['nchannels']>1:
        if ops['functional_chan'] == ops['align_by_chan']:
            bin_file = ops['reg_file']
        else:
            bin_file = ops['reg_file_chan2']
    frames = utils.get_frames(ops, ix, bin_file, badframes=True)
    refImg = frames.mean(axis=0)
    return refImg


def subsample_frames(ops, bin_file, nsamps):
    """ get nsamps frames from binary file for initial reference image
    Parameters
    ----------
    ops : dictionary
        requires 'Ly', 'Lx', 'nframes'
    bin_file : open binary file
    nsamps : int
        number of frames to return
    Returns
    -------
    frames : int16
        frames x Ly x Lx
    """
    nFrames = ops['nframes']
    Ly = ops['Ly']
    Lx = ops['Lx']
    frames = np.zeros((nsamps, Ly, Lx), dtype='int16')
    nbytesread = 2 * Ly * Lx
    istart = np.linspace(0, nFrames, 1+nsamps).astype('int64')
    #istart = np.arange(nFrames - nsamps, nFrames).astype('int64')
    for j in range(0,nsamps):
        reg_file.seek(nbytesread * istart[j], 0)
        buff = reg_file.read_nwb(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        frames[j,:,:] = np.reshape(data, (Ly, Lx))
    reg_file.close()
    return frames