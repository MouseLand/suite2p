import gc
import numpy as np
from natsort import natsorted
import math, time
import glob, h5py, os, json
from scipy import signal
from scipy import stats, signal
from scipy.sparse import linalg
import scipy.io

def default_ops():
    """ default options to run pipeline """
    ops = {
        # file paths
        'look_one_level_down': False, # whether to look in all subfolders when searching for tiffs
        'fast_disk': [], # used to store temporary binary file, defaults to save_path0
        'delete_bin': False, # whether to delete binary file after processing
        'mesoscan': False, # for reading in scanimage mesoscope files
        'bruker': False, # whether or not single page BRUKER tiffs!
        'h5py': [], # take h5py as input (deactivates data_path)
        'h5py_key': 'data', #key in h5py where data array is stored
        'save_path0': [], # stores results, defaults to first item in data_path
        'save_folder': [],
        'subfolders': [],
        'move_bin': False, # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk
        # main settings
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        'force_sktiff': False, # whether or not to use scikit-image for tiff reading
        'frames_include': -1,
        # output settings
        'preclassify': 0., # apply classifier before signal extraction with probability 0.3
        'save_mat': False, # whether to save output as matlab files
        'save_NWB': False, # whether to save output as NWB file
        'combined': True, # combine multiple planes into a single result /single canvas for GUI
        'aspect': 1.0, # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)
        # bidirectional phase offset
        'do_bidiphase': False,
        'bidiphase': 0,
        'bidi_corrected': False,
        # registration settings
        'do_registration': 1, # whether to register data (2 forces re-registration)
        'two_step_registration': False,
        'keep_movie_raw': False,
        'nimg_init': 300, # subsampled frames for finding reference image
        'batch_size': 500, # number of frames per batch
        'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'reg_tif': False, # whether to save registered tiffs
        'reg_tif_chan2': False, # whether to save channel 2 registered tiffs
        'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
        'smooth_sigma_time' : 0, # gaussian smoothing in time
        'smooth_sigma': 1.15, # ~1 good for 2P recordings, recommend >5 for 1P recordings
        'th_badframes': 1.0, # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        'pad_fft': False,
        # non rigid registration settings
        'nonrigid': True, # whether to use nonrigid registration
        'block_size': [128, 128], # block size to register (** keep this a multiple of 2 **)
        'snr_thresh': 1.2, # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        'maxregshiftNR': 5, # maximum pixel shift allowed for nonrigid, relative to rigid
        # 1P settings
        '1Preg': False, # whether to perform high-pass filtering and tapering
        'spatial_hp': 25, # window for spatial high-pass filtering before registration
        'pre_smooth': 2, # whether to smooth before high-pass filtering before registration
        'spatial_taper': 50, # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops['smooth_sigma'])
        # cell detection settings
        'roidetect': True, # whether or not to run ROI extraction
        'spikedetect': True, # whether or not to run spike deconvolution
        'sparse_mode': True, # whether or not to run sparse_mode
        'diameter': 12, # if not sparse_mode, use diameter for filtering and extracting
        'spatial_scale': 0, # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        'nbinned': 5000, # max number of binned frames for cell detection
        'max_iterations': 20, # maximum number of iterations to do cell detection
        'threshold_scaling': 1.0, # adjust the automatically determined threshold by this scalar multiplier
        'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
        'high_pass': 100, # running mean subtraction with window of size 'high_pass' (use low values for 1P)
        # ROI extraction parameters
        'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
        'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
        'allow_overlap': False, # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        # channel 2 detection settings (stat[n]['chan2'], stat[n]['not_chan2'])
        'chan2_thres': 0.65, # minimum for detection of brightness on channel 2
        # deconvolution settings
        'baseline': 'maximin', # baselining mode (can also choose 'prctile')
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
        'neucoeff': .7,  # neuropil coefficient
        'xrange': np.array([0, 0]),
        'yrange': np.array([0, 0]),
    }
    return ops


def boundary(ypix,xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    idist = ((ypix - ypix.transpose())**2 + (xpix - xpix.transpose())**2)
    idist[np.arange(0,npix),np.arange(0,npix)] = 500
    nneigh = (idist==1).sum(axis=1) # number of neighbors of each point
    iext = (nneigh<4).flatten()
    return iext

def circle(med, r):
    """ returns pixels of circle with radius 1.25x radius of cell (r) """
    theta = np.linspace(0.0,2*np.pi,100)
    x = r*1.25 * np.cos(theta) + med[0]
    y = r*1.25 * np.sin(theta) + med[1]
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    return x,y


def fitMVGaus(y,x,lam,thres=2.5):
    """ computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations.

    Parameters
    ----------
    y : float, array
        pixel locations in y
    x : float, array
        pixel locations in x
    lam : float, array
        weights of each pixel

    Returns
    -------
        mu : float
            mean of gaussian fit.
        cov : float
            covariance of gaussian fit.
        radii : float, array
            half of major and minor axis lengths of elliptical fit.
        ellipse : float, array
            coordinates of elliptical fit.
        area : float
            area of ellipse.

    """

    # normalize pixel weights
    lam /= lam.sum()
    # mean of gaussian
    yx = np.stack((y,x))
    mu  = (lam*yx).sum(axis=-1)
    yx = yx - np.expand_dims(mu, axis=1)
    yx = yx * lam**.5
    #yx  = np.concatenate((y*lam**0.5, x*lam**0.5),axis=0)
    cov = yx @ yx.transpose()
    # radii of major and minor axes
    radii,evec  = np.linalg.eig(cov)
    radii = np.maximum(0, np.real(radii))
    radii       = thres * radii**.5
    # compute pts of ellipse
    npts = 100
    p = np.expand_dims(np.linspace(0, 2*math.pi, npts),axis=1)
    p = np.concatenate((np.cos(p), np.sin(p)),axis=1)
    ellipse = (p * radii) @ evec.transpose() + mu
    area = (radii[0] * radii[1])**0.5 * math.pi
    radii  = np.sort(radii)[::-1]
    return mu, cov, radii, ellipse, area

def enhanced_mean_image(ops):
    """ computes enhanced mean image and adds it to ops

    Median filters ops['meanImg'] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops['meanImgE']

    Parameters
    ----------
    ops : dictionary
        uses 'meanImg', 'aspect', 'diameter', 'yrange' and 'xrange'

    Returns
    -------
        ops : dictionary
            'meanImgE' field added

    """

    I = ops['meanImg'].astype(np.float32)
    if 'spatscale_pix' not in ops:
        if isinstance(ops['diameter'], int):
            diameter = np.array([ops['diameter'], ops['diameter']])
        else:
            diameter = np.array(ops['diameter'])
        ops['spatscale_pix'] = diameter[1]
        ops['aspect'] = diameter[0]/diameter[1]

    diameter = 4*np.ceil(np.array([ops['spatscale_pix'] * ops['aspect'], ops['spatscale_pix']])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
        ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg
    return ops

def sub2ind(array_shape, rows, cols):
    inds = rows * array_shape[1] + cols
    return inds

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
