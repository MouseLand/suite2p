import gc
import numpy as np
from natsort import natsorted
import math, time
import glob, h5py, os, json
from scipy import signal
from scipy import stats, signal
from scipy.sparse import linalg
import scipy.io

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
