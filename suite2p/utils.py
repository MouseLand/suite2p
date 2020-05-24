import math

import numpy as np


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
