import time
from collections import namedtuple

import numpy as np
from scipy.ndimage import gaussian_filter


def bin_movie(ops):
    """ bin registered frames in 'reg_file' for ROI detection

    movie is binned then high-pass filtered to move slow changes

    Parameters
    ----------------

    ops : dictionary
        'Ly', 'Lx', 'yrange', 'xrange', 'tau', 'fs', 'nframes', 'high_pass', 'batch_size'
        (optional 'badframes')

    Returns
    ----------------

    mov : 3D array
        binned movie, size [nbins x Ly x Lx]

    max_proj : 2D array
        max projection image (mov.max(axis=0)) size [Ly x Lx]

    """
    t0 = time.time()
    badframes = False
    if 'badframes' in ops:
        badframes = True
        nframes = ops['nframes'] - ops['badframes'].sum()
    else:
        nframes = ops['nframes']
    bin_min = np.floor(nframes / ops['nbinned']).astype('int32');
    bin_min = max(bin_min, 1)
    bin_tau = np.round(ops['tau'] * ops['fs']).astype('int32');
    bin_size = max(bin_min, bin_tau)
    ops['nbinned'] = nframes // bin_size
    print('Binning movie in chunks of length %2.2d'%bin_size)
    Ly = ops['Ly']
    Lx = ops['Lx']
    Lyc = ops['yrange'][-1] - ops['yrange'][0]
    Lxc = ops['xrange'][-1] - ops['xrange'][0]

    nimgbatch = 500
    nimgbatch = min(nframes, nimgbatch)
    nimgbatch = bin_size * (nimgbatch // bin_size)
    nbytesread = np.int64(Ly*Lx*nimgbatch*2)
    mov = np.zeros((ops['nbinned'], Lyc, Lxc), np.float32)
    ix = 0
    idata = 0
    # load and bin data
    with open(ops['reg_file'], 'rb') as reg_file:
        while True:
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            buff = []
            nimgd = data.size // (Ly*Lx)
            if nimgd < bin_size:
                break
            data = np.reshape(data, (-1, Ly, Lx))
            dinds = idata + np.arange(0,data.shape[0],1,int)
            idata+=data.shape[0]
            if dinds[-1] >= ops['nframes']: # this only happens when ops['frames_include'] != -1
                break
            if badframes and np.sum(ops['badframes'][dinds])>.5:
                data = data[~ops['badframes'][dinds],:,:]
            nimgd = data.shape[0]
            if nimgd < nimgbatch:
                nmax = (nimgd // bin_size) * bin_size
                data = data[:nmax,:,:]
            dbin = np.reshape(data, (-1, bin_size, Ly, Lx))
            # crop into valid area
            mov[ix:ix+dbin.shape[0],:,:] = dbin[:, :,
                                                ops['yrange'][0]:ops['yrange'][-1],
                                                ops['xrange'][0]:ops['xrange'][-1]].mean(axis=1)
            ix += dbin.shape[0]
    mov = mov[:ix,:,:]
    max_proj = mov.max(axis=0)
    print('Binned movie [%d,%d,%d], %0.2f sec.'%(mov.shape[0], mov.shape[1], mov.shape[2], time.time()-t0))

    # data is high-pass filtered
    if ops['high_pass']<10:
        # slow high-pass
        for j in range(mov.shape[1]):
            mov[:,j,:] -= gaussian_filter(mov[:,j,:], [ops['high_pass'], 0])
    else:
        # fast approx high-pass
        hp = int(ops['high_pass'])
        for i in np.arange(0, mov.shape[0], hp):
            mov[i:i+hp,:,:] -= mov[i:i+hp,:,:].mean(axis=0)

    return mov, max_proj


def get_sdmov(mov, ops):
    """ computes standard deviation of difference between pixels across time

    difference between frames in binned movie computed then stddev
    helps to normalize image across pixels

    Parameters
    ----------------

    mov : 3D array
        size [nbins x Ly x Lx]

    ops : dictionary
        'batch_size'

    stat : array of dicts
        'ypix', 'xpix'

    Returns
    ----------------

    stat : array of dicts
        adds 'overlap'

    """
    ix = 0

    if len(mov.shape)>2:
        nbins,Ly, Lx = mov.shape
        npix = (Ly , Lx)
    else:
        nbins, npix = mov.shape
    batch_size = min(ops['batch_size'], nbins)
    sdmov = np.zeros(npix, 'float32')
    while 1:
        if ix>=nbins:
            break
        sdmov += (np.diff(mov[ix:ix+batch_size,:, :], axis = 0)**2).sum(axis=0)
        ix = ix + batch_size
    sdmov = np.maximum(1e-10, (sdmov/nbins)**0.5)
    return sdmov


EllipseData = namedtuple("EllipseData", "mu cov radii ellipse area")

def fitMVGaus(y, x, lam, thres=2.5, npts: int = 100) -> EllipseData:
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
    mu = (lam*yx).sum(axis=-1)
    yx = yx - np.expand_dims(mu, axis=1)
    yx = yx * lam ** .5
    cov = yx @ yx.T

    # radii of major and minor axes
    radii, evec = np.linalg.eig(cov)
    radii = np.maximum(0, np.real(radii))
    radii = thres * radii ** .5

    # compute pts of ellipse
    p = np.expand_dims(np.linspace(0, 2 * np.pi, npts), axis=1)
    p = np.concatenate((np.cos(p), np.sin(p)), axis=1)
    ellipse = (p * radii) @ evec.transpose() + mu
    area = (radii[0] * radii[1])**0.5 * np.pi
    radii = np.sort(radii)[::-1]

    return EllipseData(mu, cov, radii, ellipse, area)


def distance_kernel(radius: int) -> np.ndarray:
    """ Returns 2D array containing geometric distance from center, with radius 'radius'"""
    d = np.arange(-radius, radius + 1)
    dx, dy = np.meshgrid(d, d)
    dists_2d = (dy ** 2 + dx ** 2) ** 0.5
    return dists_2d