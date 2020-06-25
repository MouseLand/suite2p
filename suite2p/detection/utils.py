import time
from typing import Tuple, NamedTuple

import numpy as np
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter


def bin_movie(Ly: int, Lx: int, ops):
    """ bin registered frames in 'reg_file' for ROI detection

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
    nframes = ops['nframes'] - ops['badframes'].sum() if 'badframes' in ops else ops['nframes']
    bin_size = int(max(1, nframes // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
    ops['nbinned'] = nframes // bin_size
    print('Binning movie in chunks of length %2.2d' % bin_size)

    nimgbatch = min(nframes, 500) // bin_size * bin_size
    nbytesread = Ly * Lx * nimgbatch * 2
    mov = np.zeros((ops['nbinned'], ops['yrange'][-1] - ops['yrange'][0], ops['xrange'][-1] - ops['xrange'][0]), np.float32)
    ix, idata = 0, 0
    # load and bin data
    with open(ops['reg_file'], 'rb') as reg_file:
        while True:
            buff = reg_file.read(nbytesread)
            data = np.frombuffer(buff, dtype=np.int16, offset=0)
            if data.size == 0:
                break
            data = data.reshape(-1, Ly, Lx)
            dinds = idata + np.arange(0, data.shape[0], 1, int)
            idata += data.shape[0]

            if dinds[-1] >= ops['nframes']:
                break
            if 'badframes' in ops and np.sum(ops['badframes'][dinds]) > .5:
                data = data[~ops['badframes'][dinds], :, :]
            nimgd = data.shape[0]
            if nimgd < nimgbatch:
                nmax = (nimgd // bin_size) * bin_size
                data = data[:nmax, :, :]
            dbin = np.reshape(data, (-1, bin_size, Ly, Lx))
            # crop into valid area
            mov[ix:ix+dbin.shape[0], :, :] = dbin[:, :,
                                                ops['yrange'][0]:ops['yrange'][-1],
                                                ops['xrange'][0]:ops['xrange'][-1]].mean(axis=1)
            ix += dbin.shape[0]
    mov = mov[:ix,:,:]
    max_proj = mov.max(axis=0)
    print('Binned movie [%d,%d,%d], %0.2f sec.'%(mov.shape[0], mov.shape[1], mov.shape[2], time.time()-t0))

    return mov, max_proj


def high_pass_gaussian_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a gaussian kernel."""
    mov = mov.copy()
    for j in range(mov.shape[1]):
        mov[:, j, :] -= gaussian_filter(mov[:, j, :], [width, 0])
    return mov


def high_pass_rolling_mean_filter(mov: np.ndarray, width: int) -> np.ndarray:
    """Returns a high-pass-filtered copy of the 3D array 'mov' using a rolling mean kernel over time."""
    mov = mov.copy()
    for i in np.arange(0, mov.shape[0], width):
        mov[i:i + width, :, :] -= mov[i:i + width, :, :].mean(axis=0)
    return mov


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


class EllipseData(NamedTuple):
    mu: float
    cov: float
    radii: Tuple[float, float]
    ellipse: np.ndarray

    @property
    def area(self):
        return (self.radii[0] * self.radii[1]) ** 0.5 * np.pi


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
    """

    # normalize pixel weights
    lam /= lam.sum()

    # mean of gaussian
    yx = np.stack((y, x))
    mu = (lam * yx).sum(axis=1)
    yx = (yx - mu[:, np.newaxis]) * lam ** .5
    cov = yx @ yx.T

    # radii of major and minor axes
    radii, evec = np.linalg.eig(cov)
    radii = thres * np.maximum(0, np.real(radii)) ** .5

    # compute pts of ellipse
    theta = np.linspace(0, 2 * np.pi, npts)
    p = np.stack((np.cos(theta), np.sin(theta)))
    ellipse = (p.T * radii) @ evec.T + mu
    radii = np.sort(radii)[::-1]

    return EllipseData(mu=mu, cov=cov, radii=radii, ellipse=ellipse)


def distance_kernel(radius: int) -> np.ndarray:
    """ Returns 2D array containing geometric distance from center, with radius 'radius'"""
    d = np.arange(-radius, radius + 1)
    dists_2d = norm(np.meshgrid(d, d), axis=0)
    return dists_2d
