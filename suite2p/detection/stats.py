import numpy as np

from .utils import fitMVGaus, distance_kernel


def mean_r_squared(y, x, estimator=np.median):
    return np.mean(np.sqrt((y - estimator(y)) ** 2 + ((x - estimator(x)) ** 2)))


def calc_radii(dy, dx, ypix, xpix, lam):
    return fitMVGaus(ypix / dy, xpix / dx, lam, 2).radii


def aspect_ratio(ry, rx) -> float:
    return 2 * ry / (.01 + ry + rx)


def roi_stats(dy: int, dx: int, stats):
    """ computes statistics of ROIs

    Parameters
    ----------
    diameters : (dy, dx)

    stats : dictionary
        'ypix', 'xpix', 'lam'

    Returns
    -------
    stat : dictionary
        adds 'npix', 'npix_norm', 'med', 'footprint', 'compact', 'radius', 'aspect_ratio'

    """
    rs = distance_kernel(radius=30)
    rsort = np.sort(rs.flatten())
    for stat in stats:
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']

        # compute compactness of ROI
        mrs_val = mean_r_squared(y=ypix, x=xpix)
        stat['mrs'] = mrs_val
        stat['mrs0'] = np.mean(rsort[:ypix.size])
        stat['compact'] = stat['mrs'] / (1e-10 + stat['mrs0'])
        stat['med'] = [np.median(ypix), np.median(xpix)]
        stat['npix'] = xpix.size

        if 'radius' not in stat:
            radius = calc_radii(dy=dy, dx=dx, xpix=xpix, ypix=ypix, lam=lam)
            stat['radius'] = radius[0] * np.mean((dx, dy))
            stat['aspect_ratio'] = aspect_ratio(ry=radius[0], rx=radius[1])


    mmrs = np.nanmedian([stat['mrs'] for stat in stats[:100]])  # todo: why only include the first 100?
    for stat in stats:
        stat['mrs'] = stat['mrs'] / (1e-10 + mmrs)

    npix = np.array([stat['npix'] for stat in stats], dtype='float32')
    npix /= np.mean(npix[:100])  # todo: why only include the first 100?
    for stat, npix0 in zip(stats, npix):
        stat['npix_norm'] = npix0

    for stat in stats:
        if 'footprint' not in stat:
            stat['footprint'] = 0

    return np.array(stats)