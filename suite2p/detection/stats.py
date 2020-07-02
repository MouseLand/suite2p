from typing import Tuple

import numpy as np
from numpy.linalg import norm

from .utils import fitMVGaus, distance_kernel


def mean_r_squared(y: np.ndarray, x: np.ndarray, estimator=np.median) -> float:
    return np.mean(norm(((y - estimator(y)), (x - estimator(x))), axis=0))


def calc_radii(dy: float, dx: float, ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray) -> Tuple[float, float]:
    return fitMVGaus(ypix / dy, xpix / dx, lam, 2).radii


def aspect_ratio(ry: float, rx: float, offset: float = .01) -> float:
    return 2 * ry / (ry + rx + offset)


def norm_by_average(values: np.ndarray, estimator=np.mean, first_n: int = 100, offset: float = 0.) -> np.ndarray:
    """Returns array divided by the (average of the 'first_n' values + offset), calculating the average with 'estimator'."""
    return np.array(values, dtype='float32') / (estimator(values[:first_n]) + offset)


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
            radii = calc_radii(dy=dy, dx=dx, xpix=xpix, ypix=ypix, lam=lam)
            stat['radius'] = radii[0] * np.mean((dx, dy))
            stat['aspect_ratio'] = aspect_ratio(ry=radii[0], rx=radii[1])

    # todo: why specify the first 100?
    mrs_normeds = norm_by_average(values=[stat['mrs'] for stat in stats], estimator=np.nanmedian, offset=1e-10, first_n=100)
    npix_normeds = norm_by_average(values=[stat['npix'] for stat in stats], first_n=100)
    for stat, mrs_normed, npix_normed in zip(stats, mrs_normeds, npix_normeds):
        stat['mrs'] = mrs_normed
        stat['npix_norm'] = npix_normed
        stat['footprint'] = 0 if 'footprint' not in stat else stat['footprint']

    return np.array(stats)