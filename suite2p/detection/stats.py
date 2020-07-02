from typing import Tuple, Optional
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
from numpy.linalg import norm

from .utils import fitMVGaus, distance_kernel, norm_by_average


def mean_r_squared(y: np.ndarray, x: np.ndarray, estimator=np.median) -> float:
    return np.mean(norm(((y - estimator(y)), (x - estimator(x))), axis=0))


def calc_radii(dy: float, dx: float, ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray) -> Tuple[float, float]:
    return fitMVGaus(ypix / dy, xpix / dx, lam, 2).radii


def aspect_ratio(ry: float, rx: float, offset: float = .01) -> float:
    return 2 * ry / (ry + rx + offset)


@dataclass(frozen=True)
class ROI:
    ypix: np.ndarray
    xpix: np.ndarray
    lam: np.ndarray
    dx: Optional[int]
    dy: Optional[int]
    rsort: np.ndarray = field(default=np.sort(distance_kernel(radius=30).flatten()))

    def __post_init__(self):
        """Validate inputs."""
        if self.xpix.shape != self.ypix.shape or self.xpix.shape != self.lam.shape:
            raise TypeError("xpix, ypix, and lam should all be the same size.")

    @property
    def mean_r_squared(self) -> float:
        return mean_r_squared(y=self.ypix, x=self.xpix)

    @property
    def mean_r_squared0(self) -> float:
        return np.mean(self.rsort[:self.ypix.size])

    @property
    def mean_r_squared_compact(self) -> float:
        return self.mean_r_squared / (1e-10 + self.mean_r_squared0)

    @property
    def median_pix(self) -> Tuple[float, float]:
        return np.median(self.ypix), np.median(self.xpix)

    @property
    def n_pixels(self) -> int:
        return self.xpix.size

    @property
    def radii(self) -> Tuple[float, float]:
        if self.dx is None or self.dy is None:
            raise TypeError("dx and dy are required for calculating radii.")
        return calc_radii(dy=self.dy, dx=self.dx, xpix=self.xpix, ypix=self.ypix, lam=self.lam)

    @property
    def radius(self) -> float:
        return self.radii[0] * np.mean((self.dx, self.dy))

    @property
    def aspect_ratio(self) -> float:
        ry, rx = self.radii
        return aspect_ratio(ry=ry, rx=rx)



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
    warn("roi_stats() will be removed in a future release.  Use ROI instead.", PendingDeprecationWarning)

    for stat in stats:
        roi = ROI(ypix=stat['ypix'], xpix=stat['xpix'], lam=stat['lam'], dx=dx, dy=dy)
        stat['mrs'] = roi.mean_r_squared
        stat['mrs0'] = roi.mean_r_squared0
        stat['compact'] = roi.mean_r_squared_compact
        stat['med'] = list(roi.median_pix)
        stat['npix'] = roi.n_pixels
        if 'radius' not in stat:
            stat['radius'] = roi.radius
            stat['aspect_ratio'] = roi.aspect_ratio


    # todo: why specify the first 100?
    mrs_normeds = norm_by_average(values=[stat['mrs'] for stat in stats], estimator=np.nanmedian, offset=1e-10, first_n=100)
    npix_normeds = norm_by_average(values=[stat['npix'] for stat in stats], first_n=100)
    for stat, mrs_normed, npix_normed in zip(stats, mrs_normeds, npix_normeds):
        stat['mrs'] = mrs_normed
        stat['npix_norm'] = npix_normed
        stat['footprint'] = 0 if 'footprint' not in stat else stat['footprint']

    return np.array(stats)