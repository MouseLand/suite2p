from __future__ import annotations

from typing import Tuple, Optional, NamedTuple, Sequence, List
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
from numpy.linalg import norm

from .utils import norm_by_average


def distance_kernel(radius: int) -> np.ndarray:
    """ Returns 2D array containing geometric distance from center, with radius 'radius'"""
    d = np.arange(-radius, radius + 1)
    dists_2d = norm(np.meshgrid(d, d), axis=0)
    return dists_2d


class EllipseData(NamedTuple):
    mu: float
    cov: float
    radii: Tuple[float, float]
    ellipse: np.ndarray

    @property
    def area(self):
        return (self.radii[0] * self.radii[1]) ** 0.5 * np.pi


@dataclass(frozen=True)
class ROI:
    ypix: np.ndarray
    xpix: np.ndarray
    lam: np.ndarray
    dx: Optional[int]
    dy: Optional[int]
    rsort: np.ndarray = field(default=np.sort(distance_kernel(radius=30).flatten()), repr=False)

    def __post_init__(self):
        """Validate inputs."""
        if self.xpix.shape != self.ypix.shape or self.xpix.shape != self.lam.shape:
            raise TypeError("xpix, ypix, and lam should all be the same size.")

    @classmethod
    def get_overlap_count_image(cls, rois: Sequence[ROI], Ly: int, Lx: int) -> np.ndarray:
        return count_overlaps(Ly=Ly, Lx=Lx, ypixs=[roi.ypix for roi in rois], xpixs=[roi.xpix for roi in rois])

    @classmethod
    def filter_overlappers(cls, rois: Sequence[ROI], overlap_image: np.ndarray, max_overlap: float) -> np.ndarray:
        """returns logical array of rois that remain after removing those that overlap more than fraction max_overlap from overlap_img."""
        return filter_overlappers(
            ypixs=[roi.ypix for roi in rois],
            xpixs=[roi.xpix for roi in rois],
            overlap_image=overlap_image,
            max_overlap=max_overlap,
        )


    @property
    def mean_r_squared(self) -> float:
        return mean_r_squared(y=self.ypix, x=self.xpix)

    @property
    def mean_r_squared0(self) -> float:
        return np.mean(self.rsort[:self.ypix.size])

    @property
    def mean_r_squared_compact(self) -> float:
        return self.mean_r_squared / (1e-10 + self.mean_r_squared0)

    @classmethod
    def get_mean_r_squared_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average([roi.mean_r_squared for roi in rois], estimator=np.nanmedian, offset=1e-10, first_n=first_n)

    @property
    def median_pix(self) -> Tuple[float, float]:
        return np.median(self.ypix), np.median(self.xpix)

    @property
    def n_pixels(self) -> int:
        return self.xpix.size

    @classmethod
    def get_n_pixels_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average([roi.n_pixels for roi in rois], first_n=first_n)

    @property
    def fit_ellipse(self) -> EllipseData:
        if self.dx is None or self.dy is None:
            raise TypeError("dx and dy are required for fitting to an ellipse.")
        return fitMVGaus(self.ypix / self.dy, self.xpix / self.dx, self.lam, 2)

    @property
    def radii(self) -> Tuple[float, float]:
        return self.fit_ellipse.radii

    @property
    def radius(self) -> float:
        return self.radii[0] * np.mean((self.dx, self.dy))

    @property
    def aspect_ratio(self) -> float:
        ry, rx = self.radii
        return aspect_ratio(width=ry, height=rx)


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


def mean_r_squared(y: np.ndarray, x: np.ndarray, estimator=np.median) -> float:
    return np.mean(norm(((y - estimator(y)), (x - estimator(x))), axis=0))


def aspect_ratio(width: float, height: float, offset: float = .01) -> float:
    return 2 * width / (width + height + offset)


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


def count_overlaps(Ly: int, Lx: int, ypixs, xpixs) -> np.ndarray:
    overlap = np.zeros((Ly, Lx))
    for xpix, ypix in zip(xpixs, ypixs):
        overlap[ypix, xpix] += 1
    return overlap


def filter_overlappers(ypixs, xpixs, overlap_image: np.ndarray, max_overlap: float) -> List[bool]:
    """returns ROI indices that remain after removing those that overlap more than fraction max_overlap from overlap_img."""
    n_overlaps = overlap_image.copy()
    keep_rois = []
    for ypix, xpix in reversed(list(zip(ypixs, xpixs))):  # todo: is there an ordering effect here that affects which rois will be removed and which will stay?
        keep_roi = np.mean(n_overlaps[ypix, xpix] > 1) <= max_overlap
        keep_rois.append(keep_roi)
        if not keep_roi:
            n_overlaps[ypix, xpix] -= 1
    return keep_rois[::-1]