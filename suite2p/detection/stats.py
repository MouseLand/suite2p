from __future__ import annotations

from typing import Tuple, Optional, NamedTuple, Sequence, List, Dict, Any
from dataclasses import dataclass, field
from warnings import warn

import numpy as np
from numpy.linalg import norm
from scipy.spatial import ConvexHull


def distance_kernel(radius: int) -> np.ndarray:
    """ Returns 2D array containing geometric distance from center, with radius 'radius'"""
    d = np.arange(-radius, radius + 1)
    dists_2d = norm(np.meshgrid(d, d), axis=0)
    return dists_2d

def median_pix(ypix, xpix):
    ymed, xmed = np.median(ypix), np.median(xpix)
    imin = np.argmin((xpix-xmed)**2 + (ypix-ymed)**2)
    xmed = xpix[imin]
    ymed = ypix[imin]
    return [ymed, xmed]

class EllipseData(NamedTuple):
    mu: float
    cov: float
    radii: Tuple[float, float]
    ellipse: np.ndarray
    dy: int
    dx: int

    @property
    def area(self):
        return (self.radii[0] * self.radii[1]) ** 0.5 * np.pi

    @property
    def radius(self) -> float:
        return self.radii[0] * np.mean((self.dx, self.dy))

    @property
    def aspect_ratio(self) -> float:
        ry, rx = self.radii
        return aspect_ratio(width=ry, height=rx)


@dataclass(frozen=True)
class ROI:
    ypix: np.ndarray
    xpix: np.ndarray
    lam: np.ndarray
    med: np.ndarray
    do_crop: bool
    rsort: np.ndarray = field(default=np.sort(distance_kernel(radius=30).flatten()), repr=False)

    def __post_init__(self):
        """Validate inputs."""
        if self.xpix.shape != self.ypix.shape or self.xpix.shape != self.lam.shape:
            raise TypeError("xpix, ypix, and lam should all be the same size.")

    @classmethod
    def from_stat_dict(cls, stat: Dict[str, Any]) -> ROI:
        return cls(ypix=stat['ypix'], xpix=stat['xpix'], lam=stat['lam'])

    def to_array(self, Ly: int, Lx: int) -> np.ndarray:
        """Returns a 2D boolean array of shape (Ly x Lx) indicating where the roi is located."""
        arr = np.zeros((Ly, Lx), dtype=float)
        arr[self.ypix, self.xpix] = 1
        return arr

    @classmethod
    def stats_dicts_to_3d_array(cls, stats: Sequence[Dict[str, Any]], Ly: int, Lx: int, label_id: bool = False):
        """
        Outputs a (roi x Ly x Lx) float array from a sequence of stat dicts.
        Convenience function that repeatedly calls ROI.from_stat_dict() and ROI.to_array() for all rois.

        Parameters
        ----------
        stats : List of dictionary 'ypix', 'xpix', 'lam'
        Ly : y size of frame
        Lx : x size of frame
        label_id : whether array should be an integer value indicating ROI id or just 1 (indicating precence of ROI).
        """
        arrays = []
        for i, stat in enumerate(stats):
            array = cls.from_stat_dict(stat=stat).to_array(Ly=Ly, Lx=Lx)
            if label_id:
                array *= i + 1
            arrays.append(array)
        return np.stack(arrays)

    def ravel_indices(self, Ly: int, Lx: int) -> np.ndarray:
        """Returns a 1-dimensional array of indices from the ypix and xpix coordinates, assuming an image shape Ly x Lx."""
        return np.ravel_multi_index((self.ypix, self.xpix), (Ly, Lx))

    @classmethod
    def get_overlap_count_image(cls, rois: Sequence[ROI], Ly: int, Lx: int) -> np.ndarray:
        return count_overlaps(Ly=Ly, Lx=Lx, ypixs=[roi.ypix for roi in rois], xpixs=[roi.xpix for roi in rois])

    @classmethod
    def filter_overlappers(cls, rois: Sequence[ROI], overlap_image: np.ndarray, max_overlap: float) -> List[bool]:
        """returns logical array of rois that remain after removing those that overlap more than fraction max_overlap from overlap_img."""
        return filter_overlappers(
            ypixs=[roi.ypix for roi in rois],
            xpixs=[roi.xpix for roi in rois],
            overlap_image=overlap_image,
            max_overlap=max_overlap,
        )

    def get_overlap_image(self, overlap_count_image: np.ndarray) -> np.ndarray:
        return overlap_count_image[self.ypix, self.xpix] > 1

    @property 
    def soma_crop(self) -> np.ndarray:
        if self.do_crop and self.ypix.size > 10:
            dists = ((self.ypix - self.med[0])**2 + (self.xpix - self.med[1])**2)**0.5
            radii = np.arange(0, dists.max(), 1)
            area = np.zeros_like(radii)
            for k, radius in enumerate(radii):
                area[k] = self.lam[dists < radius].sum()
            darea = np.diff(area)
            radius = radii[-1]
            threshold = darea.max() / 3
            if len(np.nonzero(darea > threshold)[0]) > 0:
                ida = np.nonzero(darea > threshold)[0][0]
                if len(np.nonzero(darea[ida:] < threshold)[0]):
                    radius = radii[np.nonzero(darea[ida:] < threshold)[0][0] + ida]
            crop = dists < radius
            if crop.sum()==0:
                crop = np.ones(self.ypix.size, 'bool')
            return crop
        else:
            return np.ones(self.ypix.size, 'bool')

    @property
    def mean_r_squared(self) -> float:
        return mean_r_squared(y=self.ypix[self.soma_crop], x=self.xpix[self.soma_crop])
        #return mean_r_squared(y=self.ypix[self.lam > self.lam.max()/5], x=self.xpix[self.lam > self.lam.max()/5])

    @property
    def mean_r_squared0(self) -> float:
        return np.mean(self.rsort[:self.npix_soma])
        #return np.mean(self.rsort[:self.ypix[self.lam > self.lam.max()/5].size])

    @property
    def mean_r_squared_compact(self) -> float:
        return self.mean_r_squared / (1e-10 + self.mean_r_squared0)

    @property
    def solidity(self) -> float:
        if self.npix_soma > 10:
            points = np.stack((self.ypix[self.soma_crop], 
                               self.xpix[self.soma_crop]), axis=1)
            try:
                hull = ConvexHull(points)
                volume = hull.volume
            except:
                volume = 10
        else:
            volume = 10
        return self.npix_soma / volume

    @classmethod
    def get_mean_r_squared_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average([roi.mean_r_squared for roi in rois], estimator=np.nanmedian, offset=1e-10, first_n=first_n)

    @property 
    def npix_soma(self) -> int:
        return self.soma_crop.sum()

    @property
    def n_pixels(self) -> int:
        return self.xpix.size

    @classmethod
    def get_n_pixels_normed_all(cls, rois: Sequence[ROI], first_n: int = 100) -> np.ndarray:
        return norm_by_average([roi.n_pixels for roi in rois], first_n=first_n)

    def fit_ellipse(self, dy: float, dx: float) -> EllipseData:
        return fitMVGaus(self.ypix[self.soma_crop], 
                         self.xpix[self.soma_crop], 
                         self.lam[self.soma_crop], 
                         dy=dy, dx=dx, thres=2)


def roi_stats(stat, dy: int, dx: int, Ly: int, Lx: int, max_overlap=None,
              do_crop=True):
    """
    computes statistics of ROIs
    Parameters
    ----------
    stat : dictionary
        'ypix', 'xpix', 'lam'
    diameters : (dy, dx)
    
    FOV size : (Ly, Lx)
    Returns
    -------
    stat : dictionary
        adds 'npix', 'npix_norm', 'med', 'footprint', 'compact', 'radius', 'aspect_ratio'
    """
    if 'med' not in stat[0]:
        for s in stat:
            s['med'] = median_pix(s['ypix'], s['xpix'])
    
    rois = [ROI(ypix=s['ypix'], xpix=s['xpix'], 
                lam=s['lam'], med=s['med'], do_crop=do_crop) for s in stat]
    n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
    for roi, s in zip(rois, stat):
        s['mrs'] = roi.mean_r_squared
        s['mrs0'] = roi.mean_r_squared0
        s['compact'] = roi.mean_r_squared_compact
        s['solidity'] = roi.solidity
        s['npix'] = roi.n_pixels
        s['npix_soma'] = roi.npix_soma
        s['soma_crop'] = roi.soma_crop
        s['overlap'] = roi.get_overlap_image(n_overlaps)
        ellipse = roi.fit_ellipse(dy, dx)
        s['radius'] = ellipse.radius
        s['aspect_ratio'] = ellipse.aspect_ratio

    mrs_normeds = norm_by_average(
        values=np.array([s['mrs'] for s in stat]), estimator=np.nanmedian, offset=1e-10, first_n=100
    )
    npix_normeds = norm_by_average(
        values=np.array([s['npix'] for s in stat]), first_n=100
    )
    npix_soma_normeds = norm_by_average(
        values=np.array([s['npix_soma'] for s in stat]), first_n=100
    )
    for s, mrs_normed, npix_normed, npix_soma_normed in zip(stat, mrs_normeds, npix_normeds, npix_soma_normeds):
        s['mrs'] = mrs_normed
        s['npix_norm_no_crop'] = npix_normed
        s['npix_norm'] = npix_soma_normed
        s['footprint'] = 0 if 'footprint' not in s else s['footprint']

    if  max_overlap is not None and max_overlap<1.0:
        keep_rois = ROI.filter_overlappers(rois=rois, overlap_image=n_overlaps, max_overlap=max_overlap)
        stat = stat[keep_rois]
        n_overlaps = ROI.get_overlap_count_image(rois=rois, Ly=Ly, Lx=Lx)
        rois = [ROI(ypix=s['ypix'], xpix=s['xpix'], 
                lam=s['lam'], med=s['med'], do_crop=do_crop) for s in stat]
        for roi, s in zip(rois, stat):
            s['overlap'] = roi.get_overlap_image(n_overlaps)
        
    return stat


def mean_r_squared(y: np.ndarray, x: np.ndarray, estimator=np.median) -> float:
    return np.mean(norm(((y - estimator(y)), (x - estimator(x))), axis=0))


def aspect_ratio(width: float, height: float, offset: float = .01) -> float:
    return 2 * width / (width + height + offset)


def fitMVGaus(y, x, lam0, dy, dx, thres=2.5, npts: int = 100) -> EllipseData:
    """ computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations.
    Parameters
    ----------
    y : float, array
        pixel locations in y
    x : float, array
        pixel locations in x
    lam0 : float, array
        weights of each pixel
    """
    y = y / dy
    x = x / dx

    # normalize pixel weights
    lam = lam0.copy()
    ix = lam > 0#lam.max()/5
    y, x, lam = y[ix], x[ix], lam[ix]
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
    return EllipseData(mu=mu, cov=cov, radii=radii, ellipse=ellipse, dy=dy, dx=dx)

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

def norm_by_average(values: np.ndarray, estimator=np.mean, first_n: int = 100, offset: float = 0.) -> np.ndarray:
    """Returns array divided by the (average of the 'first_n' values + offset), calculating the average with 'estimator'."""
    return np.array(values, dtype='float32') / (estimator(values[:first_n]) + offset)