"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np

from .utils import circleMask

def median_pix(ypix, xpix):
    ymed, xmed = np.median(ypix), np.median(xpix)
    imin = np.argmin((xpix - xmed)**2 + (ypix - ymed)**2)
    xmed = xpix[imin]
    ymed = ypix[imin]
    return [ymed, xmed]

def fitMVGaus(y, x, lam0, dy, dx, thres=2.5, npts: int = 100):
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
    ix = lam > 0  #lam.max()/5
    y, x, lam = y[ix], x[ix], lam[ix]
    lam /= lam.sum()

    # mean of gaussian
    yx = np.stack((y, x))
    mu = (lam * yx).sum(axis=1)
    yx = (yx - mu[:, np.newaxis]) * lam**.5
    cov = yx @ yx.T

    # radii of major and minor axes
    radii, evec = np.linalg.eig(cov)
    radii = thres * np.maximum(0, np.real(radii))**.5

    # compute pts of ellipse
    theta = np.linspace(0, 2 * np.pi, npts)
    p = np.stack((np.cos(theta), np.sin(theta)))
    ellipse = (p.T * radii) @ evec.T + mu
    radii = np.sort(radii)[::-1]
    return mu, cov, radii, ellipse

def soma_crop(ypix, xpix, lam, med):
    crop = np.ones(ypix.size, "bool")
    if len(ypix) > 10:
        dists = ((ypix - med[0])**2 + (xpix - med[1])**2)**0.5
        radii = np.arange(0, dists.max(), 1)
        area = np.zeros_like(radii)
        for k, radius in enumerate(radii):
            area[k] = lam[dists < radius].sum()
        darea = np.diff(area)
        radius = radii[-1]
        threshold = darea.max() / 3
        if len(np.nonzero(darea > threshold)[0]) > 0:
            ida = np.nonzero(darea > threshold)[0][0]
            if len(np.nonzero(darea[ida:] < threshold)[0]):
                radius = radii[np.nonzero(darea[ida:] < threshold)[0][0] + ida]
        crop = dists < radius
    if crop.sum() == 0:
        crop = np.ones(ypix.size, "bool")
    return crop
    
def roi_stats(stats, Ly: int, Lx: int, diameter=[12., 12.], max_overlap=0.75,
              do_soma_crop=True, npix_norm_min=0.5, npix_norm_max=2.0,
              median=False):
    """
    Computes statistics of cells found using sourcery.

    Args:
        stats (list): List of dictionaries containing the statistics of cells.
        Ly (int): Height of the image.
        Lx (int): Width of the image.
        diameter (list or np.ndarray, optional): Diameter of the cells. Defaults to None.
        max_overlap (float, optional): Maximum overlap allowed between cells. Defaults to 0.75.
        do_soma_crop (bool, optional): Flag indicating whether to crop dendritic pixels for computing compactness and aspect ratio. Defaults to True.

    Returns:
        list: List of dictionaries containing the updated statistics of cells.

    Raises:
        None

    Examples:
        stats = roi_stats(stats, Ly, Lx, aspect=1.5, diameter=10, max_overlap=0.8, do_soma_crop=True)
    """
   
    # approx size of masks for ROI aspect ratio estimation
    dy, dx = diameter[0], diameter[1]
    d0 = np.array([int(dy), int(dx)])

    rs, dy, dx = circleMask(d0)
    dists_disk = np.sort(rs.flatten())

    for k, stat in enumerate(stats):
        ypix, xpix, lam = stat["ypix"].copy(), stat["xpix"].copy(), stat["lam"].copy()
        med = stat.get("med", median_pix(ypix, xpix)) # median of pixels
        stat["med"], stat["npix"] = med, ypix.size
        
        # crop dendritic pixels out for computing compactness and aspect ratio
        if do_soma_crop:
            crop = soma_crop(ypix, xpix, lam, med)
            stat["soma_crop"] = crop
            ypix, xpix, lam = ypix[crop], xpix[crop], lam[crop]
            med = median_pix(ypix, xpix)
        else:
            stat["soma_crop"] = np.ones(ypix.size, "bool")
        stat["med_soma"], stat["npix_soma"] = med, stat["soma_crop"].sum()  
        
        # compute compactness of ROI
        dists = (((ypix - med[0]) / d0[0])**2 + ((xpix - med[1]) / d0[1])**2)**0.5
        stat["mrs"], stat["mrs0"] = dists.mean(), dists_disk[:ypix.size].mean()
        stat["compact"] = stat["mrs"] / (1e-10 + stat["mrs0"])
        
        # compute aspect ratio
        if "radius" not in stat:
            radii = fitMVGaus(ypix, xpix, lam, dy=d0[0], dx=d0[1], thres=2)[2]
            stat["radius"] = radii[0] * d0.mean()
            stat["aspect_ratio"] = 2 * radii[0] / (.01 + radii[0] + radii[1])
        stat["footprint"] = 0

    ### compute npix_norm (normalized npix) for each ROI
    npix_soma = np.array([stat["npix_soma"] for stat in stats], dtype="float32")
    npix = np.array([stat["npix"] for stat in stats], dtype="float32")
    # use median if cellpose, otherwise use best neurons to determine normalizer
    npix_soma /= np.median(npix_soma) if median else npix_soma[:100].mean()
    npix /= np.median(npix) if median else npix[:100].mean()
    
    keep_rois = (npix_norm_min <= npix_soma) * (npix_soma <= npix_norm_max)
    stats = stats[keep_rois]
    npix_soma, npix = npix_soma[keep_rois], npix[keep_rois]
    nremove = (~keep_rois).sum()
    print(f"Removed {nremove} ROIs with npix_norm < {npix_norm_min:.2f} or npix_norm > {npix_norm_max:.2f}")

    for stat, npix_soma0, npix0 in zip(stats, npix_soma, npix):
        stat["npix_norm"] = npix_soma0
        stat["npix_norm_no_crop"] = npix0       
    
    if max_overlap is not None and max_overlap < 1.0:
        overlap = np.zeros((Ly, Lx), "int")
        for stat in stats:
            overlap[stat["ypix"], stat["xpix"]] += 1
        
        keep_rois = np.zeros(len(stats), "bool")
        # remove overlapping ROIs in reversed order, because highest variance ROIs are first
        for k, stat in enumerate(stats[::-1]): 
            keep_roi = (overlap[stat["ypix"], stat["xpix"]] > 1).mean() <= max_overlap
            keep_rois[k] = keep_roi
            if not keep_roi:
                overlap[stat["ypix"], stat["xpix"]] -= 1
        keep_rois = keep_rois[::-1]
        stats = stats[keep_rois]
        
        for stat in stats:
            stat["overlap"] = (overlap[stat["ypix"], stat["xpix"]] > 1).astype("bool")

    nremove = (~keep_rois).sum()
    print(f"Removed {nremove} ROIs with overlap > {max_overlap}")

    return stats

def assign_overlaps(stats, Ly, Lx):
    overlap = np.zeros((Ly, Lx), "int")
    for stat in stats:
        overlap[stat["ypix"], stat["xpix"]] += 1
    for stat in stats:
        stat["overlap"] = (overlap[stat["ypix"], stat["xpix"]] > 1).astype("bool")
    return stats
