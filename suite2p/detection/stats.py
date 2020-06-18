import numpy as np

from suite2p.detection import masks, utils


def mean_r_squared(y, x, estimator=np.median):
    return np.mean(np.sqrt((y - estimator(y)) ** 2 + ((x - estimator(x)) ** 2)))


def roi_stats(ops, stats):
    """ computes statistics of ROIs

    Parameters
    ----------
    ops : dictionary
        'aspect', 'diameter'

    stats : dictionary
        'ypix', 'xpix', 'lam'

    Returns
    -------
    stat : dictionary
        adds 'npix', 'npix_norm', 'med', 'footprint', 'compact', 'radius', 'aspect_ratio'

    """
    if 'aspect' in ops:
        d0 = np.array([int(ops['aspect']*10), 10])
    else:
        d0 = ops['diameter']
        if isinstance(d0, int):
            d0 = [d0,d0]

    rs = masks.circle_mask(np.array([30, 30]))
    rsort = np.sort(rs.flatten())

    ncells = len(stats)
    mrs = np.zeros(ncells)
    for k, stat in enumerate(stats):
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']

        # compute compactness of ROI
        mrs_val = mean_r_squared(y=ypix, x=xpix)
        mrs[k] = mrs_val
        stat['mrs'] = mrs_val
        stat['mrs0'] = np.mean(rsort[:ypix.size])
        stat['compact'] = stat['mrs'] / (1e-10+stat['mrs0'])
        stat['med'] = [np.median(stat['ypix']), np.median(stat['xpix'])]
        stat['npix'] = xpix.size
        if 'radius' not in stat:
            radius = utils.fitMVGaus(ypix / d0[0], xpix / d0[1], lam, 2)[2]
            stat['radius'] = radius[0] * d0.mean()
            stat['aspect_ratio'] = 2 * radius[0]/(.01 + radius[0] + radius[1])
        if 'footprint' not in stat:
            stat['footprint'] = 0
        if 'med' not in stats:
            stat['med'] = [np.median(stat['ypix']), np.median(stat['xpix'])]

    npix = np.array([stats[n]['npix'] for n in range(len(stats))]).astype('float32')
    npix /= np.mean(npix[:100])

    mmrs = np.nanmedian(mrs[:100])
    for n in range(len(stats)):
        stats[n]['mrs'] = stats[n]['mrs'] / (1e-10 + mmrs)
        stats[n]['npix_norm'] = npix[n]
    stats = np.array(stats)

    return stats