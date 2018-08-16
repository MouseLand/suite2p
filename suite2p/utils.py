import numpy as np
import math
from scipy import signal

def fitMVGaus(y,x,lam,thres=2.5):
    ''' computes 2D gaussian fit to data and returns ellipse of radius thres standard deviations
    inputs:
        y, x, lam, thres
            y,x: pixel locations
            lam: pixel weights
        thres: number of standard deviations at which to draw ellipse
    outputs:
        mu, cov, ellipse, area
            mu: mean of gaussian fit
            cov: covariance of gaussian fit
            radii: half of major and minor axis lengths of elliptical fit
            ellipse: coordinates of elliptical fit
            area: area of ellipse
    '''
    # normalize pixel weights
    lam = lam / lam.sum()
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

def enhanced_mean_image(ops):
    if 1:
        I = ops['meanImg']
        Imed = signal.medfilt2d(I, 4*ops['diameter']+1)
        I = I - Imed
        Idiv = signal.medfilt2d(np.absolute(I), 4*ops['diameter']+1)
        I = I / (1e-10 + Idiv)
        mimg1 = -6
        mimg99 = 6
        mimg0 = I
    else:
        mimg0 = ops['meanImg']
        mimg0 = mimg0 - gaussian_filter(filters.minimum_filter(mimg0,50,mode='mirror'),
                                          10,mode='mirror')
        mimg0 = mimg0 / gaussian_filter(filters.maximum_filter(mimg0,50,mode='mirror'),
                                      10,mode='mirror')
        mimg1 = np.percentile(mimg0,1)
        mimg99 = np.percentile(mimg0,99)
    mimg0 = mimg0[ops['yrange'][0]:ops['yrange'][1], ops['xrange'][0]:ops['xrange'][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0,np.minimum(1,mimg0))
    mimg = mimg0.min() * np.ones((ops['Ly'],ops['Lx']),np.float32)
    mimg[ops['yrange'][0]:ops['yrange'][1],
        ops['xrange'][0]:ops['xrange'][1]] = mimg0
    ops['meanImgE'] = mimg
    return ops
