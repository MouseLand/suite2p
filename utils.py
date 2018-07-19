import numpy as np
import math

def fitMVGaus(y,x,lam,thres=None):
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
    if thres is None:
        thres = 2.5
    # make inputs (n,1)
    y = np.expand_dims(y.flatten(),axis=1)
    x = np.expand_dims(x.flatten(),axis=1)
    lam = np.expand_dims(lam.flatten(),axis=1)
    # normalize pixel weights
    lam = lam / lam.sum() 
    # mean of gaussian
    mu  = [(lam*y).sum(),(lam*x).sum()]
    y   = y - mu[0]
    x   = x - mu[1]
    # covariance of gaussian
    yx  = np.concatenate((y*lam**0.5,x*lam**0.5),axis=1)
    cov = yx.transpose() @ yx
    # radii of major and minor axes
    radii,evec  = np.linalg.eig(cov*thres)
    radii       = np.real(radii)
    # compute pts of ellipse
    n = 100
    p = np.expand_dims(np.linspace(0, 2*math.pi, n),axis=1)
    p = np.concatenate((np.cos(p), np.sin(p)),axis=1)
    ellipse = (p * radii**0.5) @ evec.transpose() + mu
    area = (radii[0] * radii[1])**0.5 * math.pi
    radii  = np.sort(radii)[::-1]
    return mu, cov, radii, ellipse, area