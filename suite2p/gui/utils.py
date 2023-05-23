import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes

def boundary(ypix,xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    if npix>0:
        msk = np.zeros((np.ptp(ypix)+6, np.ptp(xpix)+6), np.bool)
        msk[ypix-ypix.min()+3, xpix-xpix.min()+3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)
        k = np.ones((3,3),dtype=int) # for 4-connected
        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        out = binary_dilation(msk==0, k) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext+ypix.min()-3, xext+xpix.min()-3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))
    return yext, xext


def circle(med, r):
    """ returns pixels of circle with radius 1.25x radius of cell (r) """
    theta = np.linspace(0.0,2*np.pi,100)
    x = r*1.25 * np.cos(theta) + med[0]
    y = r*1.25 * np.sin(theta) + med[1]
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    return x,y