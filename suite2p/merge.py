import numpy as np
import pyqtgraph as pg
from scipy import stats
from suite2p import utils, dcnv, celldetect2, fig
import math
from PyQt5 import QtGui
from matplotlib.colors import hsv_to_rgb
import time

def activity_stats(parent):
    i0      = int(1-parent.iscell[parent.ichosen])
    ypix = np.array([])
    xpix = np.array([])
    lam  = np.array([])
    footprints  = np.array([])
    F    = np.zeros((0,parent.Fcell.shape[1]), np.float32)
    Fneu = np.zeros((0,parent.Fcell.shape[1]), np.float32)
    for n in np.array(parent.imerge):
        ypix = np.append(ypix, parent.stat[n]["ypix"])
        xpix = np.append(xpix, parent.stat[n]["xpix"])
        lam = np.append(lam, parent.stat[n]["lam"])
        footprints = np.append(footprints, parent.stat[n]["footprint"])
        F    = np.append(F, parent.Fcell[n,:][np.newaxis,:], axis=0)
        Fneu = np.append(Fneu, parent.Fneu[n,:][np.newaxis,:], axis=0)

    # remove overlaps
    ipix = np.concatenate((ypix[:,np.newaxis], xpix[:,np.newaxis]), axis=1)
    _, goodi = np.unique(ipix, return_index=True, axis=0)
    ypix = ypix[goodi]
    xpix = xpix[goodi]
    lam = lam[goodi]
    stat0 = {}
    stat0["ypix"] = ypix.astype(np.int32)
    stat0["xpix"] = xpix.astype(np.int32)
    stat0["lam"] = lam
    stat0['med']  = [np.median(stat0["ypix"]), np.median(stat0["xpix"])]
    stat0["npix"] = ypix.size
    d0 = parent.ops["diameter"]
    radius = utils.fitMVGaus(ypix / d0[0], xpix / d0[1], lam, 2)[2]
    stat0["radius"] = radius[0] * d0.mean()
    stat0["aspect_ratio"] = 2 * radius[0]/(.01 + radius[0] + radius[1])
    npix = np.array([parent.stat[n]['npix'] for n in range(len(parent.stat))]).astype('float32')
    stat0["npix_norm"] = stat0["npix"] / npix.mean()
    # compactness
    rs,dy,dx = celldetect2.circleMask(d0)
    rsort = np.sort(rs.flatten())
    r2 = ((ypix - stat0["med"][0])/d0[0])**2 + ((xpix - stat0["med"][1])/d0[1])**2
    r2 = r2**.5
    stat0["mrs"]  = np.mean(r2)
    stat0["mrs0"] = np.mean(rsort[:r2.size])
    stat0["compact"] = stat0["mrs"] / (1e-10 + stat0["mrs0"])
    # footprint
    stat0["footprint"] = footprints.mean()
    F = F.mean(axis=0)
    Fneu = Fneu.mean(axis=0)
    dF = F - parent.ops["neucoeff"]*Fneu
    # activity stats
    stat0["skew"] = stats.skew(dF)
    stat0["std"] = dF.std()
    # compute outline and circle around cell
    iext = fig.boundary(ypix, xpix)
    stat0["yext"] = ypix[iext]
    stat0["xext"] = xpix[iext]
    ycirc, xcirc = fig.circle(stat0["med"], stat0["radius"])
    goodi = (
            (ycirc >= 0)
            & (xcirc >= 0)
            & (ycirc < parent.ops["Ly"])
            & (xcirc < parent.ops["Lx"])
            )
    stat0["ycirc"] = ycirc[goodi]
    stat0["xcirc"] = xcirc[goodi]
    # deconvolve activity
    spks = dcnv.oasis(dF[np.newaxis, :], parent.ops)
    # add cell to structs
    parent.stat = np.concatenate((parent.stat, np.array([stat0])), axis=0)
    print(parent.stat[-1]["ypix"].shape)
    parent.Fcell = np.concatenate((parent.Fcell, F[np.newaxis,:]), axis=0)
    parent.Fneu = np.concatenate((parent.Fneu, Fneu[np.newaxis,:]), axis=0)
    parent.Spks = np.concatenate((parent.Spks, spks), axis=0)
    iscell = np.array([parent.iscell[parent.ichosen]], dtype=bool)
    parent.iscell = np.concatenate((parent.iscell, iscell), axis=0)

def fig_masks(parent):
    """ merges multiple cells' colors together """
    ncells  = parent.stat.size
    nmerge  = int(ncells - 1)
    # cells or notcells
    i0      = int(1-parent.iscell[-1])
    #for n in np.array(parent.imerge):
    #    for k in range(parent.iROI.shape[1]):
    #        ipix = np.array((parent.iROI[i0,k,:,:]==n).nonzero()).astype(np.int32)
    #        parent.iROI[i0, k, ipix[0,:], ipix[1,:]] = nmerge
        #    ipix = np.array((parent.iExt[i0,k,:,:]==n).nonzero()).astype(np.int32)
        #    parent.iExt[i0, k, ipix[0,:], ipix[1,:]] = ncells + nmerged
        #ypix = np.append(ypix, parent.stat[n]["ypix"])
        #xpix = np.append(xpix, parent.stat[n]["xpix"])
    ypix = parent.stat[nmerge]["ypix"].astype(np.int32)
    xpix = parent.stat[nmerge]["xpix"].astype(np.int32)
    parent.iROI[i0, 0, ypix, xpix] = nmerge
    parent.iExt[i0, 0, ypix, xpix] = nmerge

    cols = parent.ops_plot[3]
    cols = np.concatenate((cols, np.ones((1,cols.shape[1]))), axis=0)
    parent.ops_plot[3] = cols
    print(cols[nmerge,:])
    for c in range(cols.shape[1]):
        for k in range(5):
            if k<3 or k==4:
                H = cols[parent.iROI[i0,0,ypix,xpix],c]
                S = parent.Sroi[i0,ypix,xpix]
            else:
                H = cols[parent.iExt[i0,0,ypix,xpix],c]
                S = parent.Sext[i0,ypix,xpix]
            if k==0:
                V = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i0,0,ypix,xpix]/parent.LamMean))
            elif k==1 or k==2 or k==4:
                V = parent.Vback[k-1,ypix,xpix]
                S = np.maximum(0, np.minimum(1, 1.5*0.75*parent.Lam[i0,0,ypix,xpix]/parent.LamMean))
            elif k==3:
                V = parent.Vback[k-1,ypix,xpix]
                V = np.minimum(1, V + S)
            H = np.expand_dims(H.flatten(),axis=1)
            S = np.expand_dims(S.flatten(),axis=1)
            V = np.expand_dims(V.flatten(),axis=1)
            hsv = np.concatenate((H,S,V),axis=1)
            parent.RGBall[i0,c,k,ypix,xpix,:] = hsv_to_rgb(hsv)
