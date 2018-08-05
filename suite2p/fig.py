import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
from matplotlib.colors import hsv_to_rgb

def boundary(ypix,xpix):
    ''' returns pixels of mask that are on the exterior of the mask '''
    ypix = np.expand_dims(ypix.flatten(),axis=1)
    xpix = np.expand_dims(xpix.flatten(),axis=1)
    npix = ypix.shape[0]
    idist = ((ypix - ypix.transpose())**2 + (xpix - xpix.transpose())**2)
    idist[np.arange(0,npix),np.arange(0,npix)] = 500
    nneigh = (idist==1).sum(axis=1) # number of neighbors of each point
    iext = (nneigh<4).flatten()
    return iext

def init_masks(parent):
    '''creates RGB masks using stat and puts them in M0 or M1 depending on
    whether or not iscell is True for a given ROI
    args:
        ops: mean_image, Vcorr
        stat: xpix,ypix,xext,yext
        iscell: vector with True if ROI is cell
        ops_plot: plotROI, view, color, randcols
    outputs:
        M0: ROIs that are True in iscell
        M1: ROIs that are False in iscell
    '''
    ops = parent.ops
    stat = parent.stat
    iscell = parent.iscell
    cols = parent.ops_plot[3]
    ncells = len(stat)
    Ly = ops['Ly']
    Lx = ops['Lx']
    Sroi  = np.zeros((2,Ly,Lx), np.float32)
    Sext   = np.zeros((2,Ly,Lx), np.float32)
    LamAll = np.zeros((Ly,Lx), np.float32)
    Lam    = np.zeros((2,3,Ly,Lx), np.float32)
    iExt   = -1 * np.ones((2,3,Ly,Lx), np.int32)
    iROI   = -1 * np.ones((2,3,Ly,Lx), np.int32)

    for n in range(ncells-1,-1,-1):
        ypix = stat[n]['ypix']
        if ypix is not None:
            xpix = stat[n]['xpix']
            yext = stat[n]['yext']
            xext = stat[n]['xext']
            lam = stat[n]['lam']
            lam = lam / lam.sum()
            i = int(1-iscell[n])
            # add cell on top
            iROI[i,2,ypix,xpix] = iROI[i,1,ypix,xpix]
            iROI[i,1,ypix,xpix] = iROI[i,0,ypix,xpix]
            iROI[i,0,ypix,xpix] = n
            # add outline to all layers
            iExt[i,2,yext,xext] = iExt[i,1,yext,xext]
            iExt[i,1,yext,xext] = iExt[i,0,yext,xext]
            iunder = iExt[i,1,yext,xext]
            iExt[i,0,yext,xext] = n
            #stat[n]['yext_overlap'] = np.append(stat[n]['yext_overlap'], yext[iunder>=0], axis=0)
            #stat[n]['xext_overlap'] = np.append(stat[n]['xext_overlap'], xext[iunder>=0], axis=0)
            #for k in np.unique(iunder[iunder>=0]):
            #    stat[k]['yext_overlap'] = np.append(stat[k]['yext_overlap'], yext[iunder==k], axis=0)
            #    stat[k]['xext_overlap'] = np.append(stat[k]['xext_overlap'], xext[iunder==k], axis=0)
            # add weighting to all layers
            Lam[i,2,ypix,xpix] = Lam[i,1,ypix,xpix]
            Lam[i,1,ypix,xpix] = Lam[i,0,ypix,xpix]
            Lam[i,0,ypix,xpix] = lam
            Sroi[i,ypix,xpix] = 1
            Sext[i,yext,xext] = 1
            LamAll[ypix,xpix] = lam

    LamMean = LamAll[LamAll>1e-10].mean()
    RGBall = np.zeros((2,cols.shape[1],4,Ly,Lx,3), np.float32)
    Vback   = np.zeros((4,Ly,Lx), np.float32)
    RGBback = np.zeros((4,Ly,Lx,3), np.float32)

    for i in range(2):
        for c in range(0,cols.shape[1]):
            for k in range(4):
                H = cols[iROI[i,0,:,:],c]
                if k<3:
                    S = Sroi[i,:,:]
                else:
                    S = Sext[i,:,:]
                V = np.maximum(0, np.minimum(1, 0.75*Lam[i,0,:,:]/LamMean))
                if k>0:
                    if k==1:
                        mimg = ops['meanImg']
                        mimg = mimg - gaussian_filter(filters.minimum_filter(mimg,50),10)
                        mimg = mimg / gaussian_filter(filters.maximum_filter(mimg,50),10)
                        S =     np.maximum(0,np.minimum(1, V*1.5))
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg = (mimg - mimg1) / (mimg99 - mimg1)
                        mimg = np.maximum(0,np.minimum(1,mimg))
                    elif k==2:
                        mimg = ops['meanImg']
                        S = np.maximum(0,np.minimum(1, V*1.5))
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                        mimg = np.maximum(0,np.minimum(1,mimg))
                    else:
                        vcorr = ops['Vcorr']
                        mimg1 = np.percentile(vcorr,1)
                        mimg99 = np.percentile(vcorr,99)
                        vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
                        mimg = mimg1 * np.ones((ops['Ly'],ops['Lx']),np.float32)
                        mimg[ops['yrange'][0]:ops['yrange'][1],
                            ops['xrange'][0]:ops['xrange'][1]] = vcorr
                        mimg = np.maximum(0,np.minimum(1,mimg))
                    Vback[k-1,:,:] = mimg
                    V = mimg
                    if k==3:
                        V = np.maximum(0,np.minimum(1, V + S))
                H = np.expand_dims(H,axis=2)
                S = np.expand_dims(S,axis=2)
                V = np.expand_dims(V,axis=2)
                hsv = np.concatenate((H,S,V),axis=2)
                RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)

    for k in range(3):
        H = np.zeros((Ly,Lx,1),np.float32)
        S = np.zeros((Ly,Lx,1),np.float32)
        V = np.expand_dims(Vback[k,:,:],axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        RGBback[k,:,:,:] = hsv_to_rgb(hsv)

    parent.RGBall = RGBall
    parent.RGBback = RGBback
    parent.Vback = Vback
    parent.iROI = iROI
    parent.iExt = iExt
    parent.Sroi = Sroi
    parent.Sext = Sext
    parent.Lam  = Lam
    parent.LamMean = LamMean


def make_chosen(M, ipix):
    v = (M[ipix[0,:],ipix[1,:],:]).max(axis=1)
    M[ipix[0,:],ipix[1,:],:] = np.resize(np.tile(v, 3), (3,ipix.shape[1])).transpose()
    return M

def draw_masks(parent): #ops, stat, ops_plot, iscell, ichosen):
    '''creates RGB masks using stat and puts them in M0 or M1 depending on
    whether or not iscell is True for a given ROI
    args:
        ops: mean_image, Vcorr
        stat: xpix,ypix
        iscell: vector with True if ROI is cell
        ops_plot: plotROI, view, color, randcols
    outputs:
        M0: ROIs that are True in iscell
        M1: ROIs that are False in iscell
    '''
    ncells  = parent.iscell.shape[0]
    plotROI = parent.ops_plot[0]
    view    = parent.ops_plot[1]
    color   = parent.ops_plot[2]
    if view>0 and plotROI==0:
        M0 = parent.RGBback[view-1,:,:,:]
        M1 = parent.RGBback[view-1,:,:,:]
    else:
        ichosen = parent.ichosen
        wplot   = int(1-parent.iscell[ichosen])
        M0 = np.array(parent.RGBall[0,color,view,:,:,:])
        M1 = np.array(parent.RGBall[1,color,view,:,:,:])
        ipix = np.array((parent.iROI[wplot,0,:,:]==ichosen).nonzero()).astype(np.int32)
        if wplot==0:
            M0 = make_chosen(M0, ipix)
        else:
            M1 = make_chosen(M1, ipix)
    return M0,M1

def flip_cell(parent):
    cols = parent.ops_plot[3]
    n = parent.ichosen
    i = int(1-parent.iscell[n])
    i0 = 1-i
    # ROI stats
    lam  = parent.stat[n]['lam']
    ypix = parent.stat[n]['ypix']
    xpix = parent.stat[n]['xpix']
    yext = parent.stat[n]['yext']
    xext = parent.stat[n]['xext']
    # cell indices
    ipix = np.array((parent.iROI[i0,0,:,:]==n).nonzero()).astype(np.int32)
    ipix1 = np.array((parent.iROI[i0,1,:,:]==n).nonzero()).astype(np.int32)
    ipix2 = np.array((parent.iROI[i0,2,:,:]==n).nonzero()).astype(np.int32)
    # get rid of cell and push up overlaps
    parent.iROI[i0,0,ipix[0,:],ipix[1,:]] = parent.iROI[i0,1,ipix[0,:],ipix[1,:]]
    parent.iROI[i0,0,ipix1[0,:],ipix1[1,:]] = -1
    parent.iROI[i0,1,ipix[0,:],ipix[1,:]] = parent.iROI[i0,2,ipix[0,:],ipix[1,:]]
    parent.iROI[i0,1,ipix2[0,:],ipix2[1,:]] = -1
    parent.iROI[i0,2,ipix[0,:],ipix[1,:]] = -1
    parent.Lam[i0,0,ipix[0,:],ipix[1,:]]  = parent.Lam[i0,1,ipix[0,:],ipix[1,:]]
    parent.Lam[i0,0,ipix1[0,:],ipix1[1,:]] = 0
    parent.Lam[i0,1,ipix[0,:],ipix[1,:]]  = parent.Lam[i0,2,ipix[0,:],ipix[1,:]]
    parent.Lam[i0,1,ipix2[0,:],ipix2[1,:]] = 0
    parent.Lam[i0,2,ipix[0,:],ipix[1,:]]  = 0
    ipix = np.array((parent.iExt[i0,0,:,:]==n).nonzero()).astype(np.int32)
    ipix1 = np.array((parent.iExt[i0,1,:,:]==n).nonzero()).astype(np.int32)
    ipix2 = np.array((parent.iExt[i0,2,:,:]==n).nonzero()).astype(np.int32)
    parent.iExt[i0,0,ipix[0,:],ipix[1,:]] = parent.iExt[i0,1,ipix[0,:],ipix[1,:]]
    goodi = parent.iExt[i0,0,yext,xext]<0
    parent.iExt[i0,0,ipix1[0,:],ipix1[1,:]] = -1
    parent.iExt[i0,1,ipix[0,:],ipix[1,:]] = parent.iExt[i0,2,ipix[0,:],ipix[1,:]]
    parent.iExt[i0,1,ipix2[0,:],ipix2[1,:]] = -1
    parent.iExt[i0,2,ipix[0,:],ipix[1,:]] = -1
    # add cell to other side (on top) and push down overlaps
    parent.iROI[i,2,ypix,xpix] = parent.iROI[i,1,ypix,xpix]
    parent.iROI[i,1,ypix,xpix] = parent.iROI[i,0,ypix,xpix]
    parent.iROI[i,0,ypix,xpix] = n
    parent.iExt[i,2,yext,xext] = parent.iExt[i,1,yext,xext]
    parent.iExt[i,1,yext,xext] = parent.iExt[i,0,yext,xext]
    parent.iExt[i,0,yext,xext] = n
    parent.Lam[i,2,ypix,xpix]  = parent.Lam[i,1,ypix,xpix]
    parent.Lam[i,1,ypix,xpix]  = parent.Lam[i,0,ypix,xpix]
    parent.Lam[i,0,ypix,xpix]  = lam
    yonly = ypix[~parent.stat[n]['overlap']]
    xonly = xpix[~parent.stat[n]['overlap']]
    parent.Sroi[i,ypix,xpix] = 1
    parent.Sroi[i0,yonly,xonly] = 0
    parent.Sext[i,yext,xext] = 1
    yonly = yext[goodi]
    xonly = xext[goodi]
    parent.Sext[i0,yonly,xonly] = 0

    for i in range(2):
        for c in range(cols.shape[1]):
            for k in range(4):
                if k<3:
                    H = cols[parent.iROI[i,0,ypix,xpix],c]
                    S = parent.Sroi[i,ypix,xpix]
                else:
                    H = cols[parent.iExt[i,0,ypix,xpix],c]
                    S = parent.Sext[i,ypix,xpix]
                if k==0:
                    V = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,ypix,xpix]/parent.LamMean))
                elif k==1 or k==2:
                    V = parent.Vback[k-1,ypix,xpix]
                    S = np.maximum(0, np.minimum(1, 1.5*0.75*parent.Lam[i,0,ypix,xpix]/parent.LamMean))
                elif k==3:
                    V = parent.Vback[k-1,ypix,xpix]
                    V = np.minimum(1, V + S)
                H = np.expand_dims(H.flatten(),axis=1)
                S = np.expand_dims(S.flatten(),axis=1)
                V = np.expand_dims(V.flatten(),axis=1)
                hsv = np.concatenate((H,S,V),axis=1)
                if k<3:
                    parent.RGBall[i,c,k,ypix,xpix,:] = hsv_to_rgb(hsv)
                else:
                    parent.RGBall[i,c,k,ypix,xpix,:] = hsv_to_rgb(hsv)


def ROI_index(ops, stat):
    '''matrix Ly x Lx where each pixel is an ROI index (-1 if no ROI present)'''
    ncells = len(stat)-1
    Ly = ops['Ly']
    Lx = ops['Lx']
    iROI = -1 * np.ones((Ly,Lx), dtype=np.int32)
    for n in range(ncells):
        ypix = stat[n]['ypix'][~stat[n]['overlap']]
        if ypix is not None:
            xpix = stat[n]['xpix'][~stat[n]['overlap']]
            iROI[ypix,xpix] = n
    return iROI

def make_colorbar():
    H = np.arange(0,100).astype(np.float32)
    H = H / (100*1.3)
    H = H + 0.1
    H = 1 - H
    H = np.expand_dims(H,axis=1)
    S = np.ones((100,1))
    V = np.ones((100,1))
    hsv = np.concatenate((H,S,V), axis=1)
    colormat = hsv_to_rgb(hsv)
    colormat = np.expand_dims(colormat, axis=0)
    colormat = np.tile(colormat,(20,1,1))
    return colormat
