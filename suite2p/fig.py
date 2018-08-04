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
    H      = np.zeros((cols.shape[1],Ly,Lx), np.float32)
    Sroi  = np.zeros((2,Ly,Lx), np.float32)
    Sext   = np.zeros((2,Ly,Lx), np.float32)
    LamAll = np.zeros((Ly,Lx), np.float32)
    Lam    = np.zeros((2,Ly,Lx), np.float32)
    iROI   = -1 * np.ones((Ly,Lx), np.int32)
    iExt   = -1 * np.ones((Ly,Lx), np.int32)

    for n in range(ncells):
        ypix = stat[n]['ypix']
        if ypix is not None:
            xpix = stat[n]['xpix']
            yext = stat[n]['yext']
            xext = stat[n]['xext']
            lam = stat[n]['lam']
            lam = lam / lam.sum()
            i = int(1-iscell[n])
            iROI[ypix,xpix] = n
            iExt[yext,xext] = n
            Sroi[i,ypix,xpix] = 1
            Sext[i,yext,xext] = 1
            Lam[i,ypix,xpix]  = lam
            LamAll[ypix,xpix] = lam

    # create H from cols and iROI
    for c in range(cols.shape[1]):
        H[c,iROI>=0] = cols[iROI[iROI>=0],c]

    LamMean = LamAll[LamAll>1e-10].mean()
    parent.H = H
    parent.iROI = iROI
    parent.iExt = iExt
    parent.Sroi = Sroi
    parent.Sext = Sext
    parent.Lam  = Lam
    parent.LamMean = LamMean

    # create all mask options
    parent.RGB_all = np.zeros((2,cols.shape[1],4,Ly,Lx,3), np.float32)
    parent.Vback = np.zeros((3,Ly,Lx), np.float32)
    parent.RGBback = np.zeros((3,Ly,Lx,3), np.float32)

    for i in range(2):
        for c in range(cols.shape[1]):
            for k in range(4):
                H = parent.H[c,:,:]
                if k<3:
                    S = parent.Sroi[i,:,:]
                else:
                    S = parent.Sext[i,:,:]
                V = np.maximum(0, np.minimum(1, 0.75*Lam[i,:,:]/parent.LamMean))
                if k>0:
                    if k==1:
                        mimg = ops['meanImg']
                        mimg = mimg - gaussian_filter(filters.minimum_filter(mimg,50),10)
                        mimg = mimg / gaussian_filter(filters.maximum_filter(mimg,50),10)
                        S = np.minimum(1, V*1.5)
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg = (mimg - mimg1) / (mimg99 - mimg1)
                    elif k==2:
                        mimg = ops['meanImg']
                        S = np.minimum(1, V*1.5)
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg = (mimg - mimg1) / (mimg99 - mimg1)
                    else:
                        vcorr = ops['Vcorr']
                        mimg = np.zeros((ops['Ly'],ops['Lx']),np.float32)
                        mimg[ops['yrange'][0]:ops['yrange'][1],
                            ops['xrange'][0]:ops['xrange'][1]] = vcorr
                        mimg1 = np.percentile(mimg,1)
                        mimg99 = np.percentile(mimg,99)
                        mimg = (mimg - mimg1) / (mimg99 - mimg1)

                    parent.Vback[k-1,:,:] = mimg
                    V = mimg
                    if k==3:
                        V = np.minimum(1, V + S)
                H = np.expand_dims(H,axis=2)
                S = np.expand_dims(S,axis=2)
                V = np.expand_dims(V,axis=2)
                hsv = np.concatenate((H,S,V),axis=2)
                parent.RGB_all[i,c,k,:,:,:] = hsv_to_rgb(hsv)

    for k in range(3):
        H = np.zeros((Ly,Lx,1),np.float32)
        S = np.zeros((Ly,Lx,1),np.float32)
        V = np.expand_dims(parent.Vback[k,:,:],axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        parent.RGBback[k,:,:,:] = hsv_to_rgb(hsv)

def make_chosen(M, ypix, xpix):
    v = M[ypix,xpix,:].max(axis=1)
    M[ypix,xpix,:] = np.resize(np.tile(v, 3), (3,len(ypix))).transpose()
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
        M0 = np.array(parent.RGB_all[0,color,view,:,:,:])
        M1 = np.array(parent.RGB_all[1,color,view,:,:,:])

        ypix    = parent.stat[ichosen]['ypix'].flatten()
        xpix    = parent.stat[ichosen]['xpix'].flatten()
        if wplot==0:
            M0 = make_chosen(M0, ypix, xpix)
        else:
            M1 = make_chosen(M1, ypix, xpix)
    return M0,M1

def flip_cell(parent):
    n = parent.ichosen
    i = int(1-parent.iscell[n])
    i0 = 1-i
    # cell indic
    nin = parent.iROI==n
    next = parent.iExt==n
    lam0 = np.array(parent.Lam[i0,nin])
    parent.Lam[i,nin] = lam0
    parent.Lam[i0,nin] = 0
    parent.Sroi[i,nin] = 1
    parent.Sroi[i0,nin] = 0
    parent.Sext[i,next] = 1
    parent.Sext[i0,next] = 0

    for i in range(2):
        for c in range(parent.H.shape[0]):
            for k in range(4):
                if k<3:
                    H = parent.H[c,nin]
                    S = parent.Sroi[i,nin]
                else:
                    H = parent.H[c,next]
                    S = parent.Sext[i,next]
                if k==0:
                    V = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,nin]/parent.LamMean))
                elif k==1 or k==2:
                    V = parent.Vback[k-1,nin]
                    S = np.maximum(0, np.minimum(1, 1.5*0.75*parent.Lam[i,nin]/parent.LamMean))
                elif k==3:
                    V = parent.Vback[k-1,next]
                    V = np.minimum(1, V + S)
                H = np.expand_dims(H,axis=1)
                S = np.expand_dims(S,axis=1)
                V = np.expand_dims(V,axis=1)
                hsv = np.concatenate((H,S,V),axis=1)
                if k<3:
                    parent.RGB_all[i,c,k,nin,:] = hsv_to_rgb(hsv)
                else:
                    parent.RGB_all[i,c,k,next,:] = hsv_to_rgb(hsv)


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
