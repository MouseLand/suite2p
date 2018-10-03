import numpy as np
import pyqtgraph as pg
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy import ndimage
from suite2p import utils
import math
from PyQt5 import QtGui
from matplotlib.colors import hsv_to_rgb


def plot_colorbar(parent, bid):
    if bid==0:
        parent.colorbar.setImage(np.zeros((20,100,3)))
    else:
        parent.colorbar.setImage(parent.colormat)
    for k in range(3):
        parent.clabel[k].setText('%1.2f'%parent.clabels[bid][k])

def plot_trace(parent):
    parent.p3.clear()
    ax = parent.p3.getAxis('left')
    if len(parent.imerge)==1:
        n = parent.imerge[0]
        f = parent.Fcell[n,:]
        fneu = parent.Fneu[n,:]
        sp = parent.Spks[n,:]
        fmax = np.maximum(f.max(), fneu.max())
        fmin = np.minimum(f.min(), fneu.min())
        #sp from 0 to fmax
        sp /= sp.max()
        sp *= fmax# - fmin
        #sp += fmin
        parent.p3.plot(parent.trange,f,pen='b')
        parent.p3.plot(parent.trange,fneu,pen='r')
        parent.p3.plot(parent.trange,sp,pen=(255,255,255,100))
        parent.fmin=0
        parent.fmax=fmax
        ax.setTicks(None)
        for n in range(3):
            parent.traceLabel[n].setText(parent.traceText[n])
    else:
        nmax = int(parent.ncedit.text())
        kspace = 1.0/parent.sc
        ttick = list()
        pmerge = parent.imerge[:np.minimum(len(parent.imerge),nmax)]
        k=len(pmerge)-1
        i = parent.activityMode
        favg = np.zeros((parent.Fcell.shape[1],))
        for n in pmerge[::-1]:
            if i==0:
                f = parent.Fcell[n,:]
            elif i==1:
                f = parent.Fneu[n,:]
            elif i==2:
                f = parent.Fcell[n,:] - 0.7*parent.Fneu[n,:]
            else:
                f = parent.Spks[n,:]
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            f = (f - fmin) / (fmax - fmin)
            rgb = hsv_to_rgb([parent.ops_plot[3][n,0],1,1])*255
            parent.p3.plot(parent.trange,f+k*kspace,pen=rgb)
            ttick.append((k*kspace+f.mean(), str(n)))
            k-=1
        bsc = len(pmerge)/25 + 1
        # at bottom plot behavior and avg trace
        if parent.bloaded:
            favg -= favg.min()
            favg /= favg.max()
            parent.p3.plot(parent.trange,-1*bsc+parent.beh*bsc,pen='w')
            parent.p3.plot(parent.trange,-1*bsc+favg*bsc,pen=(140,140,140))
            parent.traceLabel[0].setText("<font color='gray'>mean activity</font>")
            parent.traceLabel[1].setText("<font color='white'>1D variable</font>")
            parent.traceLabel[2].setText("")
            parent.fmin=-1*bsc
        else:
            for n in range(3):
                parent.traceLabel[n].setText("")
            parent.fmin=0
        #ttick.append((-0.5*bsc,'1D var'))

        parent.fmax=(len(pmerge)-1)*kspace + 1
        ax.setTicks([ttick])
    parent.p3.setXRange(0,parent.Fcell.shape[1])
    parent.p3.setYRange(parent.fmin,parent.fmax)

def plot_masks(parent,M):
    parent.img1.setImage(M[0],levels=(0.0,1.0))
    parent.img2.setImage(M[1],levels=(0.0,1.0))
    parent.img1.show()
    parent.img2.show()

def init_range(parent):
    parent.p1.setXRange(0,parent.ops['Lx'])
    parent.p1.setYRange(0,parent.ops['Ly'])
    parent.p2.setXRange(0,parent.ops['Lx'])
    parent.p2.setYRange(0,parent.ops['Ly'])
    parent.p3.setLimits(xMin=0,xMax=parent.Fcell.shape[1])
    parent.trange = np.arange(0, parent.Fcell.shape[1])

def make_colors(parent):
    parent.clabels = []
    ncells = len(parent.stat)
    allcols = np.random.random((ncells,1))
    b=0
    for names in parent.colors[:-1]:
        if b > 0:
            istat = np.zeros((ncells,1))
            if b<len(parent.colors)-2:
                for n in range(0,ncells):
                    istat[n] = parent.stat[n][names]
            else:
                istat = np.expand_dims(parent.probcell, axis=1)
            istat1 = np.percentile(istat,2)
            istat99 = np.percentile(istat,98)
            parent.clabels.append([istat1,
                                 (istat99-istat1)/2 + istat1,
                                 istat99])
            istat = istat - istat1
            istat = istat / (istat99-istat1)
            istat = np.maximum(0, np.minimum(1, istat))
            istat = istat / 1.3
            istat = istat + 0.1
            icols = 1 - istat
            allcols = np.concatenate((allcols, icols), axis=1)
        else:
            parent.clabels.append([0,0.5,1])
        b+=1
    parent.clabels.append([0,0.5,1])
    parent.ops_plot[3] = allcols
    #parent.ops_plot[4] = corrcols
    #parent.cc = cc

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

def circle(med, r):
    ''' returns pixels of circle with radius 1.25x radius of cell (r)'''
    theta = np.linspace(0.0,2*np.pi,100)
    x = r*1.35 * np.cos(theta) + med[0]
    y = r*1.35 * np.sin(theta) + med[1]
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    return x,y

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
    RGBall = np.zeros((2,cols.shape[1]+1,5,Ly,Lx,3), np.float32)
    Vback   = np.zeros((4,Ly,Lx), np.float32)
    RGBback = np.zeros((4,Ly,Lx,3), np.float32)

    for k in range(5):
        if k>0:
            if k==2:
                if 'meanImgE' not in ops:
                    ops = utils.enhanced_mean_image(ops)
                mimg = ops['meanImgE']
            elif k==1:
                mimg = ops['meanImg']
                S = np.maximum(0,np.minimum(1, Vorig*1.5))
                mimg1 = np.percentile(mimg,1)
                mimg99 = np.percentile(mimg,99)
                mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                mimg = np.maximum(0,np.minimum(1,mimg))
            elif k==3:
                vcorr = ops['Vcorr']
                mimg1 = np.percentile(vcorr,1)
                mimg99 = np.percentile(vcorr,99)
                vcorr = (vcorr - mimg1) / (mimg99 - mimg1)
                mimg = mimg1 * np.ones((ops['Ly'],ops['Lx']),np.float32)
                mimg[ops['yrange'][0]:ops['yrange'][1],
                    ops['xrange'][0]:ops['xrange'][1]] = vcorr
                mimg = np.maximum(0,np.minimum(1,mimg))
            else:
                if ops['nchannels']>1:
                    mimg = ops['meanImg_chan2']
                    mimg1 = np.percentile(mimg,1)
                    mimg99 = np.percentile(mimg,99)
                    mimg     = (mimg - mimg1) / (mimg99 - mimg1)
                    mimg = np.maximum(0,np.minimum(1,mimg))
                else:
                    mimg = np.zeros((ops['Ly'],ops['Lx']),np.float32)

            Vback[k-1,:,:] = mimg
            V = mimg
            V = np.expand_dims(V,axis=2)
        for i in range(2):
            Vorig = np.maximum(0, np.minimum(1, 0.75*Lam[i,0,:,:]/LamMean))
            Vorig = np.expand_dims(Vorig,axis=2)
            if k==3:
                S = np.expand_dims(Sext[i,:,:],axis=2)
                Va = np.maximum(0,np.minimum(1, V + S))
            else:
                S = np.expand_dims(Sroi[i,:,:],axis=2)
                if k>0:
                    S     = np.maximum(0,np.minimum(1, Vorig*1.5))
                    Va    = V
                else:
                    Va = Vorig
            for c in range(0,cols.shape[1]):
                H = cols[iROI[i,0,:,:],c]
                H = np.expand_dims(H,axis=2)
                hsv = np.concatenate((H,S,Va),axis=2)
                RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)

    for k in range(4):
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

def rastermap_masks(parent):
    k = parent.ops_plot[1]
    c = parent.ops_plot[3].shape[1]+2
    n = np.array(parent.imerge)
    inactive=False
    no_1d = False
    istat = parent.isort
    # no 1D variable loaded -- leave blank
    if len(parent.clabels)==len(parent.colors):
        parent.clabels.append([])
        no_1d = True
    if len(parent.clabels)==len(parent.colors)+1:
        parent.clabels.append([0, istat.max()/2, istat.max()])
        inactive=True
    else:
        parent.clabels[-1] = [0, istat.max()/2, istat.max()]

    istat = istat / istat.max()
    istat = istat / 1.3
    istat = istat + 0.1
    cols = 1 - istat
    cols[parent.isort==-1] = 0
    parent.ops_plot[6] = cols
    if inactive:
        nb,Ly,Lx = parent.Vback.shape[0]+1, parent.Vback.shape[1], parent.Vback.shape[2]
        rgb = np.zeros((2,1,nb,Ly,Lx,3),np.float32)
    for i in range(2):
        H = cols[parent.iROI[i,0,:,:]]
        Vorig = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,:,:]/parent.LamMean))
        if k==0:
            S = parent.Sroi[i,:,:]
            V = Vorig
        elif k==3:
            S = parent.Sext[i,:,:]
            V = parent.Vback[k-1,:,:]
            V = np.maximum(0,np.minimum(1, V + S))
        else:
            S = np.maximum(0,np.minimum(1, Vorig*1.5))
            V = parent.Vback[k-1,:,:]
        H = np.expand_dims(H,axis=2)
        S = np.expand_dims(S,axis=2)
        V = np.expand_dims(V,axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        if inactive:
            rgb[i,0,k,:,:,:] = hsv_to_rgb(hsv)
        else:
            parent.RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)
    if inactive:
        if no_1d:
            parent.RGBall = np.concatenate([parent.RGBall,rgb], axis=1)
        parent.RGBall = np.concatenate([parent.RGBall,rgb], axis=1)


def beh_masks(parent):
    k = parent.ops_plot[1]
    c = parent.ops_plot[3].shape[1]+1
    print(c)
    n = np.array(parent.imerge)
    nb = int(np.floor(parent.beh.size/parent.bin))
    sn = np.reshape(parent.beh[:nb*parent.bin],(nb,parent.bin)).mean(axis=1)
    sn -= sn.mean()
    snstd = (sn**2).sum()
    cc = np.dot(parent.Fbin, sn.T) / np.sqrt(np.dot(parent.Fstd,snstd))
    cc[n] = cc.mean()
    istat = cc
    inactive=False
    istat_min = istat.min()
    istat_max = istat.max()
    istat = istat - istat.min()
    istat = istat / istat.max()
    istat = istat / 1.3
    istat = istat + 0.1
    cols = 1 - istat
    parent.ops_plot[5] = cols
    if len(parent.clabels)==len(parent.colors):
        parent.clabels.append([istat_min,
                              (istat_max-istat_min)/2 + istat_min,
                              istat_max])
        inactive=True
    else:
        parent.clabels[-1] = [istat_min,
                              (istat_max-istat_min)/2 + istat_min,
                              istat_max]
    if inactive:
        nb,Ly,Lx = parent.Vback.shape[0]+1, parent.Vback.shape[1], parent.Vback.shape[2]
        rgb = np.zeros((2,1,nb,Ly,Lx,3),np.float32)
    for i in range(2):
        H = cols[parent.iROI[i,0,:,:]]
        Vorig = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,:,:]/parent.LamMean))
        if k==0:
            S = parent.Sroi[i,:,:]
            V = Vorig
        elif k==3:
            S = parent.Sext[i,:,:]
            V = parent.Vback[k-1,:,:]
            V = np.maximum(0,np.minimum(1, V + S))
        else:
            S = np.maximum(0,np.minimum(1, Vorig*1.5))
            V = parent.Vback[k-1,:,:]
        H = np.expand_dims(H,axis=2)
        S = np.expand_dims(S,axis=2)
        V = np.expand_dims(V,axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        if inactive:
            rgb[i,0,k,:,:,:] = hsv_to_rgb(hsv)
        else:
            parent.RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)
    if inactive:
        parent.RGBall = np.concatenate([parent.RGBall,rgb], axis=1)

def corr_masks(parent):
    k = parent.ops_plot[1]
    c = parent.ops_plot[3].shape[1]
    n = np.array(parent.imerge)
    sn = parent.Fbin[n,:].mean(axis=0)
    sn -= sn.mean()
    snstd = (sn**2).sum()
    cc = np.dot(parent.Fbin, sn.T) / np.sqrt(np.dot(parent.Fstd,snstd))
    cc[n] = cc.mean()
    istat = cc
    parent.clabels[len(parent.colors)-1] = [istat.min(),
                         (istat.max()-istat.min())/2 + istat.min(),
                         istat.max()]
    istat = istat - istat.min()
    istat = istat / istat.max()
    istat = istat / 1.3
    istat = istat + 0.1
    cols = 1 - istat
    parent.ops_plot[4] = cols
    for i in range(2):
        H = cols[parent.iROI[i,0,:,:]]
        Vorig = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,:,:]/parent.LamMean))
        if k==0:
            S = parent.Sroi[i,:,:]
            V = Vorig
        elif k==3:
            S = parent.Sext[i,:,:]
            V = parent.Vback[k-1,:,:]
            V = np.maximum(0,np.minimum(1, V + S))
        else:
            S = np.maximum(0,np.minimum(1, Vorig*1.5))
            V = parent.Vback[k-1,:,:]
        H = np.expand_dims(H,axis=2)
        S = np.expand_dims(S,axis=2)
        V = np.expand_dims(V,axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        parent.RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)

def draw_corr(parent):
    k = parent.ops_plot[1]
    c = parent.ops_plot[3].shape[1]
    cols = parent.ops_plot[4]
    for i in range(2):
        H = cols[parent.iROI[i,0,:,:]]
        Vorig = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,:,:]/parent.LamMean))
        if k==0:
            S = parent.Sroi[i,:,:]
            V = Vorig
        elif k==3:
            S = parent.Sext[i,:,:]
            V = parent.Vback[k-1,:,:]
            V = np.maximum(0,np.minimum(1, V + S))
        else:
            S = np.maximum(0,np.minimum(1, Vorig*1.5))
            V = parent.Vback[k-1,:,:]
        H = np.expand_dims(H,axis=2)
        S = np.expand_dims(S,axis=2)
        V = np.expand_dims(V,axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        parent.RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)

def class_masks(parent):
    cols = parent.ops_plot[3]
    c = cols.shape[1] - 1
    k = parent.ops_plot[1]
    for i in range(2):
        H = cols[parent.iROI[i,0,:,:],c]
        if k<3:
            S = parent.Sroi[i,:,:]
        else:
            S = parent.Sext[i,:,:]
        V = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,:,:]/parent.LamMean))
        if k>0:
            V = parent.Vback[k-1,:,:]
            if k==3:
                V = np.maximum(0,np.minimum(1, V + S))
        H = np.expand_dims(H,axis=2)
        S = np.expand_dims(S,axis=2)
        V = np.expand_dims(V,axis=2)
        hsv = np.concatenate((H,S,V),axis=2)
        parent.RGBall[i,c,k,:,:,:] = hsv_to_rgb(hsv)

def flip_for_class(parent, iscell):
    ncells = iscell.size
    if (iscell==parent.iscell).sum() < 100:
        for n in range(ncells):
            if iscell[n] != parent.iscell[n]:
                parent.iscell[n] = iscell[n]
                parent.ichosen = n
                flip_cell(parent)
    else:
        parent.iscell = iscell
        init_masks(parent)

def make_chosen_ROI(M0, ypix, xpix, lam):
    v = lam
    M0[ypix,xpix,:] = np.resize(np.tile(v, 3), (3,ypix.size)).transpose()
    return M0

def make_chosen_circle(M0, ycirc, xcirc, col, sat):
    ncirc = ycirc.size
    pix = np.concatenate((col*np.ones((ncirc,1),np.float32),
                          sat*np.ones((ncirc,1), np.float32),
                          np.ones((ncirc,1), np.float32)),axis=1)
    M0[ycirc,xcirc,:] = hsv_to_rgb(pix)
    return M0

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
    cols    = parent.ops_plot[3]
    if view>0 and plotROI==0:
        M = [parent.RGBback[view-1,:,:,:],
             parent.RGBback[view-1,:,:,:]]
    else:
        wplot   = int(1-parent.iscell[parent.ichosen])
        M = [np.array(parent.RGBall[0,color,view,:,:,:]), np.array(parent.RGBall[1,color,view,:,:,:])]
        ypixA = np.zeros((0,),np.int32)
        xpixA = np.zeros((0,),np.int32)
        vbackA = np.zeros((0,3),np.float32)
        if view==0:
            ischosen = np.isin(parent.iROI[wplot,0,:,:], parent.imerge)
            M[wplot][ischosen,:] = 255
            #lam = parent.stat[n]['lam']
            #lam /= lam.sum()
            #lam = np.maximum(0, np.minimum(1, 0.75 * lam / parent.LamMean))
            #M[wplot] = make_chosen_ROI(M[wplot], ypix, xpix, lam)
        else:
            for n in parent.imerge:
                ypix = parent.stat[n]['ypix'].flatten()
                xpix = parent.stat[n]['xpix'].flatten()
                ypixA = np.concatenate((ypixA,ypix),axis=0)
                xpixA = np.concatenate((xpixA,xpix),axis=0)
                vbackA = np.concatenate((vbackA, parent.RGBback[view-1,ypix,xpix,:]),axis=0)
            M[wplot][ypixA,xpixA,:] = vbackA
            for n in parent.imerge:
                ycirc = parent.stat[n]['ycirc']
                xcirc = parent.stat[n]['xcirc']
                if color==cols.shape[1]:
                    col = parent.ops_plot[4][n]
                    sat = 0
                    M[wplot] = make_chosen_circle(M[wplot], ycirc, xcirc, col, sat)
                else:
                    col   = cols[n,color]
                    sat = 1
                    M[wplot] = make_chosen_circle(M[wplot], ycirc, xcirc, col, sat)
    return M[0],M[1]

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
    parent.Lam[i,0,ypix,xpix]  = lam / lam.sum()
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
            for k in range(5):
                if k<3 or k==4:
                    H = cols[parent.iROI[i,0,ypix,xpix],c]
                    S = parent.Sroi[i,ypix,xpix]
                else:
                    H = cols[parent.iExt[i,0,ypix,xpix],c]
                    S = parent.Sext[i,ypix,xpix]
                if k==0:
                    V = np.maximum(0, np.minimum(1, 0.75*parent.Lam[i,0,ypix,xpix]/parent.LamMean))
                elif k==1 or k==2 or k==4:
                    V = parent.Vback[k-1,ypix,xpix]
                    S = np.maximum(0, np.minimum(1, 1.5*0.75*parent.Lam[i,0,ypix,xpix]/parent.LamMean))
                elif k==3:
                    V = parent.Vback[k-1,ypix,xpix]
                    V = np.minimum(1, V + S)
                H = np.expand_dims(H.flatten(),axis=1)
                S = np.expand_dims(S.flatten(),axis=1)
                V = np.expand_dims(V.flatten(),axis=1)
                hsv = np.concatenate((H,S,V),axis=1)
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
