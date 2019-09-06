import numpy as np
import pyqtgraph as pg
from scipy import stats
from suite2p import utils, dcnv, sparsedetect, fig
import math
from PyQt5 import QtGui
from matplotlib.colors import hsv_to_rgb
import time

def distance_matrix(parent, ilist):
    idist = 1e6 * np.ones((len(ilist), len(ilist)))
    for ij,j in enumerate(ilist):
        for ik,k in enumerate(ilist):
            if ij<ik:
                idist[ij,ik] = (((parent.stat[j]['ypix'][np.newaxis,:] -
                                  parent.stat[k]['ypix'][:,np.newaxis])**2 +
                                 (parent.stat[j]['xpix'][np.newaxis,:] -
                                  parent.stat[k]['xpix'][:,np.newaxis])**2) ** 0.5).min()
    return idist

def merge_ROIs(parent):
    merge_activity_masks(parent)
    n = parent.iscell.size-1
    redraw_masks(parent, n)
    parent.ichosen = n
    parent.imerge = [n]
    M = fig.draw_masks(parent)
    fig.plot_masks(parent, M)
    fig.plot_trace(parent)
    parent.win.show()
    parent.show()

def merge_activity_masks(parent):
    print('merging activity... this may take some time')
    i0      = int(1-parent.iscell[parent.ichosen])
    ypix = np.array([])
    xpix = np.array([])
    lam  = np.array([])
    footprints  = np.array([])
    F    = np.zeros((0,parent.Fcell.shape[1]), np.float32)
    Fneu = np.zeros((0,parent.Fcell.shape[1]), np.float32)
    probcell = []
    probredcell = []
    merged_cells = []
    remove_merged = []
    for n in np.array(parent.imerge):
        if len(parent.stat[n]['imerge']) > 0:
            remove_merged.append(n)
            for k in parent.stat[n]['imerge']:
                merged_cells.append(k)
        else:
            merged_cells.append(n)
    merged_cells = np.unique(np.array(merged_cells))

    for n in merged_cells:
        ypix = np.append(ypix, parent.stat[n]["ypix"])
        xpix = np.append(xpix, parent.stat[n]["xpix"])
        lam = np.append(lam, parent.stat[n]["lam"])
        footprints = np.append(footprints, parent.stat[n]["footprint"])
        F    = np.append(F, parent.Fcell[n,:][np.newaxis,:], axis=0)
        Fneu = np.append(Fneu, parent.Fneu[n,:][np.newaxis,:], axis=0)
        probcell.append(parent.probcell[n])
        probredcell.append(parent.probredcell[n])

    probcell = np.array(probcell)
    probredcell = np.array(probredcell)
    pmean = probcell.mean()
    prmean = probredcell.mean()

    # remove overlaps
    ipix = np.concatenate((ypix[:,np.newaxis], xpix[:,np.newaxis]), axis=1)
    _, goodi = np.unique(ipix, return_index=True, axis=0)
    ypix = ypix[goodi]
    xpix = xpix[goodi]
    lam = lam[goodi]

    stat0 = {}
    if 'aspect' in parent.ops:
        d0 = np.array([int(parent.ops['aspect']*10), 10])
    else:
        d0 = parent.ops['diameter']
        if isinstance(d0, int):
            d0 = [d0,d0]

    ### compute statistics of merges
    stat0["imerge"] = merged_cells
    stat0["ypix"] = ypix.astype(np.int32)
    stat0["xpix"] = xpix.astype(np.int32)
    stat0["lam"] = lam / lam.sum() * merged_cells.size
    stat0['med']  = [np.median(stat0["ypix"]), np.median(stat0["xpix"])]
    stat0["npix"] = ypix.size
    radius = utils.fitMVGaus(ypix/d0[0], xpix/d0[1], lam, 2)[2]
    stat0["radius"] = radius[0] * d0.mean()
    stat0["aspect_ratio"] = 2 * radius[0]/(.01 + radius[0] + radius[1])
    npix = np.array([parent.stat[n]['npix'] for n in range(len(parent.stat))]).astype('float32')
    stat0["npix_norm"] = stat0["npix"] / npix.mean()
    # compactness
    rs,dy,dx = sparsedetect.circleMask(d0)
    rsort = np.sort(rs.flatten())
    r2 = ((ypix - stat0["med"][0])/d0[0])**2 + ((xpix - stat0["med"][1])/d0[1])**2
    r2 = r2**.5
    stat0["mrs"]  = np.mean(r2)
    stat0["mrs0"] = np.mean(rsort[:r2.size])
    stat0["compact"] = stat0["mrs"] / (1e-10 + stat0["mrs0"])
    # footprint
    stat0["footprint"] = footprints.mean()
    # inmerge
    stat0["inmerge"] = 0

    ### compute activity of merged cells
    F = F.mean(axis=0)
    Fneu = Fneu.mean(axis=0)
    dF = F - parent.ops["neucoeff"]*Fneu
    # activity stats
    stat0["skew"] = stats.skew(dF)
    stat0["std"] = dF.std()

    ### for GUI drawing
    # compute outline and circle around cell
    iext = fig.boundary(ypix, xpix)
    stat0["yext"] = ypix[iext].astype(np.int32)
    stat0["xext"] = xpix[iext].astype(np.int32)
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

    ### remove previously merged cell (do not replace)
    for k in remove_merged:
        remove_mask(parent, k)
        np.delete(parent.stat, k, 0)
        np.delete(parent.Fcell, k, 0)
        np.delete(parent.Fneu, k, 0)
        np.delete(parent.Spks, k, 0)
        np.delete(parent.iscell, k, 0)
        np.delete(parent.probcell, k, 0)
        np.delete(parent.probredcell, k, 0)
        np.delete(parent.redcell, k, 0)
        np.delete(parent.notmerged, k, 0)

    # add cell to structs
    parent.stat = np.concatenate((parent.stat, np.array([stat0])), axis=0)
    parent.stat = sparsedetect.get_overlaps(parent.stat, parent.ops)
    parent.stat = np.array(parent.stat)
    parent.Fcell = np.concatenate((parent.Fcell, F[np.newaxis,:]), axis=0)
    parent.Fneu = np.concatenate((parent.Fneu, Fneu[np.newaxis,:]), axis=0)
    parent.Spks = np.concatenate((parent.Spks, spks), axis=0)
    iscell = np.array([parent.iscell[parent.ichosen]], dtype=bool)
    parent.iscell = np.concatenate((parent.iscell, iscell), axis=0)
    parent.probcell = np.append(parent.probcell, pmean)
    parent.probredcell = np.append(parent.probredcell, prmean)
    parent.redcell = np.append(parent.redcell, prmean > parent.chan2prob)
    parent.notmerged = np.append(parent.notmerged, False)

    # recompute binned F
    parent.mode_change(parent.activityMode)

    for n in merged_cells:
        parent.stat[n]['inmerge'] = parent.stat.size-1

    add_mask(parent, parent.iscell.size-1)

def remove_mask(parent, n):
    i0 = int(1-parent.iscell[n])
    for k in range(parent.iROI.shape[1]):
        ipix = np.array((parent.iROI[i0,k,:,:]==n).nonzero()).astype(np.int32)
        parent.iROI[i0, k, ipix[0,:], ipix[1,:]] = -1
        ipix = np.array((parent.iExt[i0,k,:,:]==n).nonzero()).astype(np.int32)
        parent.iExt[i0, k, ipix[0,:], ipix[1,:]] = -1
    np.delete(parent.ops_plot[3], n, 0)

def add_mask(parent, n):
    ypix = parent.stat[n]["ypix"].astype(np.int32)
    xpix = parent.stat[n]["xpix"].astype(np.int32)
    i0      = int(1-parent.iscell[n])
    parent.iROI[i0, 0, ypix, xpix] = n
    parent.iExt[i0, 0, ypix, xpix] = n

    cols = parent.ops_plot[3]
    cols = np.concatenate((cols, cols[parent.stat[n]['imerge'][0]]*np.ones((1,cols.shape[1]))), axis=0)
    parent.ops_plot[3] = cols

def redraw_masks(parent, n):
    '''
    redraws masks with new cell ids
    '''
    i0 = int(1-parent.iscell[n])
    ypix = parent.stat[n]["ypix"].astype(np.int32).flatten()
    xpix = parent.stat[n]["xpix"].astype(np.int32).flatten()
    cols = parent.ops_plot[3]
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

class MergeWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(MergeWindow, self).__init__(parent)
        self.setGeometry(700,300,700,700)
        self.setWindowTitle('Choose merge options')
        self.cwidget = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout()
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        self.cwidget.setLayout(self.layout)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win, 11, 0, 4, 4)
        self.p0 = self.win.addPlot(row=0, col=0)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.enableAutoRange(x=True,y=True)
        # initial ops values
        mkeys = ['corr_thres', 'dist_thres']
        mlabels = ['correlation threshold', 'euclidean distance threshold']
        self.ops = {'corr_thres': 0.8, 'dist_thres': 100.0}
        self.layout.addWidget(QtGui.QLabel('Press enter in a text box to update params'), 0, 0, 1,2)
        self.layout.addWidget(QtGui.QLabel('(Correlations use "activity mode" and "bin" from main GUI)'), 1, 0, 1,2)
        self.layout.addWidget(QtGui.QLabel('>>>>>>>>>>>> Parameters <<<<<<<<<<<'), 2, 0, 1,2)
        self.doMerge = QtGui.QPushButton('merge selected ROIs', default=False, autoDefault=False)
        self.doMerge.clicked.connect(lambda: self.do_merge(parent))
        self.doMerge.setEnabled(False)
        self.layout.addWidget(self.doMerge, 9,0,1,1)

        self.suggestMerge = QtGui.QPushButton('next merge suggestion', default=False, autoDefault=False)
        self.suggestMerge.clicked.connect(lambda: self.suggest_merge(parent))
        self.suggestMerge.setEnabled(False)
        self.layout.addWidget(self.suggestMerge, 10,0,1,1)

        self.nMerge = QtGui.QLabel('= X possible merges found with these parameters')
        self.layout.addWidget(self.nMerge, 7,0,1,2)

        self.iMerge = QtGui.QLabel('suggested ROIs to merge: ')
        self.layout.addWidget(self.iMerge, 8,0,1,2)

        self.editlist = []
        self.keylist = []
        k=1
        for lkey,llabel in zip(mkeys, mlabels):
            qlabel = QtGui.QLabel(llabel)
            qlabel.setFont(QtGui.QFont("Times",weight=QtGui.QFont.Bold))
            self.layout.addWidget(qlabel, k*2+1,0,1,2)
            qedit = LineEdit(lkey,self)
            qedit.set_text(self.ops)
            qedit.setFixedWidth(90)
            qedit.returnPressed.connect(lambda: self.compute_merge_list(parent))
            self.layout.addWidget(qedit, k*2+2,0,1,2)
            self.editlist.append(qedit)
            self.keylist.append(lkey)
            k+=1

        print('creating merge window... this may take some time')
        self.CC  = np.matmul(parent.Fbin[parent.iscell], parent.Fbin[parent.iscell].T) / parent.Fbin.shape[-1]
        self.CC /= np.matmul(parent.Fstd[parent.iscell][:,np.newaxis],
                             parent.Fstd[parent.iscell][np.newaxis,:]) + 1e-3
        self.CC -= np.diag(np.diag(self.CC))

        self.compute_merge_list(parent)

    def do_merge(self, parent):
        merge_ROIs(parent)
        for ilist in self.merge_list:
            for n in range(ilist.size):
                if parent.stat[ilist[n]]['inmerge'] > 0:
                    ilist[n] = parent.stat[ilist[n]]['inmerge']
            ilist = np.unique(ilist)
        self.unmerged[self.n-1] = False

        self.cc_row  = np.matmul(parent.Fbin[parent.iscell], parent.Fbin[-1].T) / parent.Fbin.shape[-1]
        self.cc_row /= parent.Fstd[parent.iscell] * parent.Fstd[-1] + 1e-3
        self.cc_row[-1] = 0
        self.CC = np.concatenate((self.CC, self.cc_row[np.newaxis, :-1]), axis=0)
        self.CC = np.concatenate((self.CC, self.cc_row[:,np.newaxis]), axis=1)

        parent.ichosen = parent.stat.size-1
        parent.imerge = [parent.ichosen]
        self.iMerge.setText('ROIs merged: %s'%parent.stat[parent.ichosen]['imerge'])
        self.doMerge.setEnabled(False)
        parent.ichosen_stats()
        M = fig.draw_masks(parent)
        fig.plot_masks(parent, M)
        fig.plot_trace(parent)
        parent.win.show()
        parent.show()
        #self.suggest_merge(parent)

    def compute_merge_list(self, parent):
        print('computing automated merge suggestions...')
        for k,key in enumerate(self.keylist):
            self.ops[key] = self.editlist[k].get_text()
        goodind = []
        NN = len(parent.stat[parent.iscell])
        notused = np.ones(NN, np.bool) # not in a suggested merge
        icell = np.where(parent.iscell)[0]
        for k in range(NN):
            if notused[k]:
                ilist = [i for i, x in enumerate(self.CC[k]) if x >= self.ops['corr_thres']]
                ilist.append(k)
                if len(ilist) > 1:
                    for n,i in enumerate(ilist):
                        if notused[i]:
                            ilist[n] = icell[i]
                            if parent.stat[ilist[n]]['inmerge'] > 0:
                                ilist[n] = parent.stat[ilist[n]]['inmerge']
                    ilist = np.unique(np.array(ilist))
                    if ilist.size > 1:
                        idist = distance_matrix(parent,ilist)
                        idist = idist.min(axis=1)
                        ilist = ilist[idist <= self.ops['dist_thres']]
                        if ilist.size > 1:
                            for i in ilist:
                                notused[parent.iscell[:i].sum()] = False
                            goodind.append(ilist)
        self.nMerge.setText('= %d possible merges found with these parameters'%len(goodind))
        self.merge_list = goodind
        self.n = 0
        if len(self.merge_list) > 0:
            self.suggestMerge.setEnabled(True)
            self.unmerged = np.ones(len(self.merge_list), np.bool)
            self.suggest_merge(parent)

    def suggest_merge(self, parent):
        parent.ichosen = self.merge_list[self.n][0]
        parent.imerge  = list(self.merge_list[self.n])
        if self.unmerged[self.n]:
            self.iMerge.setText('suggested ROIs to merge: %s'%parent.imerge)
            self.doMerge.setEnabled(True)
            self.p0.clear()
            cell0 = parent.imerge[0]
            sstring = ''
            for i in parent.imerge[1:]:
                rgb = hsv_to_rgb([parent.ops_plot[3][i,0],1,1])*255
                pen = pg.mkPen(rgb, width=3)
                scatter=pg.ScatterPlotItem(parent.Fbin[cell0], parent.Fbin[i], pen=pen)
                self.p0.addItem(scatter)
                sstring += ' %d '%i
            self.p0.setLabel('left', sstring)
            self.p0.setLabel('bottom', str(cell0))
        else:
            # set to the merged ROI index
            parent.ichosen = parent.stat[parent.ichosen]['inmerge']
            parent.imerge = [parent.ichosen]
            self.iMerge.setText('ROIs merged: %s'%list(parent.stat[parent.ichosen]['imerge']))
            self.doMerge.setEnabled(False)
            self.p0.clear()

        self.n+=1
        if self.n > len(self.merge_list)-1:
            self.n = 0
        parent.ichosen_stats()
        M = fig.draw_masks(parent)
        fig.plot_masks(parent, M)
        fig.plot_trace(parent)
        parent.zoom_to_cell()
        parent.win.show()
        parent.show()

class LineEdit(QtGui.QLineEdit):
    def __init__(self,key,parent=None):
        super(LineEdit,self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self):
        key = self.key
        okey = float(self.text())
        return okey

    def set_text(self,ops):
        key = self.key
        dstr = str(ops[key])
        self.setText(dstr)
