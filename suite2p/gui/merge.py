"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import numpy as np
import pyqtgraph as pg
from qtpy import QtGui
from qtpy.QtWidgets import QDialog, QLineEdit, QGridLayout, QMessageBox, QLabel, QPushButton, QWidget
from scipy import stats

from . import masks, io
from . import utils
from ..detection.stats import roi_stats, median_pix
from ..extraction.dcnv import oasis


def distance_matrix(parent, ilist):
    idist = 1e6 * np.ones((len(ilist), len(ilist)))
    for ij, j in enumerate(ilist):
        for ik, k in enumerate(ilist):
            if ij < ik:
                idist[ij,
                      ik] = (((parent.stat[j]["ypix"][np.newaxis, :] -
                               parent.stat[k]["ypix"][:, np.newaxis])**2 +
                              (parent.stat[j]["xpix"][np.newaxis, :] -
                               parent.stat[k]["xpix"][:, np.newaxis])**2)**0.5).mean()
    return idist


def do_merge(parent):
    dm = QMessageBox.question(
        parent,
        "Merge cells",
        "Do you want to merge selected cells?",
        QMessageBox.Yes | QMessageBox.No,
    )
    if dm == QMessageBox.Yes:
        merge_activity_masks(parent)
        parent.merged.append(parent.imerge)
        parent.update_plot()
        print(parent.merged)
        print("merged ROIs")


def merge_activity_masks(parent):
    print("merging activity... this may take some time")
    i0 = int(1 - parent.iscell[parent.ichosen])
    ypix = np.zeros((0,), np.int32)
    xpix = np.zeros((0,), np.int32)
    lam = np.zeros((0,), np.float32)
    footprints = np.array([])
    F = np.zeros((0, parent.Fcell.shape[1]), np.float32)
    Fneu = np.zeros((0, parent.Fcell.shape[1]), np.float32)
    if parent.hasred:
        F_chan2 = np.zeros((0, parent.Fcell.shape[1]), np.float32)
        Fneu_chan2 = np.zeros((0, parent.Fcell.shape[1]), np.float32)
        if not hasattr(parent, "F_chan2"):
            parent.F_chan2 = np.load(os.path.join(parent.basename, "F_chan2.npy"))
            parent.Fneu_chan2 = np.load(os.path.join(parent.basename, "Fneu_chan2.npy"))

    probcell = []
    probredcell = []
    merged_cells = []
    remove_merged = []
    for n in np.array(parent.imerge):
        if len(parent.stat[n]["imerge"]) > 0:
            remove_merged.append(n)
            for k in parent.stat[n]["imerge"]:
                merged_cells.append(k)
        else:
            merged_cells.append(n)
    merged_cells = np.unique(np.array(merged_cells))

    for n in merged_cells:
        ypix = np.append(ypix, parent.stat[n]["ypix"])
        xpix = np.append(xpix, parent.stat[n]["xpix"])
        lam = np.append(lam, parent.stat[n]["lam"])
        footprints = np.append(footprints, parent.stat[n]["footprint"])
        F = np.append(F, parent.Fcell[n, :][np.newaxis, :], axis=0)
        Fneu = np.append(Fneu, parent.Fneu[n, :][np.newaxis, :], axis=0)
        if parent.hasred:
            F_chan2 = np.append(F_chan2, parent.F_chan2[n, :][np.newaxis, :], axis=0)
            Fneu_chan2 = np.append(Fneu_chan2, parent.Fneu_chan2[n, :][np.newaxis, :],
                                   axis=0)
        probcell.append(parent.probcell[n])
        probredcell.append(parent.probredcell[n])

    probcell = np.array(probcell)
    probredcell = np.array(probredcell)
    pmean = probcell.mean()
    prmean = probredcell.mean()

    # remove overlaps
    ipix = np.concatenate((ypix[:, np.newaxis], xpix[:, np.newaxis]), axis=1)
    _, goodi = np.unique(ipix, return_index=True, axis=0)
    ypix = ypix[goodi]
    xpix = xpix[goodi]
    lam = lam[goodi]

    ### compute statistics of merges
    stat0 = {}
    stat0["imerge"] = merged_cells
    if "iplane" in parent.stat[merged_cells[0]]:
        stat0["iplane"] = parent.stat[merged_cells[0]]["iplane"]
    stat0["ypix"] = ypix
    stat0["xpix"] = xpix
    stat0["med"] = median_pix(ypix, xpix)
    stat0["lam"] = lam / lam.sum()

    if "aspect" in parent.ops:
        d0 = np.array([int(parent.ops["aspect"] * 10), 10])
    else:
        d0 = parent.ops["diameter"]
        if isinstance(d0, int):
            d0 = [d0, d0]

    # red prob
    stat0["chan2_prob"] = -1
    # inmerge
    stat0["inmerge"] = -1

    ### compute activity of merged cells
    F = F.mean(axis=0)
    Fneu = Fneu.mean(axis=0)
    if parent.hasred:
        F_chan2 = F_chan2.mean(axis=0)
        Fneu_chan2 = Fneu_chan2.mean(axis=0)
    dF = F - parent.ops["neucoeff"] * Fneu
    # activity stats
    stat0["skew"] = stats.skew(dF)
    stat0["std"] = dF.std()

    spks = oasis(F=dF[np.newaxis, :], batch_size=parent.ops["batch_size"],
                 tau=parent.ops["tau"], fs=parent.ops["fs"])

    ### remove previously merged cell from FOV (do not replace)
    for k in remove_merged:
        masks.remove_roi(parent, k, i0)
        np.delete(parent.stat, k, 0)
        np.delete(parent.Fcell, k, 0)
        np.delete(parent.Fneu, k, 0)
        np.delete(parent.F_chan2, k, 0)
        np.delete(parent.Fneu_chan2, k, 0)
        np.delete(parent.Spks, k, 0)
        np.delete(parent.iscell, k, 0)
        np.delete(parent.probcell, k, 0)
        np.delete(parent.probredcell, k, 0)
        np.delete(parent.redcell, k, 0)
        np.delete(parent.notmerged, k, 0)

    # add cell to structs
    parent.stat = np.concatenate((parent.stat, np.array([stat0])), axis=0)
    parent.stat = roi_stats(parent.stat, parent.Ly, parent.Lx,
                            aspect=parent.ops.get("aspect", None),
                            diameter=parent.ops.get("diameter", None),
                            do_crop=parent.ops.get("soma_crop", 1))
    parent.stat[-1]["lam"] = parent.stat[-1]["lam"] * merged_cells.size
    parent.Fcell = np.concatenate((parent.Fcell, F[np.newaxis, :]), axis=0)
    parent.Fneu = np.concatenate((parent.Fneu, Fneu[np.newaxis, :]), axis=0)
    if parent.hasred:
        parent.F_chan2 = np.concatenate((parent.F_chan2, F_chan2[np.newaxis, :]),
                                        axis=0)
        parent.Fneu_chan2 = np.concatenate(
            (parent.Fneu_chan2, Fneu_chan2[np.newaxis, :]), axis=0)
    parent.Spks = np.concatenate((parent.Spks, spks), axis=0)
    iscell = np.array([parent.iscell[parent.ichosen]], dtype=bool)
    parent.iscell = np.concatenate((parent.iscell, iscell), axis=0)
    parent.probcell = np.append(parent.probcell, pmean)
    parent.probredcell = np.append(parent.probredcell, -1)
    parent.redcell = np.append(parent.redcell, False)
    parent.notmerged = np.append(parent.notmerged, False)

    ### for GUI drawing
    ycirc, xcirc = utils.circle(parent.stat[-1]["med"], parent.stat[-1]["radius"])
    goodi = ((ycirc >= 0) & (xcirc >= 0) & (ycirc < parent.ops["Ly"]) &
             (xcirc < parent.ops["Lx"]))
    parent.stat[-1]["ycirc"] = ycirc[goodi]
    parent.stat[-1]["xcirc"] = xcirc[goodi]

    # * add colors *
    masks.make_colors(parent)
    # recompute binned F
    parent.mode_change(parent.activityMode)

    for n in merged_cells:
        parent.stat[n]["inmerge"] = len(parent.stat) - 1
        masks.remove_roi(parent, n, i0)
    masks.add_roi(parent, len(parent.stat) - 1, i0)
    masks.redraw_masks(parent, ypix, xpix)


class MergeWindow(QDialog):

    def __init__(self, parent=None):
        super(MergeWindow, self).__init__(parent)
        self.setGeometry(700, 300, 700, 700)
        self.setWindowTitle("Choose merge options")
        self.cwidget = QWidget(self)
        self.layout = QGridLayout()
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        self.cwidget.setLayout(self.layout)
        self.win = pg.GraphicsLayoutWidget()
        self.layout.addWidget(self.win, 11, 0, 4, 4)
        self.p0 = self.win.addPlot(row=0, col=0)
        self.p0.setMouseEnabled(x=False, y=False)
        self.p0.enableAutoRange(x=True, y=True)
        # initial settings values
        mkeys = ["corr_thres", "dist_thres"]
        mlabels = ["correlation threshold", "euclidean distance threshold"]
        self.settings = {"corr_thres": 0.8, "dist_thres": 100.0}
        self.layout.addWidget(QLabel("Press enter in a text box to update params"), 0,
                              0, 1, 2)
        self.layout.addWidget(
            QLabel("(Correlations use 'activity mode' and 'bin' from main GUI)"), 1, 0,
            1, 2)
        self.layout.addWidget(QLabel(">>>>>>>>>>>> Parameters <<<<<<<<<<<"), 2, 0, 1, 2)
        self.doMerge = QPushButton("merge selected ROIs", default=False,
                                   autoDefault=False)
        self.doMerge.clicked.connect(lambda: self.do_merge(parent))
        self.doMerge.setEnabled(False)
        self.layout.addWidget(self.doMerge, 9, 0, 1, 1)

        self.suggestMerge = QPushButton("next merge suggestion", default=False,
                                        autoDefault=False)
        self.suggestMerge.clicked.connect(lambda: self.suggest_merge(parent))
        self.suggestMerge.setEnabled(False)
        self.layout.addWidget(self.suggestMerge, 10, 0, 1, 1)

        self.nMerge = QLabel("= X possible merges found with these parameters")
        self.layout.addWidget(self.nMerge, 7, 0, 1, 2)

        self.iMerge = QLabel("suggested ROIs to merge: ")
        self.layout.addWidget(self.iMerge, 8, 0, 1, 2)

        self.editlist = []
        self.keylist = []
        k = 1
        for lkey, llabel in zip(mkeys, mlabels):
            qlabel = QLabel(llabel)
            qlabel.setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))
            self.layout.addWidget(qlabel, k * 2 + 1, 0, 1, 2)
            qedit = LineEdit(lkey, self)
            qedit.set_text(self.settings)
            qedit.setFixedWidth(90)
            qedit.returnPressed.connect(lambda: self.compute_merge_list(parent))
            self.layout.addWidget(qedit, k * 2 + 2, 0, 1, 2)
            self.editlist.append(qedit)
            self.keylist.append(lkey)
            k += 1

        print("creating merge window... this may take some time")
        self.CC = np.matmul(parent.Fbin[parent.iscell],
                            parent.Fbin[parent.iscell].T) / parent.Fbin.shape[-1]
        self.CC /= np.matmul(parent.Fstd[parent.iscell][:, np.newaxis],
                             parent.Fstd[parent.iscell][np.newaxis, :]) + 1e-3
        self.CC -= np.diag(np.diag(self.CC))

        self.compute_merge_list(parent)

    def do_merge(self, parent):
        merge_activity_masks(parent)
        parent.merged.append(parent.imerge)
        parent.update_plot()

        self.cc_row = np.matmul(parent.Fbin[parent.iscell],
                                parent.Fbin[-1].T) / parent.Fbin.shape[-1]
        self.cc_row /= parent.Fstd[parent.iscell] * parent.Fstd[-1] + 1e-3
        self.cc_row[-1] = 0
        self.CC = np.concatenate((self.CC, self.cc_row[np.newaxis, :-1]), axis=0)
        self.CC = np.concatenate((self.CC, self.cc_row[:, np.newaxis]), axis=1)
        for n in parent.imerge:
            self.CC[parent.imerge] = 0
            self.CC[:, parent.imerge] = 0

        parent.ichosen = parent.stat.size - 1
        parent.imerge = [parent.ichosen]
        print("ROIs merged: %s" % parent.stat[parent.ichosen]["imerge"])
        self.compute_merge_list(parent)

    def compute_merge_list(self, parent):
        print("computing automated merge suggestions...")
        for k, key in enumerate(self.keylist):
            self.settings[key] = self.editlist[k].get_text()
        goodind = []
        NN = len(parent.stat[parent.iscell])
        notused = np.ones(NN, "bool")  # not in a suggested merge
        icell = np.where(parent.iscell)[0]
        for k in range(NN):
            if notused[k]:
                ilist = [
                    i for i, x in enumerate(self.CC[k]) if x >= self.settings["corr_thres"]
                ]
                ilist.append(k)
                if len(ilist) > 1:
                    for n, i in enumerate(ilist):
                        if notused[i]:
                            ilist[n] = icell[i]
                            if parent.stat[ilist[n]]["inmerge"] > 0:
                                ilist[n] = parent.stat[ilist[n]]["inmerge"]
                    ilist = np.unique(np.array(ilist))
                    if ilist.size > 1:
                        idist = distance_matrix(parent, ilist)
                        idist = idist.min(axis=1)
                        ilist = ilist[idist <= self.settings["dist_thres"]]
                        if ilist.size > 1:
                            for i in ilist:
                                notused[parent.iscell[:i].sum()] = False
                            goodind.append(ilist)
        self.set_merge_list(parent, goodind)

    def set_merge_list(self, parent, goodind):
        self.nMerge.setText("= %d possible merges found with these parameters" %
                            len(goodind))
        self.merge_list = goodind
        self.n = 0
        if len(self.merge_list) > 0:
            self.suggestMerge.setEnabled(True)
            self.unmerged = np.ones(len(self.merge_list), bool)
            self.suggest_merge(parent)

    def suggest_merge(self, parent):
        parent.ichosen = self.merge_list[self.n][0]
        parent.imerge = list(self.merge_list[self.n])
        if self.unmerged[self.n]:
            self.iMerge.setText("suggested ROIs to merge: %s" % parent.imerge)
            self.doMerge.setEnabled(True)
            self.p0.clear()
            cell0 = parent.imerge[0]
            sstring = ""
            for i in parent.imerge[1:]:
                rgb = parent.colors["cols"][0, i]
                pen = pg.mkPen(rgb, width=3)
                scatter = pg.ScatterPlotItem(parent.Fbin[cell0], parent.Fbin[i],
                                             pen=pen)
                self.p0.addItem(scatter)
                sstring += " %d " % i
            self.p0.setLabel("left", sstring)
            self.p0.setLabel("bottom", str(cell0))
        else:
            # set to the merged ROI index
            parent.ichosen = parent.stat[parent.ichosen]["inmerge"]
            parent.imerge = [parent.ichosen]
            self.iMerge.setText("ROIs merged: %s" %
                                list(parent.stat[parent.ichosen]["imerge"]))
            self.doMerge.setEnabled(False)
            self.p0.clear()

        self.n += 1
        if self.n > len(self.merge_list) - 1:
            self.n = 0
        parent.checkBoxz.setChecked(True)
        parent.update_plot()
        parent.win.show()
        parent.show()


class LineEdit(QLineEdit):

    def __init__(self, key, parent=None):
        super(LineEdit, self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self):
        key = self.key
        okey = float(self.text())
        return okey

    def set_text(self, settings):
        key = self.key
        dstr = str(settings[key])
        self.setText(dstr)


def apply(parent):
    classval = float(parent.probedit.text())
    iscell = parent.probcell > classval
    masks.flip_for_class(parent, iscell)
    parent.update_plot()
    io.save_iscell(parent)
