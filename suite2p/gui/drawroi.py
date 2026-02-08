"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time
import math

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QLabel, QLineEdit, QMainWindow, QGridLayout, QButtonGroup, QMessageBox, QWidget
from matplotlib.colors import hsv_to_rgb
from scipy import stats
from scipy.ndimage import rotate

from . import io
from ..extraction import masks
from ..detection.stats import roi_stats
from ..extraction import preprocess
from ..extraction.dcnv import oasis


def masks_and_traces(settings, stat_manual, stat_orig):
    """ main extraction function
        inputs: settings and stat
        creates cell and neuropil masks and extracts traces
        returns: F (ROIs x time), Fneu (ROIs x time), F_chan2, Fneu_chan2, settings, stat
        F_chan2 and Fneu_chan2 will be empty if no second channel
    """

    t0 = time.time()
    
    # Concatenate stat so a good neuropil function can be formed
    stat_all = stat_manual.copy()
    for n in range(len(stat_orig)):
        stat_all.append(stat_orig[n])

    stat_all = roi_stats(stat_all, settings["Ly"], settings["Lx"], aspect=settings.get("aspect", None),
                         diameter=settings["diameter"])
    cell_masks = [
        masks.create_cell_mask(stat, Ly=settings["Ly"], Lx=settings["Lx"],
                               allow_overlap=settings["allow_overlap"]) for stat in stat_all
    ]
    cell_pix = masks.create_cell_pix(stat_all, Ly=settings["Ly"], Lx=settings["Lx"])
    manual_roi_stats = stat_all[:len(stat_manual)]
    manual_cell_masks = cell_masks[:len(stat_manual)]
    manual_neuropil_masks = masks.create_neuropil_masks(
        ypixs=[stat["ypix"] for stat in manual_roi_stats],
        xpixs=[stat["xpix"] for stat in manual_roi_stats],
        cell_pix=cell_pix,
        inner_neuropil_radius=settings["inner_neuropil_radius"],
        min_neuropil_pixels=settings["min_neuropil_pixels"],
    )
    print("Masks made in %0.2f sec." % (time.time() - t0))

    F, Fneu, F_chan2, Fneu_chan2 = extract_traces_from_masks(settings, manual_cell_masks,
                                                             manual_neuropil_masks)

    # compute activity statistics for classifier
    npix = np.array([stat_orig[n]["npix"] for n in range(len(stat_orig))
                    ]).astype("float32")
    for n in range(len(manual_roi_stats)):
        manual_roi_stats[n]["npix_norm"] = manual_roi_stats[n]["npix"] / np.mean(
            npix[:100])  # What if there are less than 100 cells?
        manual_roi_stats[n]["compact"] = 1
        manual_roi_stats[n]["footprint"] = 2
        manual_roi_stats[n]["manual"] = 1  # Add manual key
        if "iplane" in stat_orig[0]:
            manual_roi_stats[n]["iplane"] = stat_orig[0]["iplane"]

    # subtract neuropil and compute skew, std from F
    dF = F - settings["neucoeff"] * Fneu
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)

    for n in range(F.shape[0]):
        manual_roi_stats[n]["skew"] = sk[n]
        manual_roi_stats[n]["std"] = sd[n]
        manual_roi_stats[n]["med"] = [
            np.mean(manual_roi_stats[n]["ypix"]),
            np.mean(manual_roi_stats[n]["xpix"])
        ]

    dF = preprocess(F=dF, baseline=settings["baseline"], win_baseline=settings["win_baseline"],
                    sig_baseline=settings["sig_baseline"], fs=settings["fs"],
                    prctile_baseline=settings["prctile_baseline"])
    spks = oasis(F=dF, batch_size=settings["batch_size"], tau=settings["tau"], fs=settings["fs"])

    return F, Fneu, F_chan2, Fneu_chan2, spks, settings, manual_roi_stats


class ViewButton(QPushButton):

    def __init__(self, bid, Text, parent=None):
        super(ViewButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        parent.img0.setImage(parent.masked_images[:, :, :, bid])

        parent.win.show()
        parent.show()


class ROIDraw(QMainWindow):

    def __init__(self, parent):
        super(ROIDraw, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.parent = parent
        self.setGeometry(70, 70, 1400, 800)
        self.setWindowTitle("extract ROI activity")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        # layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")

        # self.p0 = pg.ViewBox(lockAspect=False,name="plot1",border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 3, 0, 13, 14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot(row=0, col=1)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.setMenuEnabled(False)
        self.p1.scene().sigMouseMoved.connect(self.mouse_moved)

        self.p0 = self.win.addViewBox(name="plot1", lockAspect=True, row=0, col=0,
                                      invertY=True)
        self.img0 = pg.ImageItem()
        self.p0.addItem(self.img0)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.instructions = QLabel("Add ROI: button / Alt+CLICK")
        self.instructions.setStyleSheet("color: white;")
        self.l0.addWidget(self.instructions, 0, 0, 1, 4)
        self.instructions = QLabel("Remove last clicked ROI: D")
        self.instructions.setStyleSheet("color: white;")
        self.l0.addWidget(self.instructions, 1, 0, 1, 4)

        self.addROI = QPushButton("add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(lambda: self.add_ROI(pos=None))
        self.addROI.setEnabled(True)
        self.addROI.setFixedWidth(60)
        self.addROI.setStyleSheet(self.styleUnpressed)
        self.l0.addWidget(self.addROI, 2, 0, 1, 1)
        lbl = QLabel("diameter:")
        lbl.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        lbl.setStyleSheet("color: white;")
        lbl.setFixedWidth(60)
        self.l0.addWidget(lbl, 2, 1, 1, 1)
        self.diam = QLineEdit(self)
        self.diam.setValidator(QtGui.QIntValidator(0, 10000))
        self.diam.setText("12")
        self.diam.setFixedWidth(35)
        self.diam.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.diam, 2, 2, 1, 1)
        self.ROIs = []
        self.cell_pos = []
        self.extracted = False
        self.procROI = QPushButton("extract ROIs")
        self.procROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.procROI.setStyleSheet(self.styleUnpressed)
        self.procROI.setCheckable(False)
        self.procROI.clicked.connect(self.proc_ROI)
        self.l0.addWidget(self.procROI, 3, 0, 1, 3)
        self.l0.addWidget(QLabel(""), 4, 0, 1, 3)
        self.l0.setRowStretch(4, 1)

        # CloseGUI button - once done save the ROIs to stat.npy
        self.saveGUI = False
        self.closeGUI = QPushButton("Save and Quit")
        self.closeGUI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.closeGUI.clicked.connect(self.close_GUI)
        self.closeGUI.setEnabled(False)
        self.closeGUI.setFixedWidth(100)
        self.closeGUI.setStyleSheet(self.styleUnpressed)
        self.l0.addWidget(self.closeGUI, 0, 5, 1, 1)

        # view buttons
        self.views = [
            "W: mean img", "E: mean img (enhanced)", "R: correlation map",
            "T: max projection"
        ]
        b = 0
        self.viewbtns = QButtonGroup(self)
        for names in self.views:
            btn = ViewButton(b, "&" + names, self)
            self.viewbtns.addButton(btn, b)
            self.l0.addWidget(btn, b, 4, 1, 1)
            btn.setEnabled(True)
            b += 1
        b = 0
        self.viewbtns.button(b).setChecked(True)
        self.viewbtns.button(b).setStyleSheet(self.stylePressed)

        self.l0.addWidget(QLabel("neuropil"), 13, 13, 1, 1)

        self.Ly = self.parent.ops["Ly"]
        self.Lx = self.parent.ops["Lx"]
        self.iscell = self.parent.iscell

        # Get maskf for pixels that are cells
        self.masked_images = self.normalize_img_add_masks()
        # Mean image as default
        self.img0.setImage(self.masked_images[:, :, :, 0])

    def closeEvent(self, event):
        print("closing GUI")
        # if user didn"t click "save & quit" button
        if not self.saveGUI:
            self.check_proc(event)

    def check_proc(self, event):
        cproc = QMessageBox.question(
            self, "PROC",
            "Would you like to save traces before closing? (if you havent extracted the traces, click Cancel and extract!)",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if cproc == QMessageBox.Yes:
            self.close_GUI()
        elif cproc == QMessageBox.Cancel:
            event.ignore()

    def close_GUI(self):
        # Replace old stat file
        print("Saving old stat")
        np.save(os.path.join(self.parent.basename, "stat_orig.npy"), self.parent.stat)

        # Save iscell
        print("Num cells", self.nROIs)

        # Append new stat file with old and save
        print("Saving new stat")
        stat_all = self.new_stat.copy()
        for n in range(len(self.parent.stat)):
            stat_all.append(self.parent.stat[n])
        np.save(os.path.join(self.parent.basename, "stat.npy"), stat_all)
        iscell_prob = np.concatenate(
            (self.parent.iscell[:, np.newaxis], self.parent.probcell[:, np.newaxis]),
            axis=1)

        new_iscell = np.ones((self.nROIs, 2))
        new_iscell = np.concatenate((new_iscell, iscell_prob), axis=0)
        np.save(os.path.join(self.parent.basename, "iscell.npy"), new_iscell)

        # Save fluorescence traces
        Fcell = np.concatenate((self.Fcell, self.parent.Fcell), axis=0)
        Fneu = np.concatenate((self.Fneu, self.parent.Fneu), axis=0)
        Spks = np.concatenate(
            (self.Spks, self.parent.Spks), axis=0
        )  # For now convert spikes to 0 for the new ROIS and then fix it later
        np.save(os.path.join(self.parent.basename, "F.npy"), Fcell)
        np.save(os.path.join(self.parent.basename, "Fneu.npy"), Fneu)
        np.save(os.path.join(self.parent.basename, "spks.npy"), Spks)

        if "reg_file_chan2" in self.parent.ops:
            F_chan2 = np.load(os.path.join(self.parent.basename, "F_chan2.npy"))
            Fneu_chan2 = np.load(os.path.join(self.parent.basename, "Fneu_chan2.npy"))
            redorig = np.load(os.path.join(self.parent.basename, "redcell.npy"))
            F_chan2 = np.concatenate((self.F_chan2, F_chan2), axis=0)
            Fneu_chan2 = np.concatenate((self.Fneu_chan2, Fneu_chan2), axis=0)
            Fneu = np.concatenate((self.Fneu, self.parent.Fneu), axis=0)
            new_redcell = np.zeros((self.nROIs, 2))
            new_redcell = np.concatenate((new_redcell, redorig), axis=0)
            np.save(os.path.join(self.parent.basename, "F_chan2.npy"), F_chan2)
            np.save(os.path.join(self.parent.basename, "Fneu_chan2.npy"), Fneu_chan2)
            np.save(os.path.join(self.parent.basename, "redcell.npy"), new_redcell)

        print(np.shape(Fcell), np.shape(Fneu), np.shape(Spks), np.shape(new_iscell),
              np.shape(stat_all))

        # close GUI
        io.load_proc(self.parent)
        self.saveGUI = True
        self.close()

    def normalize_img_add_masks(self):
        masked_image = np.zeros(
            ((self.Ly, self.Lx, 3, 4)))  # 3 for RGB and 4 for buttons
        for i in np.arange(4):  # 4 because 4 buttons
            if i == 0:
                mimg = np.zeros((self.Ly, self.Lx), np.float32)
                mimg[self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                     self.parent.ops["xrange"][0]:self.parent.
                     settings["xrange"][1]] = self.parent.ops["meanImg"][
                         self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                         self.parent.ops["xrange"][0]:self.parent.ops["xrange"][1]]

            elif i == 1:
                mimg = np.zeros((self.Ly, self.Lx), np.float32)
                mimg[self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                     self.parent.ops["xrange"][0]:self.parent.
                     settings["xrange"][1]] = self.parent.ops["meanImgE"][
                         self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                         self.parent.ops["xrange"][0]:self.parent.ops["xrange"][1]]
            elif i == 2:
                mimg = np.zeros((self.Ly, self.Lx), np.float32)
                mimg[self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                     self.parent.ops["xrange"][0]:self.parent.
                     settings["xrange"][1]] = self.parent.ops["Vcorr"]

            else:
                mimg = np.zeros((self.Ly, self.Lx), np.float32)
                if "max_proj" in self.parent.ops:
                    mimg[self.parent.ops["yrange"][0]:self.parent.ops["yrange"][1],
                         self.parent.ops["xrange"][0]:self.parent.
                         settings["xrange"][1]] = self.parent.ops["max_proj"]

            mimg1 = np.percentile(mimg, 1)
            mimg99 = np.percentile(mimg, 99)
            mimg = (mimg - mimg1) / (mimg99 - mimg1)
            mimg = np.maximum(0, np.minimum(1, mimg))
            masked_image[:, :, :, i] = self.create_masks_of_cells(mimg)

        return masked_image

    def create_masks_of_cells(self, mean_img):
        H = np.zeros_like(mean_img)
        S = np.zeros_like(mean_img)
        columncol = self.parent.colors["istat"][0]

        for n in np.arange(np.shape(self.parent.iscell)[0]):
            if self.parent.iscell[n] == 1:
                ypix = self.parent.stat[n]["ypix"].flatten()
                xpix = self.parent.stat[n]["xpix"].flatten()
                H[ypix, xpix] = np.random.rand()
                S[ypix, xpix] = 1

        pix = np.concatenate(
            ((H[:, :, np.newaxis]), S[:, :, np.newaxis], mean_img[:, :, np.newaxis]),
            axis=-1)
        pix = hsv_to_rgb(pix)
        return pix

    def mouse_moved(self, pos):
        if self.extracted:
            if self.p1.sceneBoundingRect().contains(pos):
                x = self.p1.vb.mapSceneToView(pos).x()
                y = self.p1.vb.mapSceneToView(pos).y()
                self.ineuron = self.nROIs - y + 1
                # print(self.ineuron)

    def keyPressEvent(self, event):
        if event.modifiers() != QtCore.Qt.AltModifier and event.modifiers(
        ) != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_D:
                self.ROIs[self.iROI].remove(self)
            elif event.key() == QtCore.Qt.Key_W:
                self.viewbtns.button(0).setChecked(True)
                self.viewbtns.button(0).press(self, 0)
            elif event.key() == QtCore.Qt.Key_E:
                self.viewbtns.button(1).setChecked(True)
                self.viewbtns.button(1).press(self, 1)
            elif event.key() == QtCore.Qt.Key_R:
                self.viewbtns.button(2).setChecked(True)
                self.viewbtns.button(2).press(self, 2)
            elif event.key() == QtCore.Qt.Key_T:
                self.viewbtns.button(3).setChecked(True)
                self.viewbtns.button(3).press(self, 3)

    def add_ROI(self, pos=None):
        self.iROI = len(self.ROIs)
        self.nROIs = len(self.ROIs)
        self.ROIs.append(
            sROI(iROI=self.nROIs, parent=self, pos=pos, diameter=int(self.diam.text())))
        self.ROIs[-1].position(self)
        self.nROIs += 1
        print("%d cells added to manual GUI" % self.nROIs)
        self.closeGUI.setEnabled(False)

    def plot_clicked(self, event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x == self.img0:
                pos = self.p0.mapSceneToView(event.scenePos())
                posy = pos.x()
                posx = pos.y()
                if event.modifiers() == QtCore.Qt.AltModifier:
                    self.add_ROI(pos=np.array([
                        posx - 5, posy - 5,
                        int(self.diam.text()),
                        int(self.diam.text())
                    ]))
                if event.double():
                    self.p0.setXRange(0, self.Lx)
                    self.p0.setYRange(0, self.Ly)
            elif x == self.p1:
                if event.double():
                    self.p1.setXRange(0, self.trange.size)
                    self.p1.setYRange(self.fmin, self.fmax)

    def proc_ROI(self):
        stat0 = []
        if self.extracted:
            for t, s in zip(self.scatter, self.tlabel):
                self.p0.removeItem(s)
                self.p0.removeItem(t)
        self.scatter = []
        self.tlabel = []
        for n in range(self.nROIs):
            ellipse = self.ROIs[n].ellipse
            yrange = self.ROIs[n].yrange
            xrange = self.ROIs[n].xrange
            med = self.ROIs[n].med
            x, y = np.meshgrid(xrange, yrange)
            ypix = y[ellipse].flatten()
            xpix = x[ellipse].flatten()
            lam = np.ones(ypix.shape)
            stat0.append({
                "ypix": ypix,
                "xpix": xpix,
                "lam": lam,
                "npix": ypix.size,
                "med": med
            })
            self.tlabel.append(pg.TextItem(str(n), self.ROIs[n].color, anchor=(0, 0)))
            self.tlabel[-1].setPos(xpix.mean(), ypix.mean())
            self.p0.addItem(self.tlabel[-1])
            self.scatter.append(
                pg.ScatterPlotItem([xpix.mean()], [ypix.mean()], pen=self.ROIs[n].color,
                                   symbol="+"))
            self.p0.addItem(self.scatter[-1])
        if not os.path.isfile(self.parent.ops["reg_file"]):
            self.parent.ops["reg_file"] = os.path.join(self.parent.basename, "data.bin")

        F, Fneu, F_chan2, Fneu_chan2, spks, settings, stat = masks_and_traces(
            self.parent.ops, stat0, self.parent.stat)
        self.Fcell = F
        self.Fneu = Fneu
        self.F_chan2 = F_chan2
        self.Fneu_chan2 = Fneu_chan2
        self.Spks = spks
        self.plot_trace()
        self.extracted = True
        self.new_stat = stat
        self.closeGUI.setEnabled(True)

    def plot_trace(self):
        self.trange = np.arange(0, self.Fcell.shape[1])
        self.p1.clear()
        kspace = 1.0
        ax = self.p1.getAxis("left")
        favg = 0
        k = self.nROIs - 1
        ttick = list()
        for n in range(self.nROIs):
            f = self.Fcell[n, :]
            fneu = self.Fneu[n, :]
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            f = (f - fmin) / (fmax - fmin)
            rgb = self.ROIs[n].color
            self.p1.plot(self.trange, f + k * kspace, pen=rgb)
            fneu = (fneu - fmin) / (fmax - fmin)
            if self.nROIs == 1:
                self.p1.plot(self.trange, fneu + k * kspace, pen="r")
            ttick.append((k * kspace + f.mean(), str(n)))
            k -= 1
        self.fmax = (self.nROIs - 1) * kspace + 1
        self.fmin = 0
        ax.setTicks([ttick])
        self.p1.setXRange(0, self.Fcell.shape[1])
        # self.p1.setYRange(self.fmin, self.fmax)


class sROI():

    def __init__(self, iROI, parent=None, pos=None, diameter=None, color=None,
                 yrange=None, xrange=None):
        # what type of ROI it is

        self.iROI = iROI
        self.xrange = xrange
        self.yrange = yrange
        if color is None:
            self.color = hsv_to_rgb(np.array([np.random.rand() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * self.color)
        else:
            self.color = color
        if pos is None:
            view = parent.p0.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            if diameter is None:
                dx = np.maximum(3, np.minimum(dx, parent.Ly * 0.2))
                dy = np.maximum(3, np.minimum(dy, parent.Lx * 0.2))
            else:
                dx = diameter
                dy = diameter
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = pos[0]
            imx = pos[1]
            dy = pos[2]
            dx = pos[3]

        self.draw(parent, imy, imx, dy, dx)
        self.position(parent)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3, style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI([imx, imy], [dx, dy], pen=roipen, removable=True)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.addRotateHandle([0.5, 1], [0.5, 0.5])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.med = [imy, imx]
        parent.p0.addItem(self.ROI)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        for i in range(len(parent.ROIs)):
            if i > self.iROI:
                parent.ROIs[i].iROI -= 1
        del parent.ROIs[self.iROI]
        parent.iROI = min(len(parent.ROIs) - 1, max(0, parent.iROI))
        parent.nROIs -= 1
        parent.win.show()
        parent.show()

    def rotate_ROI(self, parent, ellipse, xrange, yrange, posx, posy):
        #Rotates ROI depending on Rotatehandle degree
        ellipse = rotate(ellipse, angle=math.floor(self.ROI.angle()), order=0)
        ellipse = np.flip(ellipse, axis=0)
        xrange = (np.arange(-1 * int(ellipse.shape[1] - 1), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(ellipse.shape[0] - 1), 1) + int(posy)).astype(np.int32)
        yrange += int(np.floor(ellipse.shape[0] / 2)) + 1
        return ellipse, xrange, yrange

    def position(self, parent):
        parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        sizex, sizey = self.ROI.size()
        pos = parent.p0.mapSceneToView(pos0[0][1])
        br = self.ROI.boundingRect()
        posy = pos.y()
        posx = pos.x()

        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        yrange += int(np.floor(sizey / 2)) + 1
        # what is ellipse circling?
        br = self.ROI.boundingRect()
        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = ((y - br.center().y())**2 / (br.height() / 2)**2 +
                   (x - br.center().x())**2 / (br.width() / 2)**2) <= 1
        if self.ROI.angle() not in (0, 180, -180):
            ellipse, xrange, yrange = self.rotate_ROI(parent, ellipse, xrange, yrange, posx, posy)
        #ensures that ROI is not placed outside of movie coordinates
        ellipse = ellipse[:, np.logical_and(xrange >= 0, xrange < parent.Lx)]
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.Lx)]
        ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.Ly), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.Ly)]

        self.ellipse = ellipse
        self.xrange = xrange
        self.yrange = yrange
