"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
# heavily modified script from a pyqt4 release
import os
import time

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QStyle
from qtpy.QtWidgets import QMainWindow, QGridLayout, QCheckBox, QLabel, QLineEdit, QSlider, QFileDialog, QPushButton, QToolButton, QButtonGroup, QWidget
from scipy.ndimage import gaussian_filter1d
from natsort import natsorted
from tifffile import imread
import json

from . import masks, views, graphics, traces, classgui, utils
from .. import registration
from ..io.save import compute_dydx
from ..io import BinaryFile


class BinaryPlayer(QMainWindow):

    def __init__(self, parent=None):
        super(BinaryPlayer, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1070, 1070)
        self.setWindowTitle("View registered binary")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        #self.p0 = pg.ViewBox(lockAspect=False,name="plot1",border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, 13, 14)
        layout = self.win.ci.layout
        self.loaded = False
        self.zloaded = False

        # A plot area (ViewBox + axes) for displaying the image
        self.vmain = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        self.win.addItem(self.vmain, row=0, col=0)
        self.vmain.setMenuEnabled(False)
        self.imain = pg.ImageItem()
        self.vmain.addItem(self.imain)
        self.cellscatter = pg.ScatterPlotItem()
        self.vmain.addItem(self.cellscatter)
        self.maskmain = pg.ImageItem()

        # side box
        self.vside = pg.ViewBox(lockAspect=True, invertY=True)
        self.vside.setMenuEnabled(False)
        self.iside = pg.ImageItem()
        self.vside.addItem(self.iside)
        self.cellscatter_side = pg.ScatterPlotItem()
        self.vside.addItem(self.cellscatter_side)
        self.maskside = pg.ImageItem()

        # view red channel
        self.redbox = QCheckBox("view red channel")
        self.redbox.setStyleSheet("color: white;")
        self.redbox.setEnabled(False)
        self.redbox.toggled.connect(self.add_red)
        self.l0.addWidget(self.redbox, 0, 5, 1, 1)
        # view masks
        self.maskbox = QCheckBox("view masks")
        self.maskbox.setStyleSheet("color: white;")
        self.maskbox.setEnabled(False)
        self.maskbox.toggled.connect(self.add_masks)
        self.l0.addWidget(self.maskbox, 0, 6, 1, 1)
        # view raw binary
        self.rawbox = QCheckBox("view raw binary")
        self.rawbox.setStyleSheet("color: white;")
        self.rawbox.setEnabled(False)
        self.rawbox.toggled.connect(self.add_raw)
        self.l0.addWidget(self.rawbox, 0, 7, 1, 1)
        # view zstack
        self.zbox = QCheckBox("view z-stack")
        self.zbox.setStyleSheet("color: white;")
        self.zbox.setEnabled(False)
        self.zbox.toggled.connect(self.add_zstack)
        self.l0.addWidget(self.zbox, 0, 8, 1, 1)

        zlabel = QLabel("Z-plane:")
        zlabel.setStyleSheet("color: white;")
        self.l0.addWidget(zlabel, 0, 9, 1, 1)

        self.Zedit = QLineEdit(self)
        self.Zedit.setValidator(QtGui.QIntValidator(0, 0))
        self.Zedit.setText("0")
        self.Zedit.setFixedWidth(30)
        self.Zedit.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.Zedit, 0, 10, 1, 1)

        self.p1 = self.win.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.setMenuEnabled(False)
        self.scatter1 = pg.ScatterPlotItem()
        self.scatter1.setData([0, 0], [0, 0])
        self.p1.addItem(self.scatter1)

        self.p2 = self.win.addPlot(name="plot_F", row=2, col=0, colspan=2)
        self.p2.setMouseEnabled(x=True, y=False)
        self.p2.setMenuEnabled(False)
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.setXLink("plot_shift")

        self.p3 = self.win.addPlot(name="plot_Z", row=3, col=0, colspan=2)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.setMenuEnabled(False)
        self.scatter3 = pg.ScatterPlotItem()
        self.p3.setXLink("plot_shift")

        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0, 12)
        self.movieLabel = QLabel("No settings chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self.createButtons(parent)
        # create ROI chooser
        self.l0.addWidget(QLabel(""), 6, 0, 1, 2)
        qlabel = QLabel(self)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, 7, 0, 1, 2)
        self.ROIedit = QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, 8, 0, 1, 1)
        # create frame slider
        self.frameLabel = QLabel("Current frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.frameNumber = QLabel("0")
        self.frameNumber.setStyleSheet("color: white;")
        self.frameSlider = QSlider(QtCore.Qt.Horizontal)
        #self.frameSlider.setTickPosition(QSlider.TicksBelow)
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameDelta = 10
        self.l0.addWidget(QLabel(""), 12, 0, 1, 1)
        self.l0.setRowStretch(12, 1)
        self.l0.addWidget(self.frameLabel, 13, 0, 1, 2)
        self.l0.addWidget(self.frameNumber, 14, 0, 1, 2)
        self.l0.addWidget(self.frameSlider, 13, 2, 14, 13)
        self.l0.addWidget(QLabel(""), 14, 1, 1, 1)
        ll = QLabel("(when paused, left/right arrow keys can move slider)")
        ll.setStyleSheet("color: white;")
        self.l0.addWidget(ll, 16, 0, 1, 3)
        #speedLabel = QLabel("Speed:")
        #self.speedSpinBox = QtGui.QSpinBox()
        #self.speedSpinBox.setRange(1, 9999)
        #self.speedSpinBox.setValue(100)
        #self.speedSpinBox.setSuffix("%")
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movieLabel, 0, 0, 1, 5)
        self.updateFrameSlider()
        self.updateButtons()
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.Floaded = False
        self.raw_on = False
        self.red_on = False
        self.z_on = False
        self.wraw = False
        self.wred = False
        self.wraw_red = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, "ops"):
            if parent.ops["save_path"][-8:] != "combined":
                self.Fcell = parent.Fcell
                self.stat = parent.stat
                self.iscell = parent.iscell
                self.settings = parent.ops
                self.basename = parent.basename
                self.ops = parent.ops
                self.Floaded = True
                self.openFile(True)

    def add_masks(self):
        print(self.allmasks[100:110,100:110])
        if self.loaded:
            if self.maskbox.isChecked():
                self.vmain.addItem(self.maskmain)
                self.vside.addItem(self.maskside)
            else:
                self.vmain.removeItem(self.maskmain)
                self.vside.removeItem(self.maskside)

    def add_red(self):
        if self.loaded:
            if self.redbox.isChecked():
                self.red_on = True
            else:
                self.red_on = False
            self.next_frame()

    def zoom_image(self):
        self.vmain.setRange(yRange=(0, self.LY), xRange=(0, self.LX))
        if self.raw_on or self.z_on:
            if self.z_on:
                self.vside.setRange(yRange=(0, self.zLy), xRange=(0, self.zLx))
            else:
                self.vside.setRange(yRange=(0, self.LY), xRange=(0, self.LX))
            self.vside.setXLink("plot1")
            self.vside.setYLink("plot1")

    def add_raw(self):
        if self.loaded:
            if self.rawbox.isChecked():
                self.raw_on = True
                self.win.addItem(self.vside, row=0, col=1)
                self.zoom_image()
            else:
                self.raw_on = False
                self.win.removeItem(self.vside)
            self.next_frame()

    def add_zstack(self):
        if self.loaded:
            if self.zbox.isChecked():
                if self.rawbox.isChecked():
                    self.rawbox.setChecked(False)
                    self.add_raw()
                self.z_on = True
                self.win.addItem(self.vside, row=0, col=1)
            else:
                self.z_on = False
                self.win.removeItem(self.vside)
            self.next_frame()

    def next_frame(self):
        
        # loop after video finishes
        self.cframe += 1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
            
        self.img = np.zeros((self.LY, self.LX), dtype=np.int16)
        for n in range(len(self.reg_loc)):
            img = self.reg_file[n][self.cframe]
            if self.wred and self.red_on:
                imgred = self.reg_file_red[n][self.cframe]
                img =  np.stack((img, imgred, np.zeros(img.shape, dtype=img.dtype)), axis=-1)
            self.img[self.dy[n]:self.dy[n] + self.Ly[n],
                     self.dx[n]:self.dx[n] + self.Lx[n]] = img    
            self.imain.setImage(self.img, levels=self.srange)
        
            if self.wraw and self.raw_on:
                if n==0:
                    self.imgraw = np.zeros((self.LY, self.LX), dtype=np.int16)
                img = self.raw_file[n][self.cframe]
                if self.wraw_red and self.red_on:
                    imgred = self.raw_file_red[n][self.cframe]
                    img =  np.stack((img, imgred, np.zeros(img.shape, dtype=img.dtype)), axis=-1)
                self.imgraw[self.dy[n]:self.dy[n] + self.Ly[n],
                        self.dx[n]:self.dx[n] + self.Lx[n]] = img
                self.iside.setImage(self.imgraw, levels=self.srange)
    
        if self.zloaded and self.z_on:
            if hasattr(self, "zmax"):
                self.Zedit.setText(str(self.zmax[self.cframe]))
            self.iside.setImage(self.zstack[int(self.Zedit.text())], levels=self.zrange)
        
        if self.maskbox.isChecked():
           #imgmin = self.img.min()
           #self.allmasks[:,:,-1] = np.maximum(0, ((self.img - imgmin) / (self.img.max() - imgmin) - 0.5)*2) * 255 * self.mask_bool
           self.maskmain.setImage(self.allmasks, levels=[0, 255])
           self.maskside.setImage(self.allmasks, levels=[0, 255])

        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        self.scatter1.setData([self.cframe, self.cframe],
                              [self.yoff[self.cframe], self.xoff[self.cframe]], size=10,
                              brush=pg.mkBrush(255, 0, 0))
        if self.Floaded:
            self.scatter2.setData([self.cframe, self.cframe],
                                  [self.ft[self.cframe], self.ft[self.cframe]], size=10,
                                  brush=pg.mkBrush(255, 0, 0))
        if self.zloaded and self.z_on:
            self.scatter3.setData([self.cframe, self.cframe],
                                  [self.zmax[self.cframe], self.zmax[self.cframe]],
                                  size=10, brush=pg.mkBrush(255, 0, 0))

    def make_masks(self):
        ncells = len(self.stat)
        np.random.seed(seed=0)
        allcols = np.random.random((ncells,))
        if hasattr(self, "redcell"):
            allcols = allcols / 1.4
            allcols = allcols + 0.1
            allcols[self.redcell] = 0
        self.colors = masks.hsv2rgb(allcols)
        self.RGB = -1 * np.ones((self.LY, self.LX, 3), np.int32)
        self.cellpix = -1 * np.ones((self.LY, self.LX), np.int32)
        self.sroi = np.zeros((self.LY, self.LX), np.uint8)
        for n in np.nonzero(self.iscell)[0]:
            ypix = self.stat[n]["ypix"].flatten()
            xpix = self.stat[n]["xpix"].flatten()
            ypix = ypix[~self.stat[n]["overlap"]]
            xpix = xpix[~self.stat[n]["overlap"]]
            yext, xext = utils.boundary(ypix, xpix)
            if len(yext) > 0:
                goodi = (yext >= 0) & (xext >= 0) & (yext < self.LY) & (xext < self.LX)
                self.stat[n]["yext"] = yext[goodi] + 0.5
                self.stat[n]["xext"] = xext[goodi] + 0.5
                self.sroi[yext[goodi], xext[goodi]] = 200
                #self.sroi[ypix, xpix] = 100
                #self.RGB[ypix, xpix] = self.colors[n]
                self.RGB[yext[goodi], xext[goodi]] = self.colors[n]
            else:
                self.stat[n]["yext"] = yext
                self.stat[n]["xext"] = xext
            self.cellpix[ypix, xpix] = n
        self.mask_bool = self.sroi > 0
        self.allmasks = np.concatenate((self.RGB, self.sroi[:, :, np.newaxis]), axis=-1)
        self.maskmain.setImage(self.allmasks, levels=[0, 255])
        self.maskside.setImage(self.allmasks, levels=[0, 255])

    def plot_trace(self):
        self.p2.clear()
        self.ft = self.Fcell[self.ichosen, :]
        self.p2.plot(self.ft, pen=self.colors[self.ichosen])
        self.p2.addItem(self.scatter2)
        self.scatter2.setData([self.cframe], [self.ft[self.cframe]], size=10,
                              brush=pg.mkBrush(255, 0, 0))
        self.p2.setLimits(yMin=self.ft.min(), yMax=self.ft.max())
        self.p2.setRange(xRange=(0, self.nframes),
                         yRange=(self.ft.min(), self.ft.max()), padding=0.0)
        self.p2.setLimits(xMin=0, xMax=self.nframes)

    def open(self):
        filename = QFileDialog.getOpenFileName(self, "Open single-plane db.npy file")
        # load plane npy files in same folder
        if filename:
            print(filename[0])
            self.openFile(filename=filename[0], fromgui=False)

    def open_combined(self):
        filename = QFileDialog.getExistingDirectory(
            self, "Load binaries for all planes (choose folder with planeX folders)")
        # load settings in same folder
        if filename:
            print(filename)
            self.openCombined(filename)

    def openCombined(self, save_folder):
        try:
            plane_folders = natsorted([
                f.path
                for f in os.scandir(save_folder)
                if f.is_dir() and f.name[:5] == "plane"
            ])
            settings1 = [
                np.load(os.path.join(f, "settings.npy"), allow_pickle=True).item()
                for f in plane_folders
            ]
            self.LY = 0
            self.LX = 0
            self.reg_loc = []
            self.reg_file = []
            self.Ly = []
            self.Lx = []
            self.dy = []
            self.dx = []
            self.wraw = False
            self.wred = False
            self.wraw_wred = False
            # check that all binaries still exist
            dy, dx = compute_dydx(settings1)
            for ipl, settings in enumerate(settings1):
                #if os.path.isfile(settings["reg_file"]):
                if os.path.isfile(settings["reg_file"]):
                    reg_file = settings["reg_file"]
                else:
                    reg_file = os.path.abspath(
                        os.path.join(os.path.dirname(filename), "plane%d" % ipl,
                                     "data.bin"))
                print(reg_file, os.path.isfile(reg_file))
                self.reg_loc.append(reg_file)
                self.reg_file.append(open(self.reg_loc[-1], "rb"))
                self.Ly.append(settings["Ly"])
                self.Lx.append(settings["Lx"])
                self.dy.append(dy[ipl])
                self.dx.append(dx[ipl])
                self.LY = np.maximum(self.LY, self.Ly[-1] + self.dy[-1])
                self.LX = np.maximum(self.LX, self.Lx[-1] + self.dx[-1])
                good = True
            self.Floaded = False

        except Exception as e:
            print("ERROR: %s" % e)
            print("(could be incorrect folder or missing binaries)")
            good = False
            try:
                for n in range(len(self.reg_loc)):
                    self.reg_file[n].close()
                print("closed binaries")
            except:
                print("tried to close binaries")
        if good:
            self.filename = save_folder
            self.settings = settings1
            self.setup_views()

    def openFile(self, fromgui=True, filename=None):
        
        if filename is not None:
            ext = os.path.splitext(filename)[1]
            db = np.load(filename, allow_pickle=True).item()
            dirname = os.path.dirname(filename)
            settings = np.load(os.path.join(dirname, "settings.npy"), allow_pickle=True).item()
            ops = {**db, **settings}
            try:
                reg_outputs = np.load(os.path.join(dirname, "reg_outputs.npy"), allow_pickle=True).item()
                ops = {**ops, **reg_outputs}
            except:
                print("no reg_outputs.npy found")
            self.ops = ops
            self.basename = dirname
            fromgui = False

        if 1:
            ops = self.ops
            self.LY = ops["Ly"]
            self.LX = ops["Lx"]
            self.Ly = [ops["Ly"]]
            self.Lx = [ops["Lx"]]
            self.dx = [0]
            self.dy = [0]

            if os.path.isfile(ops["reg_file"]):
                self.reg_loc = [ops["reg_file"]]
            else:
                self.reg_loc = [
                    os.path.abspath(os.path.join(self.basename, "data.bin"))
                ]
            reg_folder = os.path.dirname(self.reg_loc[0])
            self.wraw_wred = False
            self.raw_loc = [os.path.abspath(os.path.join(reg_folder, "data_raw.bin"))]
            self.wraw = os.path.exists(self.raw_loc[0])
            self.reg_loc_red = [os.path.abspath(os.path.join(reg_folder, "data_chan2.bin"))]
            self.wred = os.path.exists(self.reg_loc_red[0])
            self.raw_loc_red = [os.path.abspath(os.path.join(reg_folder, "data_raw_chan2.bin"))]
            self.wraw_red = os.path.exists(self.raw_loc_red[0])
            
            self.open_binaries()

            if not fromgui:
                if os.path.isfile(os.path.join(self.basename, "F.npy")):
                    self.Fcell = np.load(os.path.join(self.basename, "F.npy"))
                    self.stat = np.load(
                        os.path.abspath(
                            os.path.join(os.path.dirname(filename), "stat.npy")),
                        allow_pickle=True)
                    self.iscell = np.load(
                        os.path.abspath(
                            os.path.join(os.path.dirname(filename), "iscell.npy")),
                        allow_pickle=True)
                    self.Floaded = True
                else:
                    self.Floaded = False
            else:
                self.Floaded = True
            
        print(self.Floaded)
        self.filename = filename
        self.setup_views()

        #except Exception as e:
        #    print("ERROR: db.npy incorrect / missing db['reg_file'] and others")
        #    print(e)
            
    def open_binaries(self):
        print(self.reg_loc)
        self.reg_file = [BinaryFile(Ly, Lx, fname, write=False) 
                            for Ly, Lx, fname in zip(self.Ly, self.Lx, self.reg_loc)]
        if self.wraw:
            self.raw_file = [BinaryFile(Ly, Lx, fname, write=False) 
                                for Ly, Lx, fname in zip(self.Ly, self.Lx, self.raw_loc)]
        if self.wred:
            self.reg_file_red = [BinaryFile(Ly, Lx, fname, write=False) 
                            for Ly, Lx, fname in zip(self.Ly, self.Lx, self.reg_loc_red)]
            if self.wraw_red:
                self.raw_file_red = [BinaryFile(Ly, Lx, fname, write=False) 
                            for Ly, Lx, fname in zip(self.Ly, self.Lx, self.raw_loc_red)]
            

    def setup_views(self):
        self.p1.clear()
        self.p2.clear()
        self.ichosen = 0
        self.ROIedit.setText("0")
        # get scaling from 100 random frames
        frames = subsample_frames(self.ops, np.minimum(self.ops["nframes"] - 1, 100),
                                  self.reg_loc[-1])
        self.srange = frames.mean() + frames.std() * np.array([-2, 5])

        self.movieLabel.setText(self.reg_loc[-1])
        
        #aspect ratio
        if "aspect" in self.ops:
            self.xyrat = self.ops["aspect"]
        elif "diameter" in self.ops and (type(self.ops["diameter"]) is not int) and (len(
                self.ops["diameter"]) > 1):
            self.xyrat = self.ops["diameter"][0] / self.ops["diameter"][1]
        else:
            self.xyrat = 1.0
        self.vmain.setAspectLocked(lock=True, ratio=self.xyrat)
        self.vside.setAspectLocked(lock=True, ratio=self.xyrat)

        self.nframes = self.ops["nframes"]
        self.time_step = max(1, int(np.round(1. / self.ops["fs"] * 1000 / 5)))  # 5x real-time
        self.frameDelta = int(np.maximum(5, self.nframes / 200))
        self.frameSlider.setSingleStep(self.frameDelta)
        self.currentMovieDirectory = QtCore.QFileInfo(self.filename).path()
        if self.nframes > 0:
            self.updateFrameSlider()
            self.updateButtons()
        # plot self.ops X-Y offsets
        if "yoff" in self.ops:
            self.yoff = self.ops["yoff"]
            self.xoff = self.ops["xoff"]
        else:
            self.yoff = np.zeros((self.ops["nframes"],))
            self.xoff = np.zeros((self.ops["nframes"],))
        self.p1.plot(self.yoff, pen="g")
        self.p1.plot(self.xoff, pen="y")
        self.p1.setRange(
            xRange=(0, self.nframes),
            yRange=(np.minimum(self.yoff.min(), self.xoff.min()),
                    np.maximum(self.yoff.max(), self.xoff.max())), padding=0.0)
        self.p1.setLimits(xMin=0, xMax=self.nframes)
        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        self.scatter1.setData([self.cframe, self.cframe],
                              [self.yoff[self.cframe], self.xoff[self.cframe]], size=10,
                              brush=pg.mkBrush(255, 0, 0))

        if self.wraw:
            self.rawbox.setEnabled(True)
        else:
            self.rawbox.setEnabled(False)
        if self.wred:
            self.redbox.setEnabled(True)
        else:
            self.redbox.setEnabled(False)

        if self.Floaded:
            self.maskbox.setEnabled(True)
            self.make_masks()
            self.cell_chosen()

        self.cframe = -1
        self.loaded = True
        self.next_frame()

    def keyPressEvent(self, event):
        bid = -1
        if self.playButton.isEnabled():
            if event.modifiers() != QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Left:
                    self.cframe -= self.frameDelta
                    self.cframe = np.maximum(0, np.minimum(self.nframes - 1,
                                                           self.cframe))
                    self.frameSlider.setValue(self.cframe)
                elif event.key() == QtCore.Qt.Key_Right:
                    self.cframe += self.frameDelta
                    self.cframe = np.maximum(0, np.minimum(self.nframes - 1,
                                                           self.cframe))
                    self.frameSlider.setValue(self.cframe)
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    # then play
                    self.start()
                else:
                    self.pause()

    def number_chosen(self):
        self.ichosen = int(self.ROIedit.text())
        self.cell_chosen()

    def cell_chosen(self):
        if self.Floaded:
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            
            self.cell_mask()
            self.ROIedit.setText(str(self.ichosen))
            rgb = np.array(self.colors[self.ichosen])
            self.cellscatter.setData(self.xext, self.yext, pen=pg.mkPen(list(rgb)),
                                     brush=pg.mkBrush(list(rgb)), size=3)
            self.cellscatter_side.setData(self.xext, self.yext, pen=pg.mkPen(list(rgb)),
                                          brush=pg.mkBrush(list(rgb)), size=3)

            self.ft = self.Fcell[self.ichosen, :]
            self.plot_trace()
            self.p2.setXLink("plot_shift")
            self.jump_to_frame()
            self.show()

    def plot_clicked(self, event):
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        zoom = False
        zoomImg = False
        choose = False
        if self.loaded:
            for x in items:
                if x == self.p1:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 1
                elif x == self.p2 and self.Floaded:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 2
                elif x == self.vmain or x == self.vside:
                    if event.button() == QtCore.Qt.LeftButton:
                        if event.double():
                            self.zoom_image()
                        else:
                            if self.Floaded:
                                pos = x.mapSceneToView(event.scenePos())
                                posy = int(pos.x())
                                posx = int(pos.y())
                                print(posy, posx, self.cellpix[posx, posy])
                                if posy >= 0 and posy < self.LX and posx >= 0 and posx < self.LY:
                                    if self.cellpix[posx, posy] > -1:
                                        self.ichosen = self.cellpix[posx, posy]
                                        self.cell_chosen()
                if iplot == 1 or iplot == 2:
                    if event.button() == QtCore.Qt.LeftButton:
                        if event.double():
                            zoom = True
                        else:
                            choose = True
        if zoom:
            self.p1.setRange(xRange=(0, self.nframes))
            self.p2.setRange(xRange=(0, self.nframes))
            self.p3.setRange(xRange=(0, self.nframes))

        if choose:
            if self.playButton.isEnabled():
                self.cframe = np.maximum(
                    0, np.minimum(self.nframes - 1, int(np.round(posx))))
                self.frameSlider.setValue(self.cframe)
                #self.jump_to_frame()

    def load_zstack(self):
        name = QFileDialog.getOpenFileName(self, "Open zstack", filter="*.tif")
        self.fname = name[0]
        try:
            self.zstack = imread(self.fname)
            self.zLy, self.zLx = self.zstack.shape[1:]
            self.Zedit.setValidator(QtGui.QIntValidator(0, self.zstack.shape[0]))
            self.zrange = [
                np.percentile(self.zstack, 1),
                np.percentile(self.zstack, 99)
            ]

            self.computeZ.setEnabled(True)
            self.zloaded = True
            self.zbox.setEnabled(True)
            self.zbox.setChecked(True)
            self.zmax = np.zeros(self.nframes, "int")
            if "zcorr" in self.self.ops:
                if self.zstack.shape[0] == self.ops["zcorr"].shape[0]:
                    zcorr = self.ops["zcorr"]
                    self.zmax = np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1),
                                          axis=1)
                    self.plot_zcorr()

        except Exception as e:
            print("ERROR: %s" % e)

    def cell_mask(self):
        #self.cmask = np.zeros((self.Ly,self.Lx,3),np.float32)
        self.yext = self.stat[self.ichosen]["yext"]
        self.xext = self.stat[self.ichosen]["xext"]
        #self.cmask[self.yext,self.xext,2] = (self.srange[1]-self.srange[0])/2 * np.ones((self.yext.size,),np.float32)

    def go_to_frame(self):
        self.cframe = int(self.frameSlider.value())
        self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.frameSlider.setMaximum(self.nframes - 1)
        self.frameSlider.setMinimum(0)
        self.frameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def createButtons(self, parent):
        iconSize = QtCore.QSize(30, 30)
        openButton = QPushButton("load db.npy")
        openButton.setToolTip("Open a single-plane db.npy")
        openButton.clicked.connect(self.open)

        openButton2 = QPushButton("load folder")
        openButton2.setToolTip("Choose a folder with planeX folders to load together")
        openButton2.clicked.connect(self.open_combined)

        loadZ = QPushButton("load z-stack tiff")
        loadZ.clicked.connect(self.load_zstack)

        self.computeZ = QPushButton("compute z position")
        self.computeZ.setEnabled(False)
        self.computeZ.clicked.connect(lambda: self.compute_z(parent))

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QButtonGroup(self)
        btns.addButton(self.playButton, 0)
        btns.addButton(self.pauseButton, 1)
        btns.setExclusive(True)

        quitButton = QToolButton()
        quitButton.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        quitButton.setIconSize(iconSize)
        quitButton.setToolTip("Quit")
        quitButton.clicked.connect(self.close)

        self.l0.addWidget(openButton, 1, 0, 1, 2)
        self.l0.addWidget(openButton2, 2, 0, 1, 2)
        self.l0.addWidget(loadZ, 3, 0, 1, 2)
        self.l0.addWidget(self.computeZ, 4, 0, 1, 2)
        self.l0.addWidget(self.playButton, 15, 0, 1, 1)
        self.l0.addWidget(self.pauseButton, 15, 1, 1, 1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def jump_to_frame(self):
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
            self.cframe = int(self.cframe)
            self.cframe -= 1
            self.next_frame()

    def start(self):
        if self.cframe < self.nframes - 1:
            print("playing")
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(self.time_step)

    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        print("paused")

    def compute_z(self, parent):
        settings, zcorr = registration.compute_zpos(self.zstack, self.settings[0])
        parent.ops = settings
        self.zmax = np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
        np.save(self.filename, self.ops)
        self.plot_zcorr()

    def plot_zcorr(self):
        self.p3.clear()
        self.p3.plot(self.zmax, pen="r")
        self.p3.addItem(self.scatter3)
        self.p3.setRange(xRange=(0, self.nframes),
                         yRange=(self.zmax.min(), self.zmax.max() + 3), padding=0.0)
        self.p3.setLimits(xMin=0, xMax=self.nframes)
        self.p3.setXLink("plot_shift")


def subsample_frames(ops, nsamps, reg_loc):
    nFrames = ops["nframes"]
    Ly = ops["Ly"]
    Lx = ops["Lx"]
    frames = np.zeros((nsamps, Ly, Lx), dtype="int16")
    nbytesread = 2 * Ly * Lx
    istart = np.linspace(0, nFrames, 1 + nsamps).astype("int64")
    reg_file = open(reg_loc, "rb")
    for j in range(0, nsamps):
        reg_file.seek(nbytesread * istart[j], 0)
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        frames[j, :, :] = np.reshape(data, (Ly, Lx))
    reg_file.close()
    return frames


class PCViewer(QMainWindow):

    def __init__(self, parent=None):
        super(PCViewer, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1300, 800)
        self.setWindowTitle("Metrics for registration")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)

        #self.p0 = pg.ViewBox(lockAspect=False,name="plot1",border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 0, 2, 13, 14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p3 = self.win.addPlot(row=0, col=0)
        self.p3.setMouseEnabled(x=False, y=False)
        self.p3.setMenuEnabled(False)

        self.p0 = self.win.addViewBox(name="plot1", lockAspect=True, row=1, col=0,
                                      invertY=True)
        self.p1 = self.win.addViewBox(lockAspect=True, row=1, col=1, invertY=True)
        self.p1.setMenuEnabled(False)
        self.p1.setXLink("plot1")
        self.p1.setYLink("plot1")
        self.p2 = self.win.addViewBox(lockAspect=True, row=1, col=2, invertY=True)
        self.p2.setMenuEnabled(False)
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")
        self.img0 = pg.ImageItem()
        self.img1 = pg.ImageItem()
        self.img2 = pg.ImageItem()
        self.p0.addItem(self.img0)
        self.p1.addItem(self.img1)
        self.p2.addItem(self.img2)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.p4 = self.win.addPlot(row=0, col=1, colspan=2)
        self.p4.setMouseEnabled(x=False)
        self.p4.setMenuEnabled(False)

        self.PCedit = QLineEdit(self)
        self.PCedit.setText("1")
        self.PCedit.setFixedWidth(40)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.plot_frame)
        self.PCedit.textEdited.connect(self.pause)
        qlabel = QLabel("PC: ")
        boldfont = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        bigfont = QtGui.QFont("Arial", 14)
        qlabel.setFont(boldfont)
        self.PCedit.setFont(bigfont)
        qlabel.setStyleSheet("color: white;")
        #qlabel.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(QLabel(""), 1, 0, 1, 1)
        self.l0.addWidget(qlabel, 2, 0, 1, 1)
        self.l0.addWidget(self.PCedit, 2, 1, 1, 1)
        self.nums = []
        self.titles = []
        for j in range(3):
            num1 = QLabel("")
            num1.setStyleSheet("color: white;")
            self.l0.addWidget(num1, 3 + j, 0, 1, 2)
            self.nums.append(num1)
            t1 = QLabel("")
            t1.setStyleSheet("color: white;")
            self.l0.addWidget(t1, 12, 4 + j * 4, 1, 2)
            self.titles.append(t1)
        self.loaded = False
        self.wraw = False
        self.wred = False
        self.wraw_wred = False
        self.l0.addWidget(QLabel(""), 7, 0, 1, 1)
        self.l0.setRowStretch(7, 1)
        self.cframe = 0
        self.createButtons()
        self.nPCs = 50
        self.PCedit.setValidator(QtGui.QIntValidator(1, self.nPCs))
        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        #self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, "ops"):
            if parent.ops["save_path"][-8:] != "combined":
                self.ops = parent.ops
                self.openFile()
            else:
                filename = os.path.abspath(os.path.join(parent.basename, "db.npy"))
                print(filename)
                self.openFile(filename)

    def createButtons(self):
        iconSize = QtCore.QSize(30, 30)
        openButton = QToolButton()
        openButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        openButton.setIconSize(iconSize)
        openButton.setToolTip("Open ops/reg_outputs file")
        openButton.clicked.connect(self.open)

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QButtonGroup(self)
        btns.addButton(self.playButton, 0)
        btns.addButton(self.pauseButton, 1)
        btns.setExclusive(True)

        self.l0.addWidget(openButton, 0, 0, 1, 1)
        self.l0.addWidget(self.playButton, 14, 12, 1, 1)
        self.l0.addWidget(self.pauseButton, 14, 13, 1, 1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def start(self):
        if self.loaded:
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.updateTimer.start(200)

    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.pauseButton.setEnabled(False)

    def open(self):
        filename = QFileDialog.getOpenFileName(self, "Open ops.npy or reg_outputs.npy file",
                                               filter="*.npy")
        # load ops in same folder
        if filename:
            print(filename[0])
            self.openFile(filename[0])

    def openFile(self, filename=None):
        if filename is not None:
            try:
                ops = np.load(filename, allow_pickle=True).item()
            except Exception as e:
                print("ERROR: ops.npy incorrect / missing ops['regPC'] and ops['regDX']")
                print(e)
                good = False            
        else:
            self.PC = self.ops["regPC"]
            self.PC = np.clip(self.PC, np.percentile(self.PC, 1),
                              np.percentile(self.PC, 99))

            self.Ly, self.Lx = self.PC.shape[2:]
            self.DX = self.ops["regDX"]
            if "tPC" in self.ops:
                self.tPC = self.ops["tPC"]
            else:
                self.tPC = np.zeros((1, self.PC.shape[1]))
            good = True

        if good:
            self.loaded = True
            self.nPCs = self.PC.shape[1]
            self.PCedit.setValidator(QtGui.QIntValidator(1, self.nPCs))
            self.plot_frame()
            self.playButton.setEnabled(True)

    def next_frame(self):
        iPC = int(self.PCedit.text()) - 1
        pc1 = self.PC[1, iPC, :, :]
        pc0 = self.PC[0, iPC, :, :]
        if self.cframe == 0:
            self.img2.setImage(np.tile(pc0[:, :, np.newaxis], (1, 1, 3)))
            self.titles[2].setText("top")

        else:
            self.img2.setImage(np.tile(pc1[:, :, np.newaxis], (1, 1, 3)))
            self.titles[2].setText("bottom")

        self.img2.setLevels([pc0.min(), pc0.max()])
        self.cframe = 1 - self.cframe

    def plot_frame(self):
        if self.loaded:
            self.titles[0].setText("difference")
            self.titles[1].setText("merged")
            self.titles[2].setText("top")
            iPC = int(self.PCedit.text()) - 1
            pc1 = self.PC[1, iPC, :, :]
            pc0 = self.PC[0, iPC, :, :]
            diff = pc1[:, :, np.newaxis] - pc0[:, :, np.newaxis]
            diff /= np.abs(diff).max() * 2
            diff += 0.5
            self.img0.setImage(np.tile(diff * 255, (1, 1, 3)))
            self.img0.setLevels([0, 255])
            rgb = np.zeros((self.PC.shape[2], self.PC.shape[3], 3), np.float32)
            rgb[:, :, 0] = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255
            rgb[:, :, 1] = np.minimum(
                1, np.maximum(0, (pc0 - pc1.min()) / (pc1.max() - pc1.min()))) * 255
            rgb[:, :, 2] = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255
            self.img1.setImage(rgb)
            if self.cframe == 0:
                self.img2.setImage(np.tile(pc0[:, :, np.newaxis], (1, 1, 3)))
            else:
                self.img2.setImage(np.tile(pc1[:, :, np.newaxis], (1, 1, 3)))
            self.img2.setLevels([pc0.min(), pc0.max()])
            self.zoom_plot()
            self.p3.clear()
            p = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
            ptitle = ["rigid", "nonrigid", "nonrigid max"]
            if not hasattr(self, "leg"):
                self.leg = pg.LegendItem((100, 60), offset=(350, 30))
                self.leg.setParentItem(self.p3)
                drawLeg = True
            else:
                drawLeg = False
            for j in range(3):
                cj = self.p3.plot(np.arange(1, self.nPCs + 1), self.DX[:, j], pen=p[j])
                if drawLeg:
                    self.leg.addItem(cj, ptitle[j])
                self.nums[j].setText("%s: %1.3f" % (ptitle[j], self.DX[iPC, j]))
            self.scatter = pg.ScatterPlotItem()
            self.p3.addItem(self.scatter)
            #print(self.DX.shape)
            self.scatter.setData([iPC + 1, iPC + 1, iPC + 1], self.DX[iPC, :3].tolist(),
                                 size=10, brush=pg.mkBrush(255, 255, 255))
            self.p3.setLabel("left", "pixel shift")
            self.p3.setLabel("bottom", "PC #")

            self.p4.clear()
            self.p4.plot(self.tPC[:, iPC])
            self.p4.setLabel("left", "magnitude")
            self.p4.setLabel("bottom", "time")
            self.show()
            self.zoom_plot()

    def zoom_plot(self):
        self.p0.setXRange(0, self.Lx)
        self.p0.setYRange(0, self.Ly)
        self.p1.setXRange(0, self.Lx)
        self.p1.setYRange(0, self.Ly)
        self.p2.setXRange(0, self.Lx)
        self.p2.setYRange(0, self.Ly)

    def plot_clicked(self, event):
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        zoom = False
        if self.loaded:
            for x in items:
                if x == self.p0 or x == self.p1 or x == self.p2:
                    if event.button() == QtCore.Qt.LeftButton:
                        if event.double():
                            zoom = True
                            self.zoom_plot()

    def keyPressEvent(self, event):
        bid = -1
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = max(ipc - 1, 1)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Right:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = min(ipc + 1, self.nPCs)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    # then play
                    self.playButton.setChecked(True)
                    self.start()
                else:
                    self.pause()
