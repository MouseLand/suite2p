# heavily modified script from a pyqt4 release
import os
import time

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from scipy.ndimage import gaussian_filter1d
from tifffile import imread

from . import masks, views, graphics, traces, classgui
from . import utils
from .io import enable_views_and_classifier
from .. import registration


class BinaryPlayer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(BinaryPlayer, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1070,1070)
        self.setWindowTitle('View registered binary')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,1,2,13,14)
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
        self.redbox = QtGui.QCheckBox("view red channel")
        self.redbox.setStyleSheet("color: white;")
        self.redbox.setEnabled(False)
        self.redbox.toggled.connect(self.add_red)
        self.l0.addWidget(self.redbox, 0, 5, 1, 1)
        # view masks
        self.maskbox = QtGui.QCheckBox("view masks")
        self.maskbox.setStyleSheet("color: white;")
        self.maskbox.setEnabled(False)
        self.maskbox.toggled.connect(self.add_masks)
        self.l0.addWidget(self.maskbox, 0, 6, 1, 1)
        # view raw binary
        self.rawbox = QtGui.QCheckBox("view raw binary")
        self.rawbox.setStyleSheet("color: white;")
        self.rawbox.setEnabled(False)
        self.rawbox.toggled.connect(self.add_raw)
        self.l0.addWidget(self.rawbox, 0, 7, 1, 1)
        # view zstack
        self.zbox = QtGui.QCheckBox("view z-stack")
        self.zbox.setStyleSheet("color: white;")
        self.zbox.setEnabled(False)
        self.zbox.toggled.connect(self.add_zstack)
        self.l0.addWidget(self.zbox, 0, 8, 1, 1)

        zlabel = QtGui.QLabel('Z-plane:')
        zlabel.setStyleSheet("color: white;")
        self.l0.addWidget(zlabel, 0, 9, 1, 1)

        self.Zedit = QtGui.QLineEdit(self)
        self.Zedit.setValidator(QtGui.QIntValidator(0, 0))
        self.Zedit.setText('0')
        self.Zedit.setFixedWidth(30)
        self.Zedit.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.Zedit, 0, 10, 1, 1)


        self.p1 = self.win.addPlot(name='plot_shift',row=1,col=0,colspan=2)
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        self.scatter1 = pg.ScatterPlotItem()
        self.scatter1.setData([0,0],[0,0])
        self.p1.addItem(self.scatter1)

        self.p2 = self.win.addPlot(name='plot_F',row=2,col=0,colspan=2)
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.setXLink('plot_shift')

        self.p3 = self.win.addPlot(name='plot_Z',row=3,col=0,colspan=2)
        self.p3.setMouseEnabled(x=True,y=False)
        self.p3.setMenuEnabled(False)
        self.scatter3 = pg.ScatterPlotItem()
        self.p3.setXLink('plot_shift')

        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0,12)
        self.movieLabel = QtGui.QLabel("No ops chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self.createButtons()
        # create ROI chooser
        self.l0.addWidget(QtGui.QLabel(''),6,0,1,2)
        qlabel = QtGui.QLabel(self)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel,7,0,1,2)
        self.ROIedit = QtGui.QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0,10000))
        self.ROIedit.setText('0')
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, 8,0,1,1)
        # create frame slider
        self.frameLabel = QtGui.QLabel("Current frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.frameNumber = QtGui.QLabel("0")
        self.frameNumber.setStyleSheet("color: white;")
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        #self.frameSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameDelta = 10
        self.l0.addWidget(QtGui.QLabel(''),12,0,1,1)
        self.l0.setRowStretch(12,1)
        self.l0.addWidget(self.frameLabel, 13,0,1,2)
        self.l0.addWidget(self.frameNumber, 14,0,1,2)
        self.l0.addWidget(self.frameSlider, 13,2,14,13)
        self.l0.addWidget(QtGui.QLabel(''),14,1,1,1)
        ll = QtGui.QLabel('(when paused, left/right arrow keys can move slider)')
        ll.setStyleSheet("color: white;")
        self.l0.addWidget(ll,16,0,1,3)
        #speedLabel = QtGui.QLabel("Speed:")
        #self.speedSpinBox = QtGui.QSpinBox()
        #self.speedSpinBox.setRange(1, 9999)
        #self.speedSpinBox.setValue(100)
        #self.speedSpinBox.setSuffix("%")
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movieLabel,0,0,1,5)
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
        self.wraw_wred = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, 'ops'):
            if parent.ops['save_path'][-8:]!='combined':
                filename = os.path.abspath(os.path.join(parent.basename, 'ops.npy'))
                print(filename)
                self.Fcell = parent.Fcell
                self.stat = parent.stat
                self.iscell = parent.iscell
                self.Floaded = True
                self.openFile(filename, True)

    def add_masks(self):
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
        self.vmain.setRange(yRange=(0,self.LY),xRange=(0,self.LX))
        if self.raw_on or self.z_on:
            if self.z_on:
                self.vside.setRange(yRange=(0,self.zLy),xRange=(0,self.zLx))
            else:
                self.vside.setRange(yRange=(0,self.LY),xRange=(0,self.LX))
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
        self.cframe+=1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
            if self.LY>0:
                for n in range(len(self.reg_file)):
                    self.reg_file[n].seek(0, 0)
            else:
                self.reg_file.seek(0, 0)
                if self.wraw:
                    self.reg_file_raw.seek(0, 0)
                if self.wred:
                    self.reg_file_chan2.seek(0, 0)
                if self.wraw_wred:
                    self.reg_file_raw_chan2.seek(0, 0)
        self.img = np.zeros((self.LY, self.LX), dtype=np.int16)
        ichan = np.arange(0,3,1,int)
        for n in range(len(self.reg_loc)):
            buff = self.reg_file[n].read(self.nbytesread[n])
            img = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly[n],self.Lx[n]))
            img = img[self.ycrop[n][0]:self.ycrop[n][1], self.xcrop[n][0]:self.xcrop[n][1]]
            self.img[self.yrange[n][0]:self.yrange[n][1], self.xrange[n][0]:self.xrange[n][1]] = img
        if self.wred and self.red_on:
            buff = self.reg_file_chan2.read(self.nbytesread[0])
            imgred = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly[0],self.Lx[0]))[:,:,np.newaxis]
            self.img = np.concatenate((self.img[:,:,np.newaxis], imgred, np.zeros_like(imgred)), axis=-1)
        if self.wraw and self.raw_on:
            buff = self.reg_file_raw.read(self.nbytesread[0])
            self.imgraw = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly[0],self.Lx[0]))
            if self.wraw_wred:
                buff = self.reg_file_raw_chan2.read(self.nbytesread[0])
                imgred_raw = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly[0],self.Lx[0]))[:,:,np.newaxis]
                self.imgraw = np.concatenate((self.imgraw[:,:,np.newaxis], imgred_raw, np.zeros_like(imgred_raw)), axis=-1)
            self.iside.setImage(self.imgraw, levels=self.srange)
        if self.zloaded and self.z_on:
            if hasattr(self, 'zmax'):
                self.Zedit.setText(str(self.zmax[self.cframe]))
            self.iside.setImage(self.zstack[int(self.Zedit.text())], levels=self.zrange)

        self.imain.setImage(self.img, levels=self.srange)
        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        self.scatter1.setData([self.cframe,self.cframe],
                              [self.yoff[self.cframe],self.xoff[self.cframe]],
                              size=10,brush=pg.mkBrush(255,0,0))
        if self.Floaded:
            self.scatter2.setData([self.cframe,self.cframe],
                                    [self.ft[self.cframe],self.ft[self.cframe]],size=10,
                                    brush=pg.mkBrush(255,0,0))
        if self.zloaded and self.z_on:
            self.scatter3.setData([self.cframe,self.cframe],
                                  [self.zmax[self.cframe],self.zmax[self.cframe]],
                                  size=10,brush=pg.mkBrush(255,0,0))
    def make_masks(self):
        ncells = len(self.stat)
        np.random.seed(seed=0)
        allcols = np.random.random((ncells,))
        if hasattr(self, 'redcell'):
            allcols = allcols / 1.4
            allcols = allcols + 0.1
            allcols[self.redcell] = 0
        self.colors = masks.hsv2rgb(allcols)
        self.RGB = -1*np.ones((self.LY, self.LX, 3), np.int32)
        self.cellpix = -1*np.ones((self.LY, self.LX), np.int32)
        self.sroi = np.zeros((self.LY, self.LX), np.uint8)
        for n in np.nonzero(self.iscell)[0]:
            ypix = self.stat[n]['ypix'].flatten()
            xpix = self.stat[n]['xpix'].flatten()
            if not self.ops[0]['allow_overlap']:
                ypix = ypix[~self.stat[n]['overlap']]
                xpix = xpix[~self.stat[n]['overlap']]
            iext = utils.boundary(ypix, xpix)
            yext = ypix[iext]
            xext = xpix[iext]
            goodi = (yext>=0) & (xext>=0) & (yext<self.LY) & (xext<self.LX)
            self.stat[n]['yext'] = yext[goodi] + 0.5
            self.stat[n]['xext'] = xext[goodi] + 0.5
            self.cellpix[ypix, xpix] = n
            self.sroi[yext[goodi], xext[goodi]] = 200
            self.RGB[yext[goodi], xext[goodi]] = self.colors[n]

        self.allmasks = np.concatenate((self.RGB,
                                        self.sroi[:,:,np.newaxis]), axis=-1)
        self.maskmain.setImage(self.allmasks, levels=[0, 255])
        self.maskside.setImage(self.allmasks, levels=[0, 255])

    def plot_trace(self):
        self.p2.clear()
        self.ft = self.Fcell[self.ichosen,:]
        self.p2.plot(self.ft, pen=self.colors[self.ichosen])
        self.p2.addItem(self.scatter2)
        self.scatter2.setData([self.cframe],[self.ft[self.cframe]],size=10,
                                brush=pg.mkBrush(255,0,0))
        self.p2.setLimits(yMin=self.ft.min(), yMax=self.ft.max())
        self.p2.setRange(xRange=(0,self.nframes),
                         yRange=(self.ft.min(),self.ft.max()),
                         padding=0.0)
        self.p2.setLimits(xMin=0,xMax=self.nframes)

    def open(self):
        filename = QtGui.QFileDialog.getOpenFileName(self,
                            "Open single-plane ops.npy file",filter="ops*.npy")
        # load ops in same folder
        if filename:
            print(filename[0])
            self.openFile(filename[0], False)

    def open_combined(self):
        filename = QtGui.QFileDialog.getOpenFileName(self,
                            "Open multiplane ops1.npy file",filter="ops*.npy")
        # load ops in same folder
        if filename:
            print(filename[0])
            self.openCombined(filename[0])

    def openCombined(self, filename):
        try:
            ops1 = np.load(filename, allow_pickle=True)
            basefolder = ops1[0]['save_path0']
            #opsCombined = np.load(os.path.abspath(os.path.join(basefolder, 'suite2p/combined/ops.npy'), allow_pickle=True).item()
            #self.LY = opsCombined['Ly']
            #self.LX = opsCombined['Lx']
            self.LY = 0
            self.LX = 0
            self.reg_loc = []
            self.reg_file = []
            self.Ly = []
            self.Lx = []
            self.dy = []
            self.dx = []
            self.yrange = []
            self.xrange = []
            self.ycrop  = []
            self.xcrop  = []
            self.wraw = False
            self.wred = False
            self.wraw_wred = False
            # check that all binaries still exist
            for ipl,ops in enumerate(ops1):
                #if os.path.isfile(ops['reg_file']):
                if os.path.isfile(ops['reg_file']):
                    reg_file = ops['reg_file']
                else:
                    reg_file = os.path.abspath(os.path.join(os.path.dirname(filename),'plane%d'%ipl, 'data.bin'))
                print(reg_file, os.path.isfile(reg_file))
                self.reg_loc.append(reg_file)
                self.reg_file.append(open(self.reg_loc[-1], 'rb'))
                self.Ly.append(ops['Ly'])
                self.Lx.append(ops['Lx'])
                self.dy.append(ops['dy'])
                self.dx.append(ops['dx'])
                xrange = ops['xrange']
                yrange = ops['yrange']
                self.ycrop.append(yrange)
                self.xcrop.append(xrange)
                self.yrange.append([self.dy[-1]+yrange[0], self.dy[-1]+yrange[1]])
                self.xrange.append([self.dx[-1]+xrange[0], self.dx[-1]+xrange[1]])
                self.LY = np.maximum(self.LY, self.Ly[-1]+self.dy[-1])
                self.LX = np.maximum(self.LX, self.Lx[-1]+self.dx[-1])
                good = True
            self.Floaded = False
            if not fromgui:
                if os.path.isfile(os.path.abspath(os.path.join(os.path.dirname(filename), 'combined', 'F.npy'))):
                    self.Fcell = np.load(os.path.abspath(os.path.join(os.path.dirname(filename), 'combined', 'F.npy')))
                    self.stat =  np.load(os.path.abspath(os.path.join(os.path.dirname(filename), 'combined', 'stat.npy')), allow_pickle=True)
                    self.iscell =  np.load(os.path.abspath(os.path.join(os.path.dirname(filename), 'combined', 'iscell.npy')), allow_pickle=True)
                    self.Floaded = True
                else:
                    self.Floaded = False
            else:
                self.Floaded = True
        except Exception as e:
            print("ERROR: incorrect ops1.npy or missing binaries")
            good = False
            try:
                for n in range(len(self.reg_loc)):
                    self.reg_file[n].close()
                print('closed binaries')
            except:
                print('tried to close binaries')
        if good:
            self.filename = filename
            self.ops = ops1
            self.setup_views()

    def openFile(self, filename, fromgui):
        try:
            ops = np.load(filename, allow_pickle=True).item()
            self.LY = ops['Ly']
            self.LX = ops['Lx']
            self.Ly = [ops['Ly']]
            self.Lx = [ops['Lx']]
            self.ycrop = [ops['yrange']]
            self.xcrop = [ops['xrange']]
            self.yrange = self.ycrop
            self.xrange = self.xcrop

            if os.path.isfile(ops['reg_file']):
                self.reg_loc = [ops['reg_file']]
            else:
                self.reg_loc = [os.path.abspath(os.path.join(os.path.dirname(filename),'data.bin'))]
            self.reg_file = [open(self.reg_loc[-1],'rb')]
            self.wraw = False
            self.wred = False
            self.wraw_wred = False
            if 'reg_file_raw' in ops or 'raw_file' in ops:
                if self.reg_loc == ops['reg_file']:
                    if 'reg_file_raw' in ops:
                        self.reg_loc_raw = ops['reg_file_raw']
                    else:
                        self.reg_loc_raw = ops['raw_file']
                else:
                    self.reg_loc_raw = os.path.abspath(os.path.join(os.path.dirname(filename),'data_raw.bin'))
                try:
                    self.reg_file_raw = open(self.reg_loc_raw,'rb')
                    self.wraw=True
                except:
                    self.wraw = False
            if 'reg_file_chan2' in ops:
                if self.reg_loc == ops['reg_file']:
                    self.reg_loc_red = ops['reg_file_chan2']
                else:
                    self.reg_loc_red = os.path.abspath(os.path.join(os.path.dirname(filename),'data_chan2.bin'))
                self.reg_file_chan2 = open(self.reg_loc_red,'rb')
                self.wred=True
            if 'reg_file_raw_chan2' in ops or 'raw_file_chan2' in ops:
                if self.reg_loc == ops['reg_file']:
                    if 'reg_file_raw_chan2' in ops:
                        self.reg_loc_raw_chan2 = ops['reg_file_raw_chan2']
                    else:
                        self.reg_loc_raw_chan2 = ops['raw_file_chan2']
                else:
                    self.reg_loc_raw_chan2 = os.path.abspath(os.path.join(os.path.dirname(filename),'data_raw_chan2.bin'))
                try:
                    self.reg_file_raw_chan2 = open(self.reg_loc_raw_chan2,'rb')
                    self.wraw_wred=True
                except:
                    self.wraw_wred = False

            if not fromgui:
                if os.path.isfile(os.path.abspath(os.path.join(os.path.dirname(filename),'F.npy'))):
                    self.Fcell = np.load(os.path.abspath(os.path.join(os.path.dirname(filename),'F.npy')))
                    self.stat =  np.load(os.path.abspath(os.path.join(os.path.dirname(filename),'stat.npy')), allow_pickle=True)
                    self.iscell =  np.load(os.path.abspath(os.path.join(os.path.dirname(filename),'iscell.npy')), allow_pickle=True)
                    self.Floaded = True
                else:
                    self.Floaded = False
            else:
                self.Floaded = True
            good = True
            print(self.Floaded)
            self.filename = filename
        except Exception as e:
            print("ERROR: ops.npy incorrect / missing ops['reg_file'] and others")
            print(e)
            try:
                for n in range(len(self.reg_loc)):
                    self.reg_file[n].close()
                print('closed binaries')
            except:
                print('tried to close binaries')
            good = False
        if good:
            self.filename = filename
            self.ops = [ops]
            self.setup_views()

    def setup_views(self):
        self.p1.clear()
        self.p2.clear()
        self.ichosen = 0
        self.ROIedit.setText('0')
        # get scaling from 100 random frames
        ops = self.ops[-1]
        frames = subsample_frames(ops, np.minimum(ops['nframes']-1,100), self.reg_loc[-1])
        self.srange = frames.mean() + frames.std()*np.array([-2,5])

        self.movieLabel.setText(self.reg_loc[-1])
        self.nbytesread = []
        for n in range(len(self.reg_loc)):
            self.nbytesread.append(2 * self.Ly[n] * self.Lx[n])

        #aspect ratio
        if 'aspect' in ops:
            self.xyrat = ops['aspect']
        elif 'diameter' in ops and (type(ops["diameter"]) is not int) and (len(ops["diameter"]) > 1):
            self.xyrat = ops["diameter"][0] / ops["diameter"][1]
        else:
            self.xyrat = 1.0
        self.vmain.setAspectLocked(lock=True, ratio=self.xyrat)
        self.vside.setAspectLocked(lock=True, ratio=self.xyrat)

        self.nframes = ops['nframes']
        self.time_step = 1. / ops['fs'] * 1000 / 5 # 5x real-time
        self.frameDelta = int(np.maximum(5,self.nframes/200))
        self.frameSlider.setSingleStep(self.frameDelta)
        self.currentMovieDirectory = QtCore.QFileInfo(self.filename).path()
        if self.nframes > 0:
            self.updateFrameSlider()
            self.updateButtons()
        # plot ops X-Y offsets
        if 'yoff' in ops:
            self.yoff = ops['yoff']
            self.xoff = ops['xoff']
        else:
            self.yoff = np.zeros((ops['nframes'],))
            self.xoff = np.zeros((ops['nframes'],))
        self.p1.plot(self.yoff, pen='g')
        self.p1.plot(self.xoff, pen='y')
        self.p1.setRange(xRange=(0,self.nframes),
                         yRange=(np.minimum(self.yoff.min(),self.xoff.min()),
                                 np.maximum(self.yoff.max(),self.xoff.max())),
                         padding=0.0)
        self.p1.setLimits(xMin=0,xMax=self.nframes)
        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        self.scatter1.setData([self.cframe,self.cframe],
                              [self.yoff[self.cframe],self.xoff[self.cframe]],
                              size=10,brush=pg.mkBrush(255,0,0))

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
            if event.modifiers() !=  QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Left:
                    self.cframe -= self.frameDelta
                    self.cframe  = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
                    self.frameSlider.setValue(self.cframe)
                elif event.key() == QtCore.Qt.Key_Right:
                    self.cframe += self.frameDelta
                    self.cframe  = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
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
            self.cell_mask()
            self.ROIedit.setText(str(self.ichosen))
            rgb = np.array(self.colors[self.ichosen])
            self.cellscatter.setData(self.xext, self.yext,
                                     pen=pg.mkPen(list(rgb)),
                                     brush=pg.mkBrush(list(rgb)), size=3)
            self.cellscatter_side.setData(self.xext, self.yext,
                                     pen=pg.mkPen(list(rgb)),
                                     brush=pg.mkBrush(list(rgb)), size=3)

            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.cell_mask()
            self.ft = self.Fcell[self.ichosen,:]
            self.plot_trace()
            self.p2.setXLink('plot_shift')
            self.jump_to_frame()
            self.show()

    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        posx  = 0
        posy  = 0
        iplot = 0
        zoom = False
        zoomImg = False
        choose = False
        if self.loaded:
            for x in items:
                if x==self.p1:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 1
                elif x==self.p2 and self.Floaded:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 2
                elif x==self.vmain or x==self.vside:
                    if event.button()==1:
                        if event.double():
                            self.zoom_image()
                        else:
                            if self.Floaded:
                                pos = x.mapSceneToView(event.scenePos())
                                posy = int(pos.x())
                                posx = int(pos.y())
                                if posy>=0 and posy<self.LX and posx>=0 and posx<self.LY:
                                    if self.cellpix[posx,posy] > -1:
                                        self.ichosen = self.cellpix[posx,posy]
                                        self.cell_chosen()
                if iplot==1 or iplot==2:
                    if event.button()==1:
                        if event.double():
                            zoom=True
                        else:
                            choose=True
        if zoom:
            self.p1.setRange(xRange=(0,self.nframes))
            self.p2.setRange(xRange=(0,self.nframes))
            self.p3.setRange(xRange=(0,self.nframes))

        if choose:
            if self.playButton.isEnabled():
                self.cframe = np.maximum(0, np.minimum(self.nframes-1, int(np.round(posx))))
                self.frameSlider.setValue(self.cframe)
                #self.jump_to_frame()

    def load_zstack(self):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Open zstack", filter="*.tif"
        )
        self.fname = name[0]
        try:
            self.zstack = imread(self.fname)
            self.zLy, self.zLx = self.zstack.shape[1:]
            self.Zedit.setValidator(QtGui.QIntValidator(0, self.zstack.shape[0]))
            self.zrange = [np.percentile(self.zstack,1), np.percentile(self.zstack,99)]

            self.computeZ.setEnabled(True)
            self.zloaded = True
            self.zbox.setEnabled(True)
            self.zbox.setChecked(True)
            if 'zcorr' in self.ops[0]:
                if self.zstack.shape[0]==self.ops[0]['zcorr'].shape[0]:
                    zcorr = self.ops[0]['zcorr']
                    self.zmax = np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
                    self.plot_zcorr()

        except Exception as e:
            print('ERROR: %s'%e)


    def cell_mask(self):
        #self.cmask = np.zeros((self.Ly,self.Lx,3),np.float32)
        self.yext = self.stat[self.ichosen]['yext']
        self.xext = self.stat[self.ichosen]['xext']
        #self.cmask[self.yext,self.xext,2] = (self.srange[1]-self.srange[0])/2 * np.ones((self.yext.size,),np.float32)

    def go_to_frame(self):
        self.cframe = int(self.frameSlider.value())
        self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.frameSlider.setMaximum(self.nframes-1)
        self.frameSlider.setMinimum(0)
        self.frameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def createButtons(self):
        iconSize = QtCore.QSize(30, 30)
        openButton = QtGui.QPushButton('load ops.npy')
        #openButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        #openButton.setIconSize(iconSize)
        openButton.setToolTip("Open single-plane ops.npy")
        openButton.clicked.connect(self.open)

        openButton2 = QtGui.QPushButton('load ops1.npy')
        #openButton2.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DirOpenIcon))
        #openButton2.setIconSize(iconSize)
        #openButton2.setToolTip("Open multi-plane ops1.npy")
        openButton2.clicked.connect(self.open_combined)

        loadZ = QtGui.QPushButton('load z-stack tiff')
        loadZ.clicked.connect(self.load_zstack)

        self.computeZ = QtGui.QPushButton('compute z position')
        self.computeZ.setEnabled(False)
        self.computeZ.clicked.connect(self.compute_z)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QtGui.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)
        btns.setExclusive(True)

        quitButton = QtGui.QToolButton()
        quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
        quitButton.setIconSize(iconSize)
        quitButton.setToolTip("Quit")
        quitButton.clicked.connect(self.close)

        self.l0.addWidget(openButton,1,0,1,2)
        self.l0.addWidget(openButton2,2,0,1,2)
        self.l0.addWidget(loadZ,3,0,1,2)
        self.l0.addWidget(self.computeZ,4,0,1,2)
        self.l0.addWidget(self.playButton,15,0,1,1)
        self.l0.addWidget(self.pauseButton,15,1,1,1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def jump_to_frame(self):
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
            self.cframe = int(self.cframe)
            # seek to absolute position
            for n in range(len(self.reg_file)):
                self.reg_file[n].seek(self.nbytesread[n] * self.cframe, 0)
            if self.wraw:
                self.reg_file_raw.seek(self.nbytesread[-1] * self.cframe, 0)
            if self.wred:
                self.reg_file_chan2.seek(self.nbytesread[-1] * self.cframe, 0)
            if self.wraw_wred:
                self.reg_file_raw_chan2.seek(self.nbytesread[-1] * self.cframe, 0)
            self.cframe -= 1
            self.next_frame()

    def start(self):
        if self.cframe < self.nframes - 1:
            print('playing')
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(self.time_step)


    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        print('paused')

    def compute_z(self):
        ops, zcorr = registration.compute_zpos(self.zstack, self.ops[0])
        self.zmax = np.argmax(gaussian_filter1d(zcorr.T.copy(), 2, axis=1), axis=1)
        np.save(self.filename, ops)
        self.plot_zcorr()

    def plot_zcorr(self):
        self.p3.clear()
        self.p3.plot(self.zmax, pen='r')
        self.p3.addItem(self.scatter3)
        self.p3.setRange(xRange=(0,self.nframes),
                         yRange=(self.zmax.min(),
                                 self.zmax.max()+3),
                         padding=0.0)
        self.p3.setLimits(xMin=0,xMax=self.nframes)
        self.p3.setXLink('plot_shift')

def subsample_frames(ops, nsamps, reg_loc):
    nFrames = ops['nframes']
    Ly = ops['Ly']
    Lx = ops['Lx']
    frames = np.zeros((nsamps, Ly, Lx), dtype='int16')
    nbytesread = 2 * Ly * Lx
    istart = np.linspace(0, nFrames, 1+nsamps).astype('int64')
    reg_file = open(reg_loc, 'rb')
    for j in range(0,nsamps):
        reg_file.seek(nbytesread * istart[j], 0)
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0)
        buff = []
        frames[j,:,:] = np.reshape(data, (Ly, Lx))
    reg_file.close()
    return frames

class PCViewer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(PCViewer, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1300,800)
        self.setWindowTitle('Metrics for registration')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)

        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win,0,2,13,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p3 = self.win.addPlot(row=0,col=0)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.setMenuEnabled(False)

        self.p0 = self.win.addViewBox(name='plot1',lockAspect=True,row=1,col=0,invertY=True)
        self.p1 = self.win.addViewBox(lockAspect=True,row=1,col=1,invertY=True)
        self.p1.setMenuEnabled(False)
        self.p1.setXLink('plot1')
        self.p1.setYLink('plot1')
        self.p2 = self.win.addViewBox(lockAspect=True,row=1,col=2,invertY=True)
        self.p2.setMenuEnabled(False)
        self.p2.setXLink('plot1')
        self.p2.setYLink('plot1')
        self.img0=pg.ImageItem()
        self.img1=pg.ImageItem()
        self.img2=pg.ImageItem()
        self.p0.addItem(self.img0)
        self.p1.addItem(self.img1)
        self.p2.addItem(self.img2)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.p4 = self.win.addPlot(row=0,col=1,colspan=2)
        self.p4.setMouseEnabled(x=False)
        self.p4.setMenuEnabled(False)

        self.PCedit = QtGui.QLineEdit(self)
        self.PCedit.setText('1')
        self.PCedit.setFixedWidth(40)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.plot_frame)
        self.PCedit.textEdited.connect(self.pause)
        qlabel = QtGui.QLabel('PC: ')
        boldfont = QtGui.QFont("Arial", 14, QtGui.QFont.Bold)
        bigfont = QtGui.QFont("Arial", 14)
        qlabel.setFont(boldfont)
        self.PCedit.setFont(bigfont)
        qlabel.setStyleSheet('color: white;')
        #qlabel.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(QtGui.QLabel(''),1,0,1,1)
        self.l0.addWidget(qlabel,2,0,1,1)
        self.l0.addWidget(self.PCedit,2,1,1,1)
        self.nums = []
        self.titles=[]
        for j in range(3):
            num1 = QtGui.QLabel('')
            num1.setStyleSheet('color: white;')
            self.l0.addWidget(num1,3+j,0,1,2)
            self.nums.append(num1)
            t1 = QtGui.QLabel('')
            t1.setStyleSheet('color: white;')
            self.l0.addWidget(t1,12,4+j*4,1,2)
            self.titles.append(t1)
        self.loaded = False
        self.wraw = False
        self.wred = False
        self.wraw_wred = False
        self.l0.addWidget(QtGui.QLabel(''),7,0,1,1)
        self.l0.setRowStretch(7,1)
        self.cframe = 0
        self.createButtons()
        self.nPCs = 50
        self.PCedit.setValidator(QtGui.QIntValidator(1,self.nPCs))
        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        #self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, 'ops'):
            if parent.ops['save_path'][-8:]!='combined':
                filename = os.path.abspath(os.path.join(parent.basename, 'ops.npy'))
                print(filename)
                self.openFile(filename)

    def createButtons(self):
        iconSize = QtCore.QSize(30, 30)
        openButton = QtGui.QToolButton()
        openButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        openButton.setIconSize(iconSize)
        openButton.setToolTip("Open ops file")
        openButton.clicked.connect(self.open)

        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QtGui.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)
        btns.setExclusive(True)

        self.l0.addWidget(openButton,0,0,1,1)
        self.l0.addWidget(self.playButton,14,12,1,1)
        self.l0.addWidget(self.pauseButton,14,13,1,1)
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
        filename = QtGui.QFileDialog.getOpenFileName(self,
                            "Open single-plane ops.npy file",filter="ops*.npy")
        # load ops in same folder
        if filename:
            print(filename[0])
            self.openFile(filename[0])

    def openFile(self, filename):
        try:
            ops = np.load(filename, allow_pickle=True).item()
            self.PC = ops['regPC']
            self.Ly, self.Lx = self.PC.shape[2:]
            self.DX = ops['regDX']
            if 'tPC' in ops:
                self.tPC = ops['tPC']
            else:
                self.tPC = np.zeros((1,self.PC.shape[1]))
            good = True
        except Exception as e:
            print("ERROR: ops.npy incorrect / missing ops['regPC'] and ops['regDX']")
            print(e)
            good = False
        if good:
            self.loaded=True
            self.nPCs = self.PC.shape[1]
            self.PCedit.setValidator(QtGui.QIntValidator(1,self.nPCs))
            self.plot_frame()
            self.playButton.setEnabled(True)

    def next_frame(self):
        iPC = int(self.PCedit.text()) - 1
        pc1 = self.PC[1,iPC,:,:]
        pc0 = self.PC[0,iPC,:,:]
        if self.cframe==0:
            self.img2.setImage(np.tile(pc0[:,:,np.newaxis],(1,1,3)))
            self.titles[2].setText('top')

        else:
            self.img2.setImage(np.tile(pc1[:,:,np.newaxis],(1,1,3)))
            self.titles[2].setText('bottom')

        self.img2.setLevels([pc0.min(),pc0.max()])
        self.cframe = 1-self.cframe

    def plot_frame(self):
        if self.loaded:
            self.titles[0].setText('difference')
            self.titles[1].setText('merged')
            self.titles[2].setText('top')
            iPC = int(self.PCedit.text()) - 1
            pc1 = self.PC[1,iPC,:,:]
            pc0 = self.PC[0,iPC,:,:]
            self.img0.setImage(np.tile(pc1[:,:,np.newaxis]-pc0[:,:,np.newaxis],(1,1,3)))
            self.img0.setLevels([(pc1-pc0).min(),(pc1-pc0).max()])
            rgb = np.zeros((self.PC.shape[2], self.PC.shape[3],3), np.float32)
            rgb[:,:,0] = (pc1-pc1.min())/(pc1.max()-pc1.min())*255
            rgb[:,:,1] = np.minimum(1, np.maximum(0,(pc0-pc1.min())/(pc1.max()-pc1.min())))*255
            rgb[:,:,2] = (pc1-pc1.min())/(pc1.max()-pc1.min())*255
            self.img1.setImage(rgb)
            if self.cframe==0:
                self.img2.setImage(np.tile(pc0[:,:,np.newaxis],(1,1,3)))
            else:
                self.img2.setImage(np.tile(pc1[:,:,np.newaxis],(1,1,3)))
            self.img2.setLevels([pc0.min(),pc0.max()])
            self.zoom_plot()
            self.p3.clear()
            p = [(200,200,255),(255,100,100),(100,50,200)]
            ptitle = ['rigid','nonrigid','nonrigid max']
            if not hasattr(self,'leg'):
                self.leg = pg.LegendItem((100,60),offset=(350,30))
                self.leg.setParentItem(self.p3)
                drawLeg = True
            else:
                drawLeg = False
            for j in range(3):
                cj = self.p3.plot(np.arange(1,self.nPCs+1),self.DX[:,j],pen=p[j])
                if drawLeg:
                    self.leg.addItem(cj,ptitle[j])
                self.nums[j].setText('%s: %1.3f'%(ptitle[j],self.DX[iPC,j]))
            self.scatter = pg.ScatterPlotItem()
            self.p3.addItem(self.scatter)
            self.scatter.setData([iPC+1,iPC+1,iPC+1],self.DX[iPC,:].tolist(),
                                 size=10,brush=pg.mkBrush(255,255,255))
            self.p3.setLabel('left', 'pixel shift')
            self.p3.setLabel('bottom', 'PC #')

            self.p4.clear()
            self.p4.plot(self.tPC[:,iPC])
            self.p4.setLabel('left', 'magnitude')
            self.p4.setLabel('bottom', 'time')
            self.show()
            self.zoom_plot()

    def zoom_plot(self):
        self.p0.setXRange(0,self.Lx)
        self.p0.setYRange(0,self.Ly)
        self.p1.setXRange(0,self.Lx)
        self.p1.setYRange(0,self.Ly)
        self.p2.setXRange(0,self.Lx)
        self.p2.setYRange(0,self.Ly)


    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        posx  = 0
        posy  = 0
        iplot = 0
        zoom = False
        if self.loaded:
            for x in items:
                if x==self.p0 or x==self.p1 or x==self.p2:
                    if event.button()==1:
                        if event.double():
                            zoom=True
                            self.zoom_plot()



    def keyPressEvent(self, event):
        bid = -1
        if event.modifiers() !=  QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = max(ipc-1, 1)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Right:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = min(ipc+1, self.nPCs)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    # then play
                    self.playButton.setChecked(True)
                    self.start()
                else:
                    self.pause()


def make_masks_and_enable_buttons(parent):
    parent.checkBox.setChecked(True)
    parent.ops_plot['color'] = 0
    parent.ops_plot['view'] = 0
    parent.colors['cols'] = 0
    parent.colors['istat'] = 0

    parent.loadBeh.setEnabled(True)
    parent.saveMat.setEnabled(True)
    parent.saveMerge.setEnabled(True)
    parent.sugMerge.setEnabled(True)
    parent.manual.setEnabled(True)
    parent.bloaded = False
    parent.ROI_remove()
    parent.isROI = False
    parent.setWindowTitle(parent.fname)
    # set bin size to be 0.5s by default
    parent.bin = int(parent.ops["tau"] * parent.ops["fs"] / 2)
    parent.binedit.setText(str(parent.bin))
    if "chan2_thres" not in parent.ops:
        parent.ops["chan2_thres"] = 0.6
    parent.chan2prob = parent.ops["chan2_thres"]
    parent.chan2edit.setText(str(parent.chan2prob))
    # add boundaries to stat for ROI overlays
    ncells = len(parent.stat)
    for n in range(0, ncells):
        ypix = parent.stat[n]["ypix"].flatten()
        xpix = parent.stat[n]["xpix"].flatten()
        iext = utils.boundary(ypix, xpix)
        parent.stat[n]["yext"] = ypix[iext]
        parent.stat[n]["xext"] = xpix[iext]
        ycirc, xcirc = utils.circle(
            parent.stat[n]["med"],
            parent.stat[n]["radius"]
        )
        goodi = (
            (ycirc >= 0)
            & (xcirc >= 0)
            & (ycirc < parent.ops["Ly"])
            & (xcirc < parent.ops["Lx"])
        )
        parent.stat[n]["ycirc"] = ycirc[goodi]
        parent.stat[n]["xcirc"] = xcirc[goodi]
        parent.stat[n]["inmerge"] = 0
    # enable buttons
    enable_views_and_classifier(parent)

    # make views
    views.init_views(parent)
    # make color arrays for various views
    masks.make_colors(parent)

    if parent.iscell.sum() > 0:
        ich = np.nonzero(parent.iscell)[0][0]
    else:
        ich = 0
    parent.ichosen = int(ich)
    parent.imerge = [int(ich)]
    parent.iflip = int(ich)
    parent.ichosen_stats()
    parent.comboBox.setCurrentIndex(2)
    # colorbar
    parent.colormat = masks.draw_colorbar()
    masks.plot_colorbar(parent)
    tic = time.time()
    masks.init_masks(parent)
    print(time.time() - tic)
    M = masks.draw_masks(parent)
    masks.plot_masks(parent, M)
    parent.lcell1.setText("%d" % (ncells - parent.iscell.sum()))
    parent.lcell0.setText("%d" % (parent.iscell.sum()))
    graphics.init_range(parent)
    traces.plot_trace(parent)
    parent.xyrat = 1.0
    if (isinstance(parent.ops['diameter'], list) and
        len(parent.ops['diameter'])>1 and
        parent.ops['aspect']==1.0):
        parent.xyrat = parent.ops["diameter"][0] / parent.ops["diameter"][1]
    else:
        if 'aspect' in parent.ops:
            parent.xyrat = parent.ops['aspect']

    parent.p1.setAspectLocked(lock=True, ratio=parent.xyrat)
    parent.p2.setAspectLocked(lock=True, ratio=parent.xyrat)
    #parent.p2.setXLink(parent.p1)
    #parent.p2.setYLink(parent.p1)
    parent.loaded = True
    parent.mode_change(2)
    parent.show()
    # no classifier loaded
    classgui.activate(parent, False)