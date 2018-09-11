# heavily modified script from a pyqt4 release

from PyQt5 import QtGui, QtCore
from suite2p import register
import pyqtgraph as pg
import os
import sys
import numpy as np

class BinaryPlayer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(BinaryPlayer, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1100,900)
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
        self.l0.addWidget(self.win,1,1,13,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)
        self.p1 = self.win.addPlot(row=1,col=0)
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        #self.p1.autoRange(padding=0.01)
        self.p2 = self.win.addPlot(row=2,col=0)
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No ops chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        #self.movieLabel.setSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Ignored)
        self.nframes = 0
        self.cframe = 0
        self.createButtons()
        # create frame slider
        self.frameLabel = QtGui.QLabel("Current frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.frameNumber = QtGui.QLabel("0")
        self.frameNumber.setStyleSheet("color: white;")
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        #self.frameSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.frameSlider.setTickInterval(10)
        self.l0.addWidget(QtGui.QLabel(''),12,0,1,1)
        self.l0.setRowStretch(12,1)
        self.l0.addWidget(self.frameLabel, 13,0,1,2)
        self.l0.addWidget(self.frameNumber, 14,0,1,2)
        self.l0.addWidget(self.frameSlider, 13,2,14,13)
        self.l0.addWidget(QtGui.QLabel(''),14,1,1,1)
        #speedLabel = QtGui.QLabel("Speed:")
        #self.speedSpinBox = QtGui.QSpinBox()
        #self.speedSpinBox.setRange(1, 9999)
        #self.speedSpinBox.setValue(100)
        #self.speedSpinBox.setSuffix("%")
        #self.controlsLayout.addWidget(speedLabel, 2, 0)
        #self.controlsLayout.addWidget(self.speedSpinBox, 2, 1)

        #self.movie.frameChanged.connect(self.updateFrameSlider)
        #self.movie.stateChanged.connect(self.updateButtons)
        #self.fitCheckBox.clicked.connect(self.fitToWindow)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        #self.speedSpinBox.valueChanged.connect(self.movie.setSpeed)
        self.l0.addWidget(self.movieLabel,0,0,1,5)
        self.updateFrameSlider()
        self.updateButtons()
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        # if not a combined recording, automatically open binary
        if hasattr(parent, 'ops'):
            if parent.ops['save_path'][-8:]!='combined':
                fileName = os.path.join(parent.basename, 'ops.npy')
                print(fileName)
                self.openFile(fileName)

    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                            "Open single-plane ops.npy file",filter="ops.npy")
        # load ops in same folder
        if fileName:
            print(fileName[0])
            self.openFile(fileName[0])

    def openFile(self, fileName):
        try:
            ops = np.load(fileName)
            ops = ops.item()
            self.Ly = ops['Ly']
            self.Lx = ops['Lx']
            self.reg_loc = ops['reg_file']
            self.reg_file = open(self.reg_loc,'rb')
            self.reg_file.close()
            good = True
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: ops.npy incorrect / missing ops['reg_file']")
            good = False
        if good:
            self.p1.clear()
            self.p2.clear()
            # get scaling from 100 random frames
            frames = register.subsample_frames(ops, np.minimum(ops['nframes'],100))
            self.srange = frames.mean() + frames.std()*np.array([-3,3])
            #self.srange = [np.percentile(frames.flatten(),8), np.percentile(frames.flatten(),99)]
            self.reg_file = open(self.reg_loc,'rb')
            self.movieLabel.setText(self.reg_loc)
            self.cframe = 0
            self.nbytesread = 2 * self.Ly * self.Lx
            self.currentMovieDirectory = QtCore.QFileInfo(fileName).path()
            buff = self.reg_file.read(self.nbytesread)
            self.img = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly,self.Lx))
            self.pimg.setImage(self.img)
            self.pimg.setLevels(self.srange)
            self.nframes = ops['nframes']
            if self.nframes > 0:
                self.updateFrameSlider()
                self.updateButtons()
            # plot ops X-Y offsets
            self.yoff = ops['yoff']
            self.xoff = ops['xoff']
            self.p1.plot(self.yoff)
            self.p1.plot(self.xoff)
            self.p1.setRange(xRange=(0,self.nframes),
                             yRange=(self.yoff.min(),self.yoff.max()),
                             padding=0.0)
            self.scatter1 = pg.ScatterPlotItem()
            self.p1.addItem(self.scatter1)
            self.p2.setRange(xRange=(0,self.nframes),
                             yRange=(self.xoff.min(),self.xoff.max()),
                             padding=0.0)
            #self.scatter2 = pg.ScatterPlotItem()
            #self.p2.addItem(self.scatter2)
            self.scatter1.setData([self.cframe,self.cframe],
                                  [self.yoff[self.cframe],self.xoff[self.cframe]],
                                  size=10,brush=pg.mkBrush(255,0,0))
            #self.scatter2.setData([self.cframe],[self.xoff[self.cframe,0]],size=10,brush=pg.mkBrush(255,0,0))

    def go_to_frame(self, frame):
        self.jump_to_frame(frame)

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.frameSlider.setMaximum(self.nframes - 1)
        self.frameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(True)
        self.pauseButton.setChecked(True)

    def createButtons(self):
        iconSize = QtCore.QSize(30, 30)
        openButton = QtGui.QToolButton()
        openButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        openButton.setIconSize(iconSize)
        openButton.setToolTip("Open binary file")
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

        quitButton = QtGui.QToolButton()
        quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
        quitButton.setIconSize(iconSize)
        quitButton.setToolTip("Quit")
        quitButton.clicked.connect(self.close)

        self.l0.addWidget(openButton,1,0,1,1)
        self.l0.addWidget(self.playButton,15,0,1,1)
        self.l0.addWidget(self.pauseButton,15,1,1,1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def jump_to_frame(self,frame):
        if self.playButton.isEnabled():
            self.cframe = self.frameSlider.value()
            self.cframe = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
            # seek to absolute position
            self.reg_file.seek(self.nbytesread * self.cframe, 0)
            self.cframe -= 1
            self.next_frame()

    def next_frame(self):
        if self.cframe < self.nframes - 1:
            buff = self.reg_file.read(self.nbytesread)
            self.img = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly,self.Lx))
            self.pimg.setImage(self.img)
            self.pimg.setLevels(self.srange)
            self.cframe+=1
            self.frameSlider.setValue(self.cframe)
            self.frameNumber.setText(str(self.cframe))
            self.scatter1.setData([self.cframe,self.cframe],
                                  [self.yoff[self.cframe],self.xoff[self.cframe]],
                                  size=10,brush=pg.mkBrush(255,0,0))

    def start(self):
        if self.cframe < self.nframes - 1:
            print('playing')
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.updateTimer.start(25)

    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        print('paused')
