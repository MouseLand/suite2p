# heavily modified script from a pyqt4 release

from PyQt5 import QtGui, QtCore
from suite2p import register,fig
import pyqtgraph as pg
import os
import sys
import numpy as np

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
        self.l0.addWidget(self.win,1,1,13,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True)
        #self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)
        self.p1 = self.win.addPlot(name='plot1',row=1,col=0)
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        #self.p1.autoRange(padding=0.01)
        self.p2 = self.win.addPlot(name='plot2',row=2,col=0)
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No ops chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self.createButtons()
        # create ROI chooser
        self.l0.addWidget(QtGui.QLabel(''),2,0,1,2)
        qlabel = QtGui.QLabel(self)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel,3,0,1,2)
        self.ROIedit = QtGui.QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0,10000))
        self.ROIedit.setText('0')
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, 4,0,1,1)
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
        self.wraw = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, 'ops'):
            if parent.ops['save_path'][-8:]!='combined':
                fileName = os.path.join(parent.basename, 'ops.npy')
                print(fileName)
                self.Fcell = parent.Fcell
                self.stat = parent.stat
                self.Floaded = True
                self.openFile(fileName, True)

    def open(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                            "Open single-plane ops.npy file",filter="ops.npy")
        # load ops in same folder
        if fileName:
            print(fileName[0])
            self.openFile(fileName[0], False)

    def openFile(self, fileName, fromgui):
        try:
            ops = np.load(fileName)
            ops = ops.item()
            self.Ly = ops['Ly']
            self.Lx = ops['Lx']
            self.reg_loc = ops['reg_file']
            self.reg_file = open(self.reg_loc,'rb')
            self.reg_file.close()
            if not fromgui:
                if os.path.isfile(os.path.join(os.path.dirname(fileName),'F.npy')):
                    self.Fcell = np.load(os.path.join(os.path.dirname(fileName),'F.npy'))
                    self.stat =  np.load(os.path.join(os.path.dirname(fileName),'stat.npy'))
                    self.Floaded = True
                else:
                    self.Floaded = False
            if self.Floaded:
                ncells = len(self.stat)
                for n in range(0,ncells):
                    ypix = self.stat[n]['ypix'].flatten()
                    xpix = self.stat[n]['xpix'].flatten()
                    iext = fig.boundary(ypix,xpix)
                    yext = ypix[iext]
                    xext = xpix[iext]
                    #yext = np.hstack((yext,yext+1,yext+1,yext-1,yext-1))
                    #xext = np.hstack((xext,xext+1,xext-1,xext+1,xext-1))
                    goodi = (yext>=0) & (xext>=0) & (yext<self.Ly) & (xext<self.Lx)
                    self.stat[n]['yext'] = yext[goodi]
                    self.stat[n]['xext'] = xext[goodi]
            good = True
        except Exception as e:
            print("ERROR: ops.npy incorrect / missing ops['reg_file'] and others")
            print(e)
            good = False
        if good:
            self.p1.clear()
            self.p2.clear()
            self.ichosen = 0
            self.ROIedit.setText('0')
            # get scaling from 100 random frames
            frames = register.subsample_frames(ops, np.minimum(ops['nframes'],100))
            self.srange = frames.mean() + frames.std()*np.array([-3,3])
            #self.srange = [np.percentile(frames.flatten(),8), np.percentile(frames.flatten(),99)]
            self.reg_file = open(self.reg_loc,'rb')
            self.wraw = False
            if 'reg_file_raw' in ops:
                self.reg_loc_raw = ops['reg_file_raw']
                self.reg_file_raw = open(self.reg_loc_raw,'rb')
                self.wraw=True
            self.movieLabel.setText(self.reg_loc)
            self.nbytesread = 2 * self.Ly * self.Lx
            self.nframes = ops['nframes']
            self.frameDelta = int(np.maximum(5,self.nframes/200))
            self.frameSlider.setSingleStep(self.frameDelta)
            self.currentMovieDirectory = QtCore.QFileInfo(fileName).path()
            if self.nframes > 0:
                self.updateFrameSlider()
                self.updateButtons()
            # plot ops X-Y offsets
            self.yoff = ops['yoff']
            self.xoff = ops['xoff']
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
            if self.Floaded:
                self.cell_mask()
                self.ft = self.Fcell[self.ichosen,:]
                self.p2.plot(self.ft, pen='b')
                self.p2.setRange(xRange=(0,self.nframes),
                                 yRange=(self.ft.min(),self.ft.max()),
                                 padding=0.0)
                self.p2.setLimits(xMin=0,xMax=self.nframes)
                self.scatter2 = pg.ScatterPlotItem()
                self.p2.addItem(self.scatter2)
                self.scatter2.setData([self.cframe],[self.ft[self.cframe]],size=10,brush=pg.mkBrush(255,0,0))
                self.p2.setXLink('plot1')
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
        if self.Floaded:
            self.ichosen = int(self.ROIedit.text())
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.cell_mask()
            self.p2.clear()
            self.ft = self.Fcell[self.ichosen,:]
            self.p2.plot(self.ft,pen='b')
            self.p2.setRange(yRange=(self.ft.min(),self.ft.max()))
            self.scatter2 = pg.ScatterPlotItem()
            self.p2.addItem(self.scatter2)
            self.scatter2.setData([self.cframe],[self.ft[self.cframe]],size=10,brush=pg.mkBrush(255,0,0))
            self.p2.setXLink('plot1')
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
                elif x==self.p0:
                    if event.button()==1:
                        if event.double():
                            zoomImg=True
                if iplot==1 or iplot==2:
                    if event.button()==1:
                        if event.double():
                            zoom=True
                        else:
                            choose=True
        if zoomImg:
            if not self.wraw:
                self.p0.setRange(xRange=(0,self.Lx),yRange=(0,self.Ly))
            else:
                self.p0.setRange(xRange=(0,self.Lx*2+max(10,int(self.Lx*.05))),yRange=(0,self.Ly))
        if zoom:
            self.p1.setRange(xRange=(0,self.nframes))
        if choose:
            if self.playButton.isEnabled():
                self.cframe = np.maximum(0, np.minimum(self.nframes-1, int(np.round(posx))))
                self.frameSlider.setValue(self.cframe)
                #self.jump_to_frame()

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

    def jump_to_frame(self):
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
            # seek to absolute position
            self.reg_file.seek(self.nbytesread * self.cframe, 0)
            if self.wraw:
                self.reg_file_raw.seek(self.nbytesread * self.cframe, 0)
            self.cframe -= 1
            self.next_frame()

    def next_frame(self):
        # loop after video finishes
        self.cframe+=1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
            self.reg_file.seek(0, 0)
            if self.wraw:
                self.reg_file_raw.seek(0, 0)
        buff = self.reg_file.read(self.nbytesread)
        self.img = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly,self.Lx))[:,:,np.newaxis]
        self.img = np.tile(self.img,(1,1,3))
        if self.Floaded:
            self.img[self.yext,self.xext,0] = self.srange[0]
            self.img[self.yext,self.xext,1] = self.srange[0]
            self.img[self.yext,self.xext,2] = (self.srange[1]) * np.ones((self.yext.size,),np.float32)
        if self.wraw:
            buff = self.reg_file_raw.read(self.nbytesread)
            imgraw = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0),(self.Ly,self.Lx))[:,:,np.newaxis]
            imgraw = np.tile(imgraw,(1,1,3))
            blk = self.srange[0]*np.ones((imgraw.shape[0],max(10,int(imgraw.shape[1]*0.05)),3),dtype=np.int16)
            self.img = np.concatenate((imgraw,blk,self.img),axis=1)
        self.pimg.setImage(self.img)
        self.pimg.setLevels(self.srange)
        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        self.scatter1.setData([self.cframe,self.cframe],
                              [self.yoff[self.cframe],self.xoff[self.cframe]],
                              size=10,brush=pg.mkBrush(255,0,0))
        if self.Floaded:
            self.scatter2.setData([self.cframe],[self.ft[self.cframe]],size=10,brush=pg.mkBrush(255,0,0))
        #else:
        #    self.pauseButton.setChecked(True)
        #    self.pause()

    def start(self):
        if self.cframe < self.nframes - 1:
            print('playing')
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(25)


    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        print('paused')
