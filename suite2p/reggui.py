# heavily modified script from a pyqt4 release

from PyQt5 import QtGui, QtCore
from suite2p import register,fig
import pyqtgraph as pg
import os
import sys
import numpy as np

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
                fileName = os.path.join(parent.basename, 'ops.npy')
                print(fileName)
                self.openFile(fileName)

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
            self.PC = ops['regPC']
            self.DX = ops['regDX']
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
            if os.path.isfile(ops['reg_file']):
                self.reg_loc = ops['reg_file']
                self.reg_file = open(self.reg_loc,'rb')
                self.reg_file.close()
            else:
                self.reg_loc = os.path.join(os.path.dirname(fileName),'data.bin')
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
            frames = subsample_frames(ops, np.minimum(ops['nframes'],100), self.reg_loc)
            self.srange = frames.mean() + frames.std()*np.array([-3,3])
            #self.srange = [np.percentile(frames.flatten(),8), np.percentile(frames.flatten(),99)]
            self.reg_file = open(self.reg_loc,'rb')
            self.wraw = False
            if 'reg_file_raw' in ops:
                if self.reg_loc == ops['reg_file']:
                    self.reg_loc_raw = ops['reg_file_raw']
                else:
                    self.reg_loc_raw = os.path.join(os.path.dirname(fileName),'data_raw.bin')
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
            self.cframe = int(self.cframe)
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
