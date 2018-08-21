from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
import time
import sys
#sys.path.insert(0, '/media/carsen/DATA2/Github/dynamap/dynamap/')
sys.path.insert(0, 'C:/Users/carse/github/embeddings/dynamap/')
import map
from suite2p import gui


### custom QDialog which allows user to fill in ops and run suite2p!
class VisWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(VisWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1100,900)
        self.setWindowTitle('Visualize deconvolved data')
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
        self.l0.addWidget(self.win,0,0,14,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(row=0,col=0)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.p1 = self.win.addPlot(title="FULL VIEW",row=0,col=1)
        self.p1.setMouseEnabled(x=False,y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self.p1.addItem(self.img)
        # cells to plot
        if len(parent.imerge)==1:
            icell = parent.iscell[parent.imerge[0]]
            cells = (parent.iscell==icell).nonzero()
        else:
            cells = np.array(parent.imerge)
        # compute spikes
        sp = parent.Spks[cells,:]
        sp = np.squeeze(sp)
        sp = zscore(sp, axis=1)
        sp -= sp.min()
        sp /= sp.max()
        sp = np.maximum(0, np.minimum(1, sp))
        self.sp = np.maximum(0,np.minimum(1,sp))
        self.spF = self.sp
        # 100 ms bins
        self.bin = int(np.maximum(1, int(parent.ops['fs']/10)))
        # draw axes
        self.img.setImage(self.sp)
        self.p1.setXRange(0,sp.shape[1])
        self.p1.setYRange(0,sp.shape[0])
        self.p1.setLimits(xMin=-10,xMax=sp.shape[1]+10,yMin=-10,yMax=sp.shape[0]+10)
        self.p1.setLabel('left', 'neurons')
        self.p1.setLabel('bottom', 'time')
        # zoom in on a selected image region
        nt = sp.shape[1]
        nn = sp.shape[0]
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.setLabel('left', 'neurons')
        self.p2.setLabel('bottom', 'time')
        # set colormap to viridis
        colormap = cm.get_cmap("viridis")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        self.img.setLevels([0,1])
        self.imgROI.setLevels([0,1])
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,3)
        # add slider for levels
        self.sl = QtGui.QSlider(QtCore.Qt.Vertical)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(100)
        self.sl.setTickPosition(QtGui.QSlider.TicksLeft)
        self.sl.setTickInterval(10)
        self.sl.sliderReleased.connect(self.levelchange)
        self.sat = 1.0
        self.l0.addWidget(self.sl,0,2,5,1)
        qlabel = gui.VerticalLabel(text='saturation')
        #qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,1,3,3,1)
        # ROI on main plot
        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI([nt*.25, -1], [nt*.25, nn+1],
                      maxBounds=QtCore.QRectF(-1.,-1.,nt+1,nn+1),
                      pen=redpen)
        self.ROI.handleSize = 9
        self.ROI.handlePen = redpen
        # Add top and right Handles
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
        self.p1.addItem(self.ROI)
        self.ROI.setZValue(10)  # make sure ROI is drawn above image
        self.ROI_position()
        # buttons for computations
        self.PCOn = QtGui.QPushButton('compute PCs')
        self.PCOn.clicked.connect(lambda: self.PC_on(True))
        self.l0.addWidget(self.PCOn,0,0,1,2)
        self.mapOn = QtGui.QPushButton('compute activity maps')
        self.mapOn.clicked.connect(self.map_on)
        self.l0.addWidget(self.mapOn,1,0,1,2)
        self.comboBox = QtGui.QComboBox(self)
        self.l0.addWidget(self.comboBox,2,0,1,2)
        self.l0.addWidget(QtGui.QLabel(''),3,0,1,1)
        self.l0.addWidget(QtGui.QLabel(''),4,0,1,1)
        self.l0.addWidget(QtGui.QLabel(''),5,0,1,1)
        self.l0.addWidget(QtGui.QLabel(''),5,0,1,1)
        self.l0.setRowStretch(6, 1)
        self.win.show()
        self.show()

    def levelchange(self):
        self.sat = float(self.sl.value())/100
        self.img.setLevels([0,self.sat])
        self.imgROI.setLevels([0,self.sat])

    def PC_on(self, plot):
        # edit buttons
        self.PCedit = QtGui.QLineEdit(self)
        self.PCedit.setValidator(QtGui.QIntValidator(1,np.minimum(self.sp.shape[0],self.sp.shape[1])))
        self.PCedit.setText('1')
        self.PCedit.setFixedWidth(60)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        qlabel = QtGui.QLabel('PC: ')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,3,0,1,1)
        self.l0.addWidget(self.PCedit,3,1,1,1)
        self.comboBox.addItem("PC")
        self.PCedit.returnPressed.connect(lambda: self.neural_sorting(0))
        self.compute_svd(self.bin)
        self.comboBox.currentIndexChanged.connect(self.neural_sorting)
        if plot:
            self.neural_sorting(0)
        self.PCOn.setEnabled(False)

    def map_on(self):
        if not hasattr(self,'u'):
            self.PC_on(False)
        self.comboBox.addItem('activity map')
        tic = time.time()
        self.compute_map()
        print('activity map computed in %3.2f s'%(time.time()-tic))
        self.comboBox.setCurrentIndex(1)
        self.comboBox.currentIndexChanged.connect(self.neural_sorting)
        self.neural_sorting(1)
        self.PCOn.setEnabled(False)
        self.mapOn.setEnabled(False)

    def compute_map(self):
        self.isort1, self.isort2 = map.main(self.sp,None,self.u,self.sv,self.v)

    def compute_svd(self,bin):
        sp = self.sp[:,:int(np.floor(self.sp.shape[1]/bin)*bin)]
        spbin = sp.reshape((sp.shape[0],bin,int(sp.shape[1]/bin)))
        spbin = np.squeeze(spbin.mean(axis=1))
        tic=time.time()
        self.sv,self.u = eigsh(spbin @ spbin.T, k=200)
        self.sv = self.sv ** 0.5
        self.v = (self.sp).T @ self.u
        print('svd computed in %3.2f s'%(time.time()-tic))

    def ROI_position(self):
        pos = self.ROI.pos()
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = self.ROI.size()
        xrange = (np.arange(1,int(sizex)) + int(posx)).astype(np.int32)
        yrange = (np.arange(1,int(sizey)) + int(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        xrange = xrange[xrange<self.sp.shape[1]]
        yrange = yrange[yrange>=0]
        yrange = yrange[yrange<self.sp.shape[0]]
        self.imgROI.setImage(self.spF[np.ix_(yrange,xrange)])
        self.imgROI.setRect(QtCore.QRectF(xrange[0],yrange[0],
                                           xrange[-1]-xrange[0],yrange[-1]-yrange[0]))
        self.imgROI.setLevels([0,self.sat])

    def neural_sorting(self,i):
        if i==0:
            isort = np.argsort(self.u[:,int(self.PCedit.text())-1])
        elif i==1:
            isort = self.isort1
        self.spF = gaussian_filter1d(self.sp[isort,:].T,
                                     self.sp.shape[0]*0.005,
                                     axis=1)
        self.spF = self.spF.T
        self.spF -= self.spF.min()
        self.spF /= self.spF.max()
        self.img.setImage(self.spF)
        self.ROI_position()
