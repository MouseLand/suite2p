from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from scipy.stats import zscore
import math
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm

#img = pyqtgraph.ImageItem()
#p.addItem(img)
# Get the colormap

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
        self.l0.addWidget(self.win,0,0,10,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(row=0,col=0)
        self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.p1 = self.win.addPlot(title="FULL VIEW",row=0,col=1)
        self.p1.setMouseEnabled(x=False,y=False)
        self.img = pg.ImageItem()
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
        bin = int(parent.ops['fs']) # one second bins
        sp = sp[:,:int(np.floor(sp.shape[1]/bin)*bin)]
        spbin = sp.reshape((sp.shape[0],bin,int(sp.shape[1]/bin)))
        spbin = np.squeeze(spbin.mean(axis=1))
        usv = np.linalg.svd(spbin)
        self.u = usv[0]
        sp -= sp.min()
        sp /= sp.max()
        sp *= 10
        sp = np.maximum(0, np.minimum(1, sp))
        self.sp = 1-sp
        #sp = np.tile(np.expand_dims(sp,axis=2),(1,1,3))
        #self.sp = gaussian_filter1d(1-sp, bin/2, axis=1)
        self.p1.setXRange(0,sp.shape[1])
        self.p1.setYRange(0,sp.shape[0])
        self.p1.setLimits(xMin=-10,xMax=sp.shape[1]+10,yMin=-10,yMax=sp.shape[0]+10)
        self.p1.setLabel('left', 'neurons')
        self.p1.setLabel('bottom', 'time')
        # zoom in on a selected image region
        nt = sp.shape[1]
        nn = sp.shape[0]
        self.p2 = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem()
        colormap = cm.get_cmap("viridis")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3,:]
        # Apply the colormap
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        self.img.setLevels([0,1])
        self.imgROI.setLevels([0,1])
        self.p2.addItem(self.imgROI)
        self.p2.setMouseEnabled(x=False,y=False)
        self.p2.setLabel('left', 'neurons')
        self.p2.setLabel('bottom', 'time')
        layout.setColumnStretchFactor(1,3)
        layout.setRowStretchFactor(1,3)
        # edit buttons
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItem("PC")
        self.comboBox.activated[str].connect(self.sorting)
        self.l0.addWidget(self.comboBox,0,0,1,2)
        self.PCedit = QtGui.QLineEdit(self)
        self.PCedit.setValidator(QtGui.QIntValidator(1,10000))
        self.PCedit.setText('1')
        self.PCedit.setFixedWidth(35)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.sorting)
        qlabel = QtGui.QLabel('PC: ')
        qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,1,0,1,1)
        self.l0.addWidget(self.PCedit,1,1,1,1)
        self.sorting()
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
        self.win.show()

    def ROI_position(self):
        pos = self.ROI.pos()
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = self.ROI.size()
        xrange = (np.arange(1,int(sizex)) + int(posx)).astype(np.int32)
        yrange = (np.arange(1,int(sizey)) + int(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        xrange = xrange[xrange<self.spF.shape[1]]
        yrange = yrange[yrange>=0]
        yrange = yrange[yrange<self.spF.shape[0]]
        self.imgROI.setImage(self.spF[np.ix_(yrange,xrange)])
        self.imgROI.setRect(QtCore.QRectF(xrange[0],yrange[0],
                                           xrange[-1]-xrange[0],yrange[-1]-yrange[0]))

    def sorting(self):
        isort = np.argsort(self.u[:,int(self.PCedit.text())-1])
        self.spF = gaussian_filter1d(self.sp[isort,:], 3, axis=0)
        self.spF = np.maximum(0,np.minimum(1,self.spF))
        self.img.setImage(self.spF)
