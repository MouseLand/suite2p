from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter1d
from scipy import ndimage
from scipy.stats import zscore
import math
from matplotlib.colors import hsv_to_rgb

### custom QDialog which allows user to fill in ops and run suite2p!
class VisWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(VisWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(50,50,1100,600)
        self.setWindowTitle('Visualize deconvolved data')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItem("PC")
        self.comboBox.addItem("embed")
        self.comboBox.activated[str].connect(self.sorting)
        self.l0.addWidget(self.comboBox,0,0,1,2)
        self.PCedit = QtGui.QLineEdit(self)
        self.PCedit.setValidator(QtGui.QIntValidator(1,10000))
        self.PCedit.setText('1')
        self.PCedit.setFixedWidth(35)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.sorting)
        self.l0.addWidget(QtGui.QLabel('PC: '),1,0,1,1)
        self.l0.addWidget(self.PCedit,1,1,1,1)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,2,10,14)
        layout = self.win.ci.layout
        # --- cells image
        self.p0 = self.win.addViewBox(lockAspect=False,name='vis',border=[100,100,100],
                                      row=0,col=0, invertY=True)
        #self.p0.state['background'] = [1,1,1]
        self.img = pg.ImageItem()
        self.p0.addItem(self.img)
        if len(parent.imerge)==1:
            icell = parent.iscell[parent.imerge[0]]
            roi = (parent.iscell==icell).nonzero()
        else:
            roi = np.array(parent.imerge)
        sp = parent.Spks[roi,:]
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
        sp = np.tile(np.expand_dims(sp,axis=2),(1,1,3))
        self.sp = gaussian_filter1d(1-sp, bin/2, axis=1)
        self.sorting()
        self.p0.setXRange(0,sp.shape[1])
        self.p0.setYRange(0,sp.shape[0])
        self.p0.setLimits(xMin=-100,xMax=sp.shape[1]+100,yMin=-sp.shape[0],yMax=sp.shape[0]*2)
        xAx = pg.AxisItem(orientation='bottom', linkView=self.p0)
        self.p0.addItem(xAx)
        yAx = pg.AxisItem(orientation='left', linkView=self.p0)
        self.p0.addItem(yAx)
        self.p0.show()

    def sorting(self):
        isort = np.argsort(self.u[:,int(self.PCedit.text())-1])
        sp = gaussian_filter1d(self.sp[isort,:], 3, axis=0)
        self.img.setImage(sp)
