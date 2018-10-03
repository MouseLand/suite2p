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
#sys.path.insert(0, '/media/carsen/DATA2/Github/rastermap/rastermap')
from rastermap import mapping
from suite2p import gui,fig

### custom QDialog which allows user to fill in ops and run suite2p!
class VisWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(VisWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1100,900)
        self.setWindowTitle('Visualize data')
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
            self.cells = np.array((parent.iscell==icell).nonzero()).flatten()
        else:
            self.cells = np.array(parent.imerge).flatten()
        # compute spikes
        i = parent.activityMode
        if i==0:
            sp = parent.Fcell[self.cells,:]
        elif i==1:
            sp = parent.Fneu[self.cells,:]
        elif i==2:
            sp = parent.Fcell[self.cells,:] - 0.7*parent.Fneu[self.cells,:]
        else:
            sp = parent.Spks[self.cells,:]
        sp = np.squeeze(sp)
        sp = zscore(sp, axis=1)
        sp -= sp.min()
        sp /= sp.max()
        self.sp = np.maximum(0,np.minimum(1,sp))
        self.tsort = np.arange(0,sp.shape[1]).astype(np.int32)
        # 100 ms bins
        self.bin = int(np.maximum(1, int(parent.ops['fs']/10)))
        # draw axes
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
        #self.p2.setLabel('left', 'neurons')
        self.p2.hideAxis('bottom')
        self.bloaded = parent.bloaded
        self.p3 = self.win.addPlot(title='',row=2,col=0,colspan=2)
        self.avg = self.sp.mean(axis=0)
        self.p3.plot(np.arange(0,self.avg.size),self.avg)
        self.p3.setMouseEnabled(x=False,y=False)
        self.p3.getAxis('left').setTicks([[(0,'')]])
        self.p3.setLabel('bottom', 'time')
        if self.bloaded:
            self.beh = parent.beh
            self.p3.plot(np.arange(0,self.beh.size),self.beh)
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
        self.sl.valueChanged.connect(self.levelchange)
        self.sl.setTracking(False)
        self.sat = 1.0
        self.l0.addWidget(self.sl,0,2,5,1)
        qlabel = gui.VerticalLabel(text='saturation')
        #qlabel.setStyleSheet('color: white;')
        self.l0.addWidget(qlabel,1,3,3,1)
        self.isort = np.arange(0,self.cells.size).astype(np.int32)
        # ROI on main plot
        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI([nt*.25, -1], [nt*.25, nn+1],
                      maxBounds=QtCore.QRectF(-1.,-1.,nt+1,nn+1),
                      pen=redpen)
        self.ROI.handleSize = 10
        self.ROI.handlePen = redpen
        # Add top and right Handles
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
        self.p1.addItem(self.ROI)
        self.ROI.setZValue(10)  # make sure ROI is drawn above image
        self.neural_sorting(2)
        # buttons for computations
        self.PCOn = QtGui.QPushButton('compute PCs')
        self.PCOn.clicked.connect(lambda: self.PC_on(True))
        self.l0.addWidget(self.PCOn,0,0,1,2)
        self.mapOn = QtGui.QPushButton('compute raster map')
        self.mapOn.clicked.connect(lambda: self.map_on(parent))
        self.l0.addWidget(self.mapOn,1,0,1,2)
        self.comboBox = QtGui.QComboBox(self)
        self.l0.addWidget(self.comboBox,2,0,1,2)
        self.l0.addWidget(QtGui.QLabel(''),3,0,1,1)
        #self.l0.addWidget(QtGui.QLabel(''),4,0,1,1)
        self.selectBtn = QtGui.QPushButton('show selected cells in GUI')
        self.selectBtn.clicked.connect(lambda: self.select_cells(parent))
        self.selectBtn.setEnabled(True)
        self.l0.addWidget(self.selectBtn,4,0,1,2)
        self.sortTime = QtGui.QCheckBox('&Time sort')
        self.sortTime.setStyleSheet("color: white;")
        self.sortTime.stateChanged.connect(self.sort_time)
        self.l0.addWidget(self.sortTime,5,0,1,2)
        self.l0.addWidget(QtGui.QLabel(''),6,0,1,1)
        self.l0.setRowStretch(6, 1)
        self.raster = False
        self.win.show()
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()

    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x==self.p1:
                if event.button()==1:
                    if event.double():
                        self.ROI.setPos([-1,-1])
                        self.ROI.setSize([self.sp.shape[1]+1, self.sp.shape[0]+1])

    def keyPressEvent(self, event):
        bid = -1
        move = False
        if event.modifiers() !=  QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            elif event.key() == QtCore.Qt.Key_Left:
                bid=2
            elif event.key() == QtCore.Qt.Key_Right:
                bid=3
            if bid >= 0:
                xrange,yrange = self.ROI_range()
                nn,nt = self.sp.shape
                if bid<2:
                    if yrange.size < nn:
                        # can move
                        move = True
                        if bid==0:
                            yrange = yrange - np.minimum(yrange.min(),nn*0.05)
                        else:
                            yrange = yrange + np.minimum(nn-yrange.max()-1,nn*0.05)
                else:
                    if xrange.size < nt:
                        # can move
                        move = True
                        if bid==2:
                            xrange = xrange - np.minimum(xrange.min()+1,nt*0.05)
                        else:
                            xrange = xrange + np.minimum(nt-xrange.max()-1,nt*0.05)
                if move:
                    self.ROI.setPos([xrange.min()-1, yrange.min()-1])
                    self.ROI.setSize([xrange.size+1, yrange.size+1])
        else:
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            elif event.key() == QtCore.Qt.Key_Left:
                bid=2
            elif event.key() == QtCore.Qt.Key_Right:
                bid=3
            if bid >= 0:
                xrange,yrange = self.ROI_range()
                nn,nt = self.sp.shape
                dy = nn*0.05 / (nn/yrange.size)
                dx = nt*0.05 / (nt/xrange.size)
                if bid==0:
                    if yrange.size > dy:
                        # can move
                        move = True
                        ymax = yrange.size - dy
                        yrange = yrange.min() + np.arange(0,ymax).astype(np.int32)
                elif bid==1:
                    if yrange.size < nn-dy + 1:
                        move = True
                        ymax = yrange.size + dy
                        yrange = yrange.min() + np.arange(0,ymax).astype(np.int32)
                elif bid==2:
                    if xrange.size > dx:
                        # can move
                        move = True
                        xmax = xrange.size - dx
                        xrange = xrange.min() + np.arange(0,xmax).astype(np.int32)
                elif bid==3:
                    if xrange.size < nt-dx + 1:
                        move = True
                        xmax = xrange.size + dx
                        xrange = xrange.min() + np.arange(0,xmax).astype(np.int32)
                if move:
                    self.ROI.setPos([xrange.min()-1, yrange.min()-1])
                    self.ROI.setSize([xrange.size+1, yrange.size+1])

    def ROI_range(self):
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
        return xrange,yrange

    def ROI_position(self):
        xrange,yrange = self.ROI_range()
        self.imgROI.setImage(self.spF[np.ix_(yrange,xrange)])
        # also plot range of 1D variable (if active)
        self.p3.clear()
        avg = self.spF[np.ix_(yrange,xrange)].mean(axis=0)
        avg -= avg.min()
        avg /= avg.max()
        self.p3.plot(xrange,avg,pen=(140,140,140))
        self.p3.setXRange(xrange[0],xrange[-1])
        if self.bloaded:
            self.p3.plot(xrange,self.beh[xrange],pen='w')
        #self.selected = (self.sp.shape[0]-1) - yrange
        self.selected = yrange
        self.p2.setXRange(0,xrange.size)
        self.p2.setYRange(0,yrange.size)
        axy = self.p2.getAxis('left')
        axx = self.p2.getAxis('bottom')
        axy.setTicks([[(0.0,str(yrange[0])),(float(yrange.size),str(yrange[-1]))]])
        self.imgROI.setLevels([0,self.sat])

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
        self.PCedit.returnPressed.connect(self.PCreturn)
        self.compute_svd(self.bin)
        self.comboBox.currentIndexChanged.connect(self.neural_sorting)
        if plot:
            self.neural_sorting(0)
        self.PCOn.setEnabled(False)

    def PCreturn(self):
        self.comboBox.setCurrentIndex(0)
        self.neural_sorting(0)

    def map_on(self, parent):
        if not hasattr(self,'u'):
            self.PC_on(False)
        self.comboBox.addItem('raster map')
        tic = time.time()
        self.compute_map(parent)
        print('raster map computed in %3.2f s'%(time.time()-tic))
        self.comboBox.setCurrentIndex(1)
        self.comboBox.currentIndexChanged.connect(self.neural_sorting)
        self.neural_sorting(1)
        self.PCOn.setEnabled(False)
        self.mapOn.setEnabled(False)
        self.sortTime.setChecked(False)

    def compute_map(self, parent):
        self.isort1, self.isort2 = mapping.main(self.sp,None,self.u,self.sv,self.v)
        self.raster = True
        ncells = len(parent.stat)
        # cells not in sorting are set to -1
        parent.isort = -1*np.ones((ncells,),dtype=np.int64)
        nsel = len(self.cells)
        I = np.zeros(nsel)
        I[self.isort1] = np.arange(nsel).astype('int')
        parent.isort[self.cells] = I #self.isort1
        # set up colors for rastermap
        fig.rastermap_masks(parent)
        b = len(parent.colors)+1
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)

    def compute_svd(self,bin):
        sp = self.sp[:,:int(np.floor(self.sp.shape[1]/bin)*bin)]
        spbin = sp.reshape((sp.shape[0],bin,int(sp.shape[1]/bin)))
        spbin = np.squeeze(spbin.mean(axis=1))
        tic=time.time()
        nmin = min([spbin.shape[0],spbin.shape[1]])
        nmin = np.minimum(nmin-1, 200)
        spbin -= spbin.mean(axis=1)[:,np.newaxis]
        self.sv,self.u = eigsh(np.dot(spbin,spbin.T), k=nmin)
        # these eigenvectors are in the opposite order
        self.sv = self.sv[::-1]
        self.u  = self.u[:,::-1]
        self.sv = self.sv ** 0.5
        self.v = (self.sp-self.sp.mean(axis=1)[:,np.newaxis]).T @ self.u
        print('svd computed in %3.2f s'%(time.time()-tic))


    def select_cells(self,parent):
        parent.imerge = []
        if self.selected.size < 5000:
            for n in self.selected:
                parent.imerge.append(self.cells[self.isort[n]])
            parent.ichosen = parent.imerge[0]
            parent.ichosen_stats()
            M = fig.draw_masks(parent)
            fig.plot_masks(parent,M)
            fig.plot_trace(parent)
            parent.show()
        else:
            print('too many cells selected')

    def sort_time(self):
        if self.raster:
            if self.sortTime.isChecked():
                self.tsort = self.isort2
            else:
                self.tsort = np.arange(0,self.sp.shape[1]).astype(np.int32)
            self.neural_sorting(self.comboBox.currentIndex())

    def neural_sorting(self,i):
        if i==0:
            self.isort = np.argsort(self.u[:,int(self.PCedit.text())-1])
        elif i==1:
            self.isort = self.isort1
        if i<2:
            self.spF = gaussian_filter1d(self.sp[np.ix_(self.isort,self.tsort)].T,
                                        np.minimum(8,np.maximum(1,int(self.sp.shape[0]*0.005))),
                                        axis=1)
            self.spF = self.spF.T
        else:
            self.spF = self.sp
        self.spF = zscore(self.spF, axis=1)
        self.spF = np.minimum(8, self.spF)
        self.spF = np.maximum(0, self.spF)
        self.spF /= 8
        self.img.setImage(self.spF)
        self.ROI_position()
