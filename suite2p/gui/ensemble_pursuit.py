from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.linalg import eigsh
from scipy.stats import zscore
from matplotlib.colors import hsv_to_rgb
from matplotlib import cm
import time
import sys,os
from rastermap.mapping import Rastermap
from . import rungui,masks
from EnsemblePursuit.EnsemblePursuit import EnsemblePursuit
from PyQt5.QtWidgets import QGridLayout, QWidget, QLabel, QPushButton
import matplotlib.pyplot as plt
from pylab import *

class Color(QLabel):

    def __init__(self, color, ix,*args, **kwargs):
        super(Color, self).__init__(*args, **kwargs)
        self.setAutoFillBackground(True)

        '''
        pltMap = plt.get_cmap('bwr')
        colors = pltMap.colors
        colors = [c + [1.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        pgMap = pg.ColorMap(positions, colors)
        '''
        self.ix=ix
        palette = self.palette()
        colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        print('LUT',lut)
        print('LUT shp', lut.shape)
        # Get the colormap
        #colormap = cm.get_cmap("nipy_spectral")  # cm.get_cmap("CMRmap")
        #colormap._init()
        lut = (colormap._lut).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Apply the colormap
        #self.setLookupTable(lut)
        color=matplotlib.colors.rgb2hex(lut[100])
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(color))

        self.setPalette(palette)

    def mousePressEvent(self, event):
        print('ix',self.ix)
        print("test 1")
        QtGui.QWidget.mousePressEvent(self, event)


class EPWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(EPWindow, self).__init__(parent)
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

        self.sp=parent.Fbin.T

        self.type_of_plot='boxes'

        if self.type_of_plot=='U':
            self.sample_u= self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
            self.imgROI = pg.ImageItem(autoDownsample=True)
            #self.p2.addItem(self.imgROI)
            self.sample_u.setMouseEnabled(x=True,y=True)
            #self.p2.setLabel('left', 'neurons')

        if self.type_of_plot=='V':
            self.all_v = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
            self.imgROI = pg.ImageItem(autoDownsample=True)
            #self.p2.addItem(self.imgROI)
            self.all_v.setMouseEnabled(x=True,y=True)
            #self.p2.setLabel('left', 'neurons')

        if self.type_of_plot=='boxes':
            self.box_layout = pg.LayoutWidget()
            self.l0.addWidget(self.box_layout,4,4,10,10)

        self.epOn = QtGui.QPushButton('compute EnsemblePursuit')
        self.epOn.clicked.connect(lambda: self.compute_ep(parent))
        self.l0.addWidget(self.epOn,0,0,1,2)

        self.selectBtn = QtGui.QPushButton('show selected cells in GUI')
        self.selectBtn.clicked.connect(lambda: self.select_cells(parent))
        self.selectBtn.setEnabled(True)
        self.l0.addWidget(self.selectBtn,1,0,1,2)
        #ROI
        nt,nn=self.sp.shape
        redpen = pg.mkPen(pg.mkColor(255, 0, 0),
                                width=3,
                                style=QtCore.Qt.SolidLine)
        n_ensembles=25
        print('spshape',self.sp.shape)
        self.ROI = pg.RectROI([0,0], [nt, 10],
                      maxBounds=QtCore.QRectF(-1.,-1.,nt+1,nn+1),
                      pen=redpen)
        self.ROI.handleSize = 10
        self.ROI.handlePen = redpen
        # Add top and right Handles
        #self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        #self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        #self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
        #self.p1.addItem(self.ROI)

    def compute_ep(self, parent):
        ops = {'n_components': 25, 'lam': 0.01}
        self.n_components=ops['n_components']
        self.error=False
        self.finish=True
        self.epOn.setEnabled(False)
        self.tic=time.time()
        #try:
        self.model = EnsemblePursuit(n_components=ops['n_components'],lam=ops['lam']).fit(self.sp)
        self.U=self.model.weights
        self.V=self.model.components_
            #proc  = {'embedding': model.embedding, 'uv': [model.u, model.v],
            #         'ops': ops, 'filename': args.S, 'train_time': train_time}
            #basename, fname = os.path.split(args.S)
            #np.save(os.path.join(basename, 'embedding.npy'), proc)
        print('ep computed in %3.2f s'%(time.time()-self.tic))
            #self.activate(parent)
        if self.type_of_plot=='boxes':
            self.plot_boxes()

        if self.type_of_plot=='V':
            self.plot_v()
        if self.type_of_plot=='U':
            self.plot_sample_cells()
        #except:
            #print('ep issue: Interrupted by error (not finished)\n')
        #self.process.start('python -u -W ignore -m rastermap --S %s --ops %s'%
        #                    (spath, opspath))
        print(self.V,self.V.shape)
    def plot_boxes(self):
        w_1=Color('red',2)
        self.box_layout.addWidget(w_1,0,0)
        w_1.setText('Haha')
        self.box_layout.addWidget(Color('green',1), 1, 0)
        self.box_layout.addWidget(Color('blue',3), 1, 1)
        self.box_layout.addWidget(Color('purple',4), 2, 1)
        self.win.show()
        #widget=QWidget()
        #self.win.addLayout(self.box_layout,row=1,col=0,colspan=2)
        #self.setCentralWidget(widget)

    def plot_sample_cells(self):
        cells_in_u=np.nonzero(self.U[:,:25])[0].flatten()[:250]
        cell_ts=self.sp[:,cells_in_u].T
        self.cells = pg.ImageItem(autoDownsample=True)
        self.sample_u.addItem(self.cells)
        self.cells.setImage(cell_ts)
        self.sample_u.addItem(self.ROI)
        print('ts',cell_ts.shape)

    def plot_v(self):
        #self.all_v.plot(self.V.T)
        self.all_v.setYRange(0, self.V.shape[1], padding=0)
        self.linep=[]
        for j in range(0,self.V.shape[1]):
            self.linep.append(self.all_v.plot(self.V.T[j,:]+j, clickable=True))
        self.selected_ensemble=0
        self.linep[0]=self.all_v.plot(self.V.T[0,:], clickable=True,pen='b')
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()
        self.win.show()

    def plot_clicked(self,event):
        vb=self.all_v.vb
        pos = vb.mapSceneToView(event.scenePos()).y()
        ranges=np.arange(0,self.V.shape[1]+1)
        print(ranges)
        print('pos',pos)
        for j in range(0,ranges.shape[0]-1):
            if ranges[j]<=pos<=ranges[j+1]:
                print(j)
                self.linep[j]=self.all_v.plot(self.V.T[j,:]+j, clickable=True,pen='b')
                self.selected_ensemble=j
            else:
                self.linep.append(self.all_v.plot(self.V.T[j,:]+j, clickable=True))
        self.show()
        self.win.show()

    def select_cells(self,parent):
        print(self.U.shape)
        print(self.U[:,self.selected_ensemble])
        self.selected=np.nonzero(self.U[:,self.selected_ensemble])[0]
        parent.imerge = []
        if self.selected.size < 5000:
            for n in self.selected:
                parent.imerge.append(n)
            parent.ichosen = parent.imerge[0]
            parent.update_plot()
        else:
            print('too many cells selected')

    def keyPressEvent(self, event):
        if event.modifiers() !=  QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Down:
                bid = 0
            elif event.key() == QtCore.Qt.Key_Up:
                bid=1
            if bid >= 0:
                move = True
                if bid==1:
                    pos = self.ROI.pos()
                    posy = pos.y()
                    posx=pos.x()
                if move:
                    self.ROI.setPos([posx, posy+10])
