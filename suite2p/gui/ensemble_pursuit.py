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
import sys
sys.path.append("..")
from EnsemblePursuit.EnsemblePursuitModule.EnsemblePursuit import EnsemblePursuit

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

        self.all_v = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        #self.p2.addItem(self.imgROI)
        self.all_v.setMouseEnabled(x=True,y=True)
        #self.p2.setLabel('left', 'neurons')

        self.epOn = QtGui.QPushButton('compute EnsemblePursuit')
        self.epOn.clicked.connect(lambda: self.compute_ep(parent))
        self.l0.addWidget(self.epOn,0,0,1,2)

        self.selectBtn = QtGui.QPushButton('show selected cells in GUI')
        self.selectBtn.clicked.connect(lambda: self.select_cells(parent))
        self.selectBtn.setEnabled(True)
        self.l0.addWidget(self.selectBtn,1,0,1,2)

    def compute_ep(self, parent):
        ops = {'n_components': 25, 'lam': 0.01}
        self.n_components=ops['n_components']
        self.error=False
        self.finish=True
        self.epOn.setEnabled(False)
        self.tic=time.time()
        #try:
        self.model = EnsemblePursuit(n_components=ops['n_components'],lam=ops['lam'])
        self.U,self.V=self.model.fit(self.sp)
            #proc  = {'embedding': model.embedding, 'uv': [model.u, model.v],
            #         'ops': ops, 'filename': args.S, 'train_time': train_time}
            #basename, fname = os.path.split(args.S)
            #np.save(os.path.join(basename, 'embedding.npy'), proc)
        print('ep computed in %3.2f s'%(time.time()-self.tic))
            #self.activate(parent)
        self.plot_v()
        #except:
            #print('ep issue: Interrupted by error (not finished)\n')
        #self.process.start('python -u -W ignore -m rastermap --S %s --ops %s'%
        #                    (spath, opspath))
        print(self.V,self.V.shape)

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
