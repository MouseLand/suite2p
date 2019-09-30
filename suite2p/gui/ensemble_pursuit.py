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

        self.sp=parent.Fbin

        self.all_v = self.win.addPlot(title='ZOOM IN',row=1,col=0,colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        #self.p2.addItem(self.imgROI)
        self.all_v.setMouseEnabled(x=False,y=False)
        #self.p2.setLabel('left', 'neurons')
        self.all_v.hideAxis('bottom')


    def compute_ep(self, parent):
        ops = {'n_components': 3, 'lambda': 0.01}
        options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
        ep_np=EnsemblePursuitNumpyFast(n_ensembles=1,lambd=0.01,options_dict=options_dict)
        self.error=False
        self.finish=True
        self.mapOn.setEnabled(False)
        self.tic=time.time()
        try:
            self.model = EnsemblePursuitNumpyFast(n_components=ops['n_components'],lambd=ops['lambda'],options_dict=options_dict )
            self.U,self.V=self.model.fit_transform_kmeans_init(self.sp)
            #proc  = {'embedding': model.embedding, 'uv': [model.u, model.v],
            #         'ops': ops, 'filename': args.S, 'train_time': train_time}
            #basename, fname = os.path.split(args.S)
            #np.save(os.path.join(basename, 'embedding.npy'), proc)
            print('ep computed in %3.2f s'%(time.time()-self.tic))
            self.activate(parent)
        except:
            print('ep issue: Interrupted by error (not finished)\n')
        #self.process.start('python -u -W ignore -m rastermap --S %s --ops %s'%
        #                    (spath, opspath))
        print(self.V,self.V.shape)
