from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os
import pickle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import fig
import gui
import classifier
import time
class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        self.setGeometry(25,25,1600,1000)
        self.setWindowTitle('suite2p')
        #self.setStyleSheet("QMainWindow {background: 'black';}")
        self.loaded = False
        self.ops_plot = []
        # default plot options
        self.ops_plot.append(True)
        self.ops_plot.append(0)
        self.ops_plot.append(0)
        ### menu bar options
        # run suite2p from scratch
        runS2P =  QtGui.QAction('&Run suite2p ', self)
        runS2P.setShortcut('Ctrl+R')
        runS2P.triggered.connect(self.run_suite2p)
        self.addAction(runS2P)
        # load processed data
        loadProc = QtGui.QAction('&Load processed data (choose stat.npy file)', self)
        loadProc.setShortcut('Ctrl+L')
        loadProc.triggered.connect(self.load_dialog)
        self.addAction(loadProc)
        # load masks
        #loadMask = QtGui.QAction('&Load masks (stat.pkl) and extract traces', self)
        #loadMask.setShortcut('Ctrl+M')
        #self.addAction(loadMask)
        # make mainmenu!
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(runS2P)
        file_menu.addAction(loadProc)
        #file_menu.addAction(loadMask)
        # classifier menu
        self.trainfiles = []
        self.statlabels = None
        self.statclass = ['skew', 'compact', 'footprint']
        self.loadClass = QtGui.QAction('&Load classifier', self)
        self.loadClass.setShortcut('Ctrl+K')
        self.loadClass.triggered.connect(self.load_classifier)
        self.loadClass.setEnabled(False)
        self.loadTrain = QtGui.QAction('&Train classifier (choose iscell.npy files)', self)
        self.loadTrain.setShortcut('Ctrl+T')
        self.loadTrain.triggered.connect(self.load_traindata)
        self.loadTrain.setEnabled(False)
        self.saveClass = QtGui.QAction('&Save classifier', self)
        self.saveClass.setShortcut('Ctrl+S')
        self.saveClass.triggered.connect(self.save_classifier)
        self.saveClass.setEnabled(False)
        self.saveTrain = QtGui.QAction('&Save training list', self)
        #self.saveTrain.setShortcut('Ctrl+S')
        self.saveTrain.triggered.connect(self.save_trainlist)
        self.saveTrain.setEnabled(False)
        class_menu = main_menu.addMenu('&Classifier')
        class_menu.addAction(self.loadClass)
        class_menu.addAction(self.loadTrain)
        class_menu.addAction(self.saveClass)
        class_menu.addAction(self.saveTrain)

        # main widget
        cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        # ROI CHECKBOX
        checkBox = QtGui.QCheckBox('ROIs on')
        checkBox.move(30,100)
        checkBox.stateChanged.connect(self.ROIs_on)
        checkBox.toggle()
        self.l0.addWidget(checkBox,0,0,1,1)
        # MAIN PLOTTING AREA
        self.win = pg.GraphicsView()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,1,18,12)
        l = pg.GraphicsLayout(border=(100,100,100))
        self.win.setCentralItem(l)
        self.p0 = l.addLabel('load a stat.pkl file',row=0,col=0,colspan=2)
        # cells image
        self.p1 = l.addViewBox(lockAspect=True,name='plot1',row=1,col=0)
        self.img1 = pg.ImageItem()
        self.p1.setMenuEnabled(False)
        data = np.zeros((700,512,3))
        self.img1.setImage(data)
        self.p1.addItem(self.img1)
        #self.p1.setXRange(0,512,padding=0.25)
        #self.p1.setYRange(0,512,padding=0.25)
        # noncells image
        self.p2 = l.addViewBox(lockAspect=True,name='plot2',row=1,col=1)
        self.p2.setMenuEnabled(False)
        self.img2 = pg.ImageItem()
        self.img2.setImage(data)
        self.p2.addItem(self.img2)
        #self.p2.autoRange()
        self.p2.setXLink('plot1')
        self.p2.setYLink('plot1')
        # fluorescence trace plot
        self.p3 = l.addPlot(row=2,col=0,colspan=2)
        #x = np.arange(0,20000)
        #y = np.zeros((20000,))
        #self.p3.clear()
        #self.p3.plot(x,y,pen='b')
        self.p3.setMouseEnabled(x=True,y=False)
        self.p3.enableAutoRange(x=True,y=True)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.show()
        self.win.show()
        #
        #self.load_proc(['/media/carsen/DATA2/Github/data/stat.pkl','*'])
        self.fname = 'C:/Users/carse/github/data/stat.npy'
        self.load_proc()

    def make_masks_and_buttons(self, name):
        self.p0.setText(name)
        views = ['mean img', 'correlation map']
        # add boundaries to stat for ROI overlays
        ncells = self.Fcell.shape[0]
        for n in range(0,ncells):
            ypix = self.stat[n]['ypix']
            xpix = self.stat[n]['xpix']
            iext = np.expand_dims(fig.boundary(ypix,xpix),axis=0)
            self.stat[n]['yext'] = ypix[iext]
            self.stat[n]['xext'] = xpix[iext]

        if 'mean_image_red' in self.ops:
            views.append('red channel mean')
        colors = ['random', 'skew', 'compact','footprint',
                    'aspect_ratio']
        b = 0
        self.viewbtns = QtGui.QButtonGroup(self)
        vlabel = QtGui.QLabel(self)
        vlabel.setText('Background')
        vlabel.resize(vlabel.minimumSizeHint())
        self.l0.addWidget(vlabel,1,0,1,1)
        self.btnstate = []
        for names in views:
            btn  = gui.ViewButton(b,names,self)
            self.viewbtns.addButton(btn,b)
            self.l0.addWidget(btn,b+2,0,1,1)
            self.btnstate.append(False)
            b+=1
        self.colorbtns = QtGui.QButtonGroup(self)
        clabel = QtGui.QLabel(self)
        clabel.setText('Colors')
        clabel.resize(clabel.minimumSizeHint())
        self.l0.addWidget(clabel,b+2,0,1,1)
        nv = b+2
        b=0
        allcols = np.random.random((ncells,1))
        self.clabels = []
        # colorbars for different statistics
        self.colorfig = plt.figure(figsize=(1,0.05))
        self.canvas = FigureCanvas(self.colorfig)
        self.colorbar = self.colorfig.add_subplot(211)
        for names in colors:
            if names in self.stat[0] or b==0:
                if b > 0:
                    istat = np.zeros((ncells,1))
                    for n in range(0,ncells):
                        istat[n] = self.stat[n][names]
                    self.clabels.append([istat.min(), (istat.max()-istat.min())/2, istat.max()])
                    istat = istat - istat.min()
                    istat = istat / istat.max()
                    istat = istat / 1.3
                    istat = istat + 0.1
                    icols = 1 - istat
                    allcols = np.concatenate((allcols, icols), axis=1)
                else:
                    self.clabels.append([0,0.5,1])
                btn  = gui.ColorButton(b,names,self)
                self.colorbtns.addButton(btn,b)
                self.l0.addWidget(btn,nv+b+1,0,1,1)
                self.btnstate.append(False)
                if b==0:
                    btn.setChecked(True)
                b+=1
        self.classbtn  = gui.ColorButton(b,'classifier',self)
        self.colorbtns.addButton(self.classbtn,b)
        self.ncolors = b+1
        self.classbtn.setEnabled(False)
        self.l0.addWidget(self.classbtn,nv+b+1,0,1,1)
        self.btnstate.append(False)
        self.ops_plot.append(allcols)
        self.iROI = fig.ROI_index(self.ops, self.stat)
        self.ichosen = int(0)
        self.iflip = int(0)
        if not hasattr(self, 'iscell'):
            self.iscell = np.ones((ncells,), dtype=bool)
        fig.init_masks(self)
        M = fig.draw_masks(self)
        self.plot_masks(M)
        self.l0.addWidget(self.canvas,nv+b+2,0,1,1)
        self.colormat = fig.make_colorbar()
        self.plot_colorbar(0)
        #gl = pg.GradientLegend((10,300),(10,30))
        #gl.setParentItem(self.p1)
        self.p1.setXRange(0,self.ops['Ly'])
        self.p1.setYRange(0,self.ops['Lx'])
        self.p2.setXRange(0,self.ops['Ly'])
        self.p2.setYRange(0,self.ops['Lx'])
        self.p3.setLimits(xMin=0,xMax=self.Fcell.shape[1])
        self.trange = np.arange(0, self.Fcell.shape[1])
        self.plot_trace()
        self.loadClass.setEnabled(True)
        self.loadTrain.setEnabled(True)
        self.show()

    def plot_colorbar(self, bid):
        self.colorbar.clear()
        if bid==0:
            self.colorbar.imshow(np.zeros((20,100,3)))
        else:
            self.colorbar.imshow(self.colormat)
        self.colorbar.tick_params(axis='y',which='both',left=False,right=False,
                                labelleft=False,labelright=False)
        self.colorbar.set_xticks([0,50,100])
        self.colorbar.set_xticklabels(['%1.2f'%self.clabels[bid][0],
                                        '%1.2f'%self.clabels[bid][1],
                                        '%1.2f'%self.clabels[bid][2]],
                                        fontsize=6)
        self.canvas.draw()

    def plot_trace(self):
        self.p3.clear()
        self.p3.plot(self.trange,self.Fcell[self.ichosen,:],pen='b')
        self.p3.plot(self.trange,self.Fneu[self.ichosen,:],pen='r')
        self.fmax = np.maximum(self.Fcell[self.ichosen,:].max(), self.Fneu[self.ichosen,:].max())
        self.fmin = np.minimum(self.Fcell[self.ichosen,:].min(), self.Fneu[self.ichosen,:].min())
        self.p3.setXRange(0,self.Fcell.shape[1])
        self.p3.setYRange(self.fmin,self.fmax)

    def ROIs_on(self,state):
        if state == QtCore.Qt.Checked:
            self.ops_plot[0] = True
        else:
            self.ops_plot[0] = False
        if self.loaded:
            M = fig.draw_masks(self)
            self.plot_masks(M)

    def plot_masks(self,M):
        self.img1.setImage(M[0])
        self.img2.setImage(M[1])
        self.img1.show()
        self.img2.show()

    def plot_clicked(self,event):
        '''left-click chooses a cell, right-click flips cell to other view'''
        flip = False
        choose = False
        zoom = False
        items = self.win.scene().items(event.scenePos())
        posx  = 0
        posy  = 0
        iplot = 0
        if self.loaded:
            for x in items:
                if x==self.img1:
                    pos = self.p1.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    posy = pos.y()
                    iplot = 1
                elif x==self.img2:
                    pos = self.p2.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    posy = pos.y()
                    iplot = 2
                elif x==self.p3:
                    iplot = 3
                if iplot > 0 and iplot < 3:
                    if event.button()==2:
                        flip = True
                        choose = True
                    elif event.button()==1:
                        if event.double():
                            zoom = True
                        else:
                            choose = True
                if iplot==3 and event.double():
                    zoom = True
                posy = int(posy)
                posx = int(posx)
                if zoom:
                    if iplot==1:
                        self.p1.setXRange(0,self.ops['Ly'])
                        self.p1.setYRange(0,self.ops['Lx'])
                    elif iplot==2:
                        self.p2.setXRange(0,self.ops['Ly'])
                        self.p2.setYRange(0,self.ops['Lx'])
                    else:
                        self.p3.setXRange(0,self.Fcell.shape[1])
                        self.p3.setYRange(self.fmin,self.fmax)
                    self.show()
                if choose:
                    ichosen = int(self.iROI[posx,posy])
                    if self.ichosen == ichosen:
                        choose = False
                    elif ichosen >= 0:
                        self.ichosen = ichosen
                if flip and self.iflip != self.ichosen:
                    flip = True
                    iscell = int(self.iscell[self.ichosen])
                    if 2-iscell == iplot:
                        self.iscell[self.ichosen] = ~self.iscell[self.ichosen]
                        np.save(self.basename+'/iscell.npy', self.iscell)
                    self.iflip = self.ichosen
                    fig.flip_cell(self)
                else:
                    flip = False
                if choose or flip:
                    t0=time.time()
                    M = fig.draw_masks(self)
                    self.plot_masks(M)
                    self.plot_trace()
                    self.show()
                    print(time.time()-t0)

    def run_suite2p(self):
        RW = gui.RunWindow(self)
        RW.show()

    def load_dialog(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.fname = name[0]
        self.load_proc()

    def load_proc(self):
        name = self.fname
        print(name)
        try:
            self.stat = np.load(name)
            self.stat = self.stat.item()
            ypix = self.stat[0]['ypix']
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print('ERROR: this is not a stat.npy file :( (needs stat[n]["ypix"]!)')
            self.stat = None
        if self.stat is not None:
            basename, fname = os.path.split(name)
            self.basename = basename
            goodfolder = True
            try:
                self.Fcell = np.load(basename + '/F.npy')
                self.Fneu = np.load(basename + '/Fneu.npy')
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('ERROR: there are no fluorescence traces in this folder (F.npy/Fneu.npy)')
                goodfolder = False
            try:
                self.Spks = np.load(basename + '/spks.npy')
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('there are no spike deconvolved traces in this folder (spks.npy)')
            try:
                self.iscell = np.load(basename + '/iscell.npy')
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('no manual labels found (iscell.npy)')
            try:
                self.ops = np.load(basename + '/ops.npy')
                self.ops = self.ops.item()
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('ERROR: there is no ops file in this folder (ops.npy)')
                goodfolder = False
            if goodfolder:
                self.make_masks_and_buttons(name[0])
                self.loaded = True
            else:
                print('stat.npy found, but other files not in folder')
                Text = 'stat.npy found, but other files missing, choose another?'
                self.load_again(Text)
        else:
            Text = 'Incorrect file, not a stat.npy, choose another?'
            self.load_again(Text)

    def load_again(self,Text):
        tryagain = QtGui.QMessageBox.question(self, 'ERROR',
                                        Text,
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if tryagain == QtGui.QMessageBox.Yes:
            self.load_proc()

    def load_classifier(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if name:
            self.model = classifier.Classifier(classfile=name[0],
                                               trainfiles=None,
                                               statclass=None)
            if self.model.loaded:
                # statistics from current dataset for Classifier
                checkstat = [key for key in self.statclass if key in self.stat[0]]
                if len(checkstat) == len(self.statclass):
                    ncells = self.Fcell.shape[0]
                    self.statistics = np.zeros((ncells, len(self.statclass)),np.float32)
                    k=0
                    for key in self.statclass:
                        for n in range(0,ncells):
                            self.statistics[n,k] = self.stat[n][key]
                        k+=1
                    self.trainfiles = self.model.trainfiles
                    self.activate_classifier()
                else:
                    print('ERROR: classifier has fields that stat doesn''t have')

    def load_traindata(self):
        # will return
        self.traindata = np.zeros((0,len(self.statclass)+1),np.float32)
        LC = gui.ListChooser('classifier training files', self)
        result = LC.exec_()
        if result:
            print('Populating classifier:')
            self.model = classifier.Classifier(classfile=None,
                                               trainfiles=self.trainfiles,
                                               statclass=self.statclass)
            if self.trainfiles is not None:
                ncells = self.Fcell.shape[0]
                self.statistics = np.zeros((ncells, len(self.statclass)),np.float32)
                k=0
                for key in self.statclass:
                    for n in range(0,ncells):
                        self.statistics[n,k] = self.stat[n][key]
                    k+=1
                self.activate_classifier()

    def apply_classifier(self):
        classval = self.probedit.value()
        self.iscell, self.probcell = self.model.apply(self.statistics, classval)

    def save_classifier(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Save classifier')
        if name:
            try:
                self.model.save_classifier(name[0])
            except (OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
                print('ERROR: incorrect filename for saving')

    def save_trainlist(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Save list of iscell.npy')
        if name:
            try:
                with open(name[0],'w') as fid:
                    for f in self.trainfiles:
                        fid.write(f)
                        fid.write('\n')
            except (ValueError, OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
                print('ERROR: incorrect filename for saving')

    def activate_classifier(self):
        iscell, self.probcell = self.model.apply(self.statistics, 0.5)
        istat = self.probcell
        if len(self.clabels) < self.ncolors:
            self.clabels.append([istat.min(), (istat.max()-istat.min())/2, istat.max()])
        else:
            self.clabels[-1] = [istat.min(), (istat.max()-istat.min())/2, istat.max()]
        istat = istat - istat.min()
        istat = istat / istat.max()
        istat = istat / 1.3
        istat = istat + 0.1
        icols = 1 - istat
        if self.ops_plot[3].shape[1] < self.ncolors:
            self.ops_plot[3] = np.concatenate((self.ops_plot[3],
                                                np.expand_dims(icols,axis=1)), axis=1)
        else:
            self.ops_plot[3][:,-1] = icols
        self.classbtn.setEnabled(True)
        self.saveClass.setEnabled(True)
        self.saveTrain.setEnabled(True)
        applyclass = QtGui.QPushButton('apply classifier')
        applyclass.resize(100,50)
        applyclass.clicked.connect(self.apply_classifier)
        self.l0.addWidget(QtGui.QLabel('\t      cell prob'),13,0,1,1)
        self.probedit = QtGui.QDoubleSpinBox(self)
        self.probedit.setDecimals(3)
        self.probedit.setMaximum(1.0)
        self.probedit.setMinimum(0.0)
        self.probedit.setSingleStep(0.01)
        self.probedit.setValue(0.5)
        #self.probedit.setValidator(QtGui.QDoubleValidator())
        #self.probedit.setText('0.5')
        #self.probedit.setMaxLength(5)
        #qedit.move(10,600)
        self.probedit.setFixedWidth(55)
        self.l0.addWidget(self.probedit,13,0,1,1)
        self.l0.addWidget(applyclass,14,0,1,1)
        addtoclass = QtGui.QPushButton('add current data \n to classifier')
        addtoclass.resize(100,100)
        addtoclass.clicked.connect(self.add_to_classifier)
        self.l0.addWidget(addtoclass,15,0,1,1)
        saveclass = QtGui.QPushButton('save classifier')
        saveclass.resize(100,50)
        saveclass.clicked.connect(self.save_trainlist)
        self.l0.addWidget(saveclass,16,0,1,1)

    def add_to_classifier(self):
        fname = self.basename+'/iscell.npy'
        ftrue =  [f for f in self.trainfiles if fname in f]
        if len(ftrue)==0:
            self.trainfiles.append(self.basename+'/iscell.npy')
        print('Repopulating classifier including current dataset:')
        self.model = classifier.Classifier(classfile=None,
                                           trainfiles=self.trainfiles,
                                           statclass=self.statclass)

def run():
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    GUI = MainW()
    sys.exit(app.exec_())

run()
