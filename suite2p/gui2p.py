from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os
import pickle
from suite2p import fig, gui, classifier
import time
class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder='row-major')

        self.setGeometry(25,25,1600,1000)
        self.setWindowTitle('suite2p (run pipeline or load stat.npy)')
        self.setWindowIcon(QtGui.QIcon('logo.png'))
        #self.setStyleSheet("QMainWindow {background: 'black';}")
        self.loaded = False
        self.ops_plot = []
        # default plot options
        self.ops_plot.append(True)
        self.ops_plot.append(0)
        self.ops_plot.append(0)
        self.ops_plot.append(0)
        #### ------ MENU BAR ----------------- ####
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
        #loadMask = QtGui.QAction('&Load masks (stat.npy) and extract traces', self)
        #loadMask.setShortcut('Ctrl+M')
        #self.addAction(loadMask)
        # make mainmenu!
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(runS2P)
        file_menu.addAction(loadProc)
        # classifier menu
        self.trainfiles = []
        self.statlabels = None
        self.statclass = ['skew','compact','aspect_ratio','footprint']
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
        self.saveTrain.triggered.connect(self.save_trainlist)
        self.saveTrain.setEnabled(False)
        class_menu = main_menu.addMenu('&Classifier')
        class_menu.addAction(self.loadClass)
        class_menu.addAction(self.loadTrain)
        class_menu.addAction(self.saveClass)
        class_menu.addAction(self.saveTrain)

        #### --------- MAIN WIDGET LAYOUT --------- ####
        #pg.setConfigOption('background', 'w')
        cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        # ROI CHECKBOX
        checkBox = QtGui.QCheckBox('ROIs &On')
        checkBox.move(30,100)
        checkBox.stateChanged.connect(self.ROIs_on)
        checkBox.toggle()
        self.l0.addWidget(checkBox,0,0,1,1)
        # number of ROIs in each image
        self.lcell0 = QtGui.QLabel('n ROIs')
        self.l0.addWidget(self.lcell0, 0,2,1,1)
        self.lcell1 = QtGui.QLabel('n ROIs')
        self.l0.addWidget(self.lcell1, 0,8,1,1)
        #### -------- MAIN PLOTTING AREA ---------- ####
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,1,1,34,12)
        layout = self.win.ci.layout
        # --- cells image
        self.p1 = self.win.addViewBox(lockAspect=True,name='plot1',border=[100,100,100],
                                      row=0,col=0, invertY=True)
        self.img1 = pg.ImageItem()
        self.p1.setMenuEnabled(False)
        data = np.zeros((700,512,3))
        self.img1.setImage(data)
        self.p1.addItem(self.img1)
        # --- noncells image
        self.p2 = self.win.addViewBox(lockAspect=True,name='plot2',border=[100,100,100],
                                      row=0,col=1, invertY=True)
        self.p2.setMenuEnabled(False)
        self.img2 = pg.ImageItem()
        self.img2.setImage(data)
        self.p2.addItem(self.img2)
        self.p2.setXLink('plot1')
        self.p2.setYLink('plot1')
        # --- fluorescence trace plot
        self.p3 = self.win.addPlot(row=1,col=0,colspan=2)
        layout.setRowStretchFactor(0,2)
        self.p3.setMouseEnabled(x=True,y=False)
        self.p3.enableAutoRange(x=True,y=True)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.show()
        self.win.show()
        #### --------- VIEW AND COLOR BUTTONS ---------- ####
        self.views = ['Q: ROIs', 'W: mean img\n    (enhanced)', 'E: mean img', 'R: correlation map']
        self.colors = ['random', 'skew', 'compact','footprint','aspect_ratio']
        b = 0
        self.viewbtns = QtGui.QButtonGroup(self)
        vlabel = QtGui.QLabel(self)
        vlabel.setText('Background')
        vlabel.resize(vlabel.minimumSizeHint())
        self.l0.addWidget(vlabel,1,0,1,1)
        for names in self.views:
            btn  = gui.ViewButton(b,'&'+names,self)
            self.viewbtns.addButton(btn,b)
            self.l0.addWidget(btn,b+2,0,1,1)
            btn.setEnabled(False)
            b+=1
        self.viewbtns.setExclusive(True)
        # color buttons
        self.colorbtns = QtGui.QButtonGroup(self)
        clabel = QtGui.QLabel(self)
        clabel.setText('Colors')
        #self.l0.addWidget(QtGui.QLabel(''),b+2,0,1,1)
        #self.l0.setRowStretch(b+2,1)
        self.l0.addWidget(clabel,b+3,0,1,1)
        nv = b+3
        b=0
        # colorbars for different statistics
        for names in self.colors:
            btn  = gui.ColorButton(b,names,self)
            self.colorbtns.addButton(btn,b)
            self.l0.addWidget(btn,nv+b+1,0,1,1)
            btn.setEnabled(False)
            b+=1
        self.bend = nv+b+3+2
        self.classbtn  = gui.ColorButton(b,'classifier',self)
        self.colorbtns.addButton(self.classbtn,b)
        self.ncolors = b+1
        self.classbtn.setEnabled(False)
        self.l0.addWidget(self.classbtn,nv+b+1,0,1,1)
        colorbarW = pg.GraphicsLayoutWidget()
        colorbarW.setBackground(background=[255,255,255])
        colorbarW.setMaximumHeight(60)
        colorbarW.setMaximumWidth(150)
        colorbarW.ci.layout.setRowStretchFactor(0,2)
        colorbarW.ci.layout.setContentsMargins(0,0,0,0)
        self.l0.addWidget(colorbarW, nv+b+2,0,1,1)
        #self.l0.addWidget(QtGui.QLabel(''),nv+b+3,0,1,1)
        #self.l0.setRowStretch(nv+b+3, 1)
        self.colorbar = pg.ImageItem()
        cbar = colorbarW.addViewBox(row=0,col=0,colspan=3)
        cbar.setMenuEnabled(False)
        cbar.addItem(self.colorbar)
        self.clabel = [colorbarW.addLabel('0.0',color=[0,0,0],row=1,col=0),
                        colorbarW.addLabel('0.5',color=[0,0,0],row=1,col=1),
                        colorbarW.addLabel('1.0',color=[0,0,0],row=1,col=2)]
        #### ----- CLASSIFIER BUTTONS ------- ####
        applyclass = QtGui.QPushButton('apply classifier')
        applyclass.resize(100,50)
        applyclass.clicked.connect(self.apply_classifier)
        self.l0.addWidget(QtGui.QLabel('Classifer'),self.bend,0,1,1)
        self.l0.addWidget(QtGui.QLabel('\t      cell prob'),self.bend+1,0,1,1)
        applyclass.setEnabled(False)
        self.probedit = QtGui.QDoubleSpinBox(self)
        self.probedit.setDecimals(3)
        self.probedit.setMaximum(1.0)
        self.probedit.setMinimum(0.0)
        self.probedit.setSingleStep(0.01)
        self.probedit.setValue(0.5)
        self.probedit.setFixedWidth(55)
        self.l0.addWidget(self.probedit,self.bend+1,0,1,1)
        self.l0.addWidget(applyclass,self.bend+2,0,1,1)
        addtoclass = QtGui.QPushButton('add current data \n to classifier')
        addtoclass.resize(100,100)
        addtoclass.clicked.connect(self.add_to_classifier)
        addtoclass.setEnabled(False)
        self.l0.addWidget(addtoclass,self.bend+3,0,1,1)
        saveclass = QtGui.QPushButton('save classifier')
        saveclass.resize(100,50)
        saveclass.clicked.connect(self.save_classifier)
        saveclass.setEnabled(False)
        self.l0.addWidget(saveclass,self.bend+4,0,1,1)
        self.classbtns = QtGui.QButtonGroup(self)
        self.classbtns.addButton(applyclass,0)
        self.classbtns.addButton(addtoclass,1)
        self.classbtns.addButton(saveclass,2)
        #### ------ CELL STATS -------- ####
        #self.l0.setRowStretch(1, 1)
        #self.l0.setRowStretch(6, 1)
        # which stats
        self.stats_to_show = ['med','npix','skew','compact','footprint',
                              'aspect_ratio']
        self.l0.addWidget(QtGui.QLabel('Selected ROI stats'),self.bend+6,0,1,1)
        lilfont = QtGui.QFont("Arial", 8)
        qlabel = QtGui.QLabel('ROI')
        qlabel.setFont(lilfont)
        self.l0.addWidget(qlabel,self.bend+7,0,1,1)
        self.ROIstats = []
        self.ROIstats.append(qlabel)
        for k in range(1,len(self.stats_to_show)+1):
            self.ROIstats.append(QtGui.QLabel(self.stats_to_show[k-1]))
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.setRowStretch(self.bend+8+k, 0)
            self.l0.addWidget(self.ROIstats[k], self.bend+8+k,0,1,1)
        self.l0.addWidget(QtGui.QLabel(''), self.bend+9+k,0,1,1)
        self.l0.setRowStretch(self.bend+9+k, 1)
        self.fname = '/media/carsen/DATA2/Github/data2/stat.npy'
        self.load_proc()


    def make_masks_and_buttons(self):
        self.disable_classifier()
        self.ops_plot[1] = 0
        self.ops_plot[2] = 0
        self.setWindowTitle(self.fname)
        # add boundaries to stat for ROI overlays
        ncells = self.Fcell.shape[0]
        for n in range(0,ncells):
            ypix = self.stat[n]['ypix']
            xpix = self.stat[n]['xpix']
            iext = np.expand_dims(fig.boundary(ypix,xpix),axis=0)
            self.stat[n]['yext'] = ypix[iext]
            self.stat[n]['xext'] = xpix[iext]
            self.stat[n]['yext_overlap'] = np.zeros((0,),np.int32)
            self.stat[n]['xext_overlap'] = np.zeros((0,),np.int32)

        for b in range(len(self.views)):
            self.viewbtns.button(b).setEnabled(True)
            #self.viewbtns.button(b).setShortcut(QtGui.QKeySequence('R'))
            if b==0:
                self.viewbtns.button(b).setChecked(True)
        for b in range(len(self.colors)):
            self.colorbtns.button(b).setEnabled(True)
            if b==0:
                self.colorbtns.button(b).setChecked(True)
        allcols = np.random.random((ncells,1))
        self.clabels = []
        b=0
        for names in self.colors:
            if b > 0:
                istat = np.zeros((ncells,1))
                for n in range(0,ncells):
                    istat[n] = self.stat[n][names]
                self.clabels.append([istat.min(),
                                     (istat.max()-istat.min())/2 + istat.min(),
                                     istat.max()])
                istat = istat - istat.min()
                istat = istat / istat.max()
                istat = istat / 1.3
                istat = istat + 0.1
                icols = 1 - istat
                allcols = np.concatenate((allcols, icols), axis=1)
            else:
                self.clabels.append([0,0.5,1])
            b+=1

        self.ops_plot[3] = (allcols)
        #self.iROI = fig.ROI_index(self.ops, self.stat)
        self.ichosen = int(0)
        self.iflip = int(0)
        self.ichosen_stats()
        if not hasattr(self, 'iscell'):
            self.iscell = np.ones((ncells,), dtype=bool)
        nv=6
        self.colormat = fig.make_colorbar()
        self.plot_colorbar(0)
        fig.init_masks(self)
        M = fig.draw_masks(self)
        self.plot_masks(M)
        self.p1.setXRange(0,self.ops['Lx'])
        self.p1.setYRange(0,self.ops['Ly'])
        self.p2.setXRange(0,self.ops['Lx'])
        self.p2.setYRange(0,self.ops['Ly'])
        self.lcell1.setText('%d cells'%(ncells-self.iscell.sum()))
        self.lcell0.setText('%d cells'%(self.iscell.sum()))
        self.p3.setLimits(xMin=0,xMax=self.Fcell.shape[1])
        self.trange = np.arange(0, self.Fcell.shape[1])
        self.plot_trace()
        self.loadClass.setEnabled(True)
        self.loadTrain.setEnabled(True)
        self.show()

    def plot_colorbar(self, bid):
        if bid==0:
            self.colorbar.setImage(np.zeros((20,100,3)))
        else:
            self.colorbar.setImage(self.colormat)
        for k in range(3):
            self.clabel[k].setText('%1.2f'%self.clabels[bid][k])

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
        self.img1.setImage(M[0],levels=(0.0,1.0))
        self.img2.setImage(M[1],levels=(0.0,1.0))
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
            print(event.modifiers() == QtCore.Qt.ControlModifier)
            for x in items:
                if x==self.img1:
                    pos = self.p1.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 1
                elif x==self.img2:
                    pos = self.p2.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 2
                elif x==self.p3:
                    iplot = 3
                elif (x==self.p1 or x==self.p2) and x!=self.img1 and x!=self.img2:
                    iplot = 4
                    if event.double():
                        zoom = True
                if iplot==1 or iplot==2:
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
                    self.zoom_plot(iplot)
                if (choose or flip) and (iplot==1 or iplot==2):
                    ichosen = int(self.iROI[iplot-1,0,posx,posy])
                    if ichosen<0:
                        choose = False
                        flip = False
                    else:
                        if self.ichosen==ichosen:
                            choose = False
                        if choose:
                            self.ichosen = ichosen
                    if flip:
                        flip = self.flip_plot(iplot)
                    if choose or flip:
                        #tic=time.time()
                        self.ichosen_stats()
                        M = fig.draw_masks(self)
                        self.plot_masks(M)
                        self.plot_trace()
                        self.show()
                        #print(time.time()-tic)


    def ichosen_stats(self):
        n = self.ichosen
        self.ROIstats[0].setText('ROI: '+str(n))
        for k in range(1,len(self.stats_to_show)+1):
            key = self.stats_to_show[k-1]
            ival = self.stat[n][key]
            if k==1:
                self.ROIstats[k].setText(key+': [%d, %d]'%(ival[0],ival[1]))
            elif k==2:
                self.ROIstats[k].setText(key+': %d'%(ival))
            else:
                self.ROIstats[k].setText(key+': %2.2f'%(ival))

    def flip_plot(self,iplot):
        self.iflip = self.ichosen
        iscell = int(self.iscell[self.ichosen])
        if 2-iscell == iplot:
            flip = True
            self.iscell[self.ichosen] = ~self.iscell[self.ichosen]
            np.save(self.basename+'/iscell.npy', self.iscell)
            fig.flip_cell(self)
            self.lcell0.setText('%d ROIs'%(self.iscell.sum()))
            self.lcell1.setText('%d ROIs'%(self.iscell.size-self.iscell.sum()))
        else:
            flip = False
        return flip

    def zoom_plot(self,iplot):
        if iplot==1 or iplot==2 or iplot==4:
            self.p1.setXRange(0,self.ops['Lx'])
            self.p1.setYRange(0,self.ops['Ly'])
        else:
            self.p3.setXRange(0,self.Fcell.shape[1])
            self.p3.setYRange(self.fmin,self.fmax)
        self.show()

    def run_suite2p(self):
        RW = gui.RunWindow(self)
        RW.show()

    def load_dialog(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open stat.npy', filter='stat.npy')
        self.fname = name[0]
        self.load_proc()

    def load_proc(self):
        name = self.fname
        print(name)
        try:
            self.stat = np.load(name)
            #self.stat = self.stat.item()
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
                self.make_masks_and_buttons()
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
            self.load_dialog()

    def load_classifier(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if name:
            self.model = classifier.Classifier(classfile=name[0],
                                               trainfiles=None,
                                               statclass=None)
            if self.model.loaded:
                # statistics from current dataset for Classifier
                self.statclass = self.model.statclass
                # fill up with current dataset stats
                self.get_stats()
                self.trainfiles = self.model.trainfiles
                self.activate_classifier()
                #else:
                #    print('ERROR: classifier has fields that stat doesn''t have')

    def get_stats(self):
        ncells = self.Fcell.shape[0]
        self.statistics = np.zeros((ncells, len(self.statclass)),np.float32)
        k=0
        for key in self.statclass:
            for n in range(0,ncells):
                self.statistics[n,k] = self.stat[n][key]
            k+=1

    def load_traindata(self):
        # will return
        LC = gui.ListChooser('classifier training files', self)
        result = LC.exec_()
        if result:
            print('Populating classifier:')
            self.model = classifier.Classifier(classfile=None,
                                               trainfiles=self.trainfiles,
                                               statclass=self.statclass)
            if self.trainfiles is not None:
                self.get_stats()
                self.activate_classifier()

    def apply_classifier(self):
        classval = self.probedit.value()
        self.iscell, self.probcell = self.model.apply(self.statistics, classval)
        fig.init_masks(self)
        M = fig.draw_masks(self)
        self.plot_masks(M)
        self.lcell0.setText('%d ROIs'%self.iscell.sum())
        self.lcell1.setText('%d ROIs'%(self.iscell.size-self.iscell.sum()))

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
        fig.init_masks(self)
        self.classbtn.setEnabled(True)
        self.saveClass.setEnabled(True)
        self.saveTrain.setEnabled(True)
        for btns in self.classbtns.buttons():
            btns.setEnabled(True)

    def disable_classifier(self):
        self.classbtn.setEnabled(False)
        self.saveClass.setEnabled(False)
        self.saveTrain.setEnabled(False)
        for btns in self.classbtns.buttons():
            btns.setEnabled(False)


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

#run()
