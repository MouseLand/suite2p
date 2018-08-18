from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os
import pickle
from suite2p import fig, gui, classifier, visualize
import time

#class EventWidget(QtGui.QWidget):
#    def __init__(self,parent=None):
#        super(EventWidget, self, parent).__init__()
#    def

class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(25,25,1600,1000)
        self.setWindowTitle('suite2p (run pipeline or load stat.npy)')
        icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'logo/logo.png')
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16,16))
        app_icon.addFile(icon_path, QtCore.QSize(24,24))
        app_icon.addFile(icon_path, QtCore.QSize(32,32))
        app_icon.addFile(icon_path, QtCore.QSize(48,48))
        app_icon.addFile(icon_path, QtCore.QSize(96,96))
        app_icon.addFile(icon_path, QtCore.QSize(256,256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
        self.styleUnpressed = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
        self.loaded = False
        self.ops_plot = []
        # default plot options
        self.ops_plot.append(True)
        self.ops_plot.append(0)
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
        self.loadClass = QtGui.QAction('Load classifier', self)
        self.loadClass.setShortcut('Ctrl+K')
        self.loadClass.triggered.connect(self.load_classifier)
        self.loadClass.setEnabled(False)
        self.loadTrain = QtGui.QAction('Train classifier (choose iscell.npy files)', self)
        self.loadTrain.setShortcut('Ctrl+T')
        self.loadTrain.triggered.connect(lambda: classifier.load_data(self))
        self.loadTrain.setEnabled(False)
        self.saveClass = QtGui.QAction('&Save classifier', self)
        self.saveClass.setShortcut('Ctrl+S')
        self.saveClass.triggered.connect(lambda: classifier.save(self))
        self.saveClass.setEnabled(False)
        self.saveDefault = QtGui.QAction('Save classifier as default', self)
        self.saveDefault.triggered.connect(self.class_default)
        self.saveDefault.setEnabled(False)
        self.addToClass = QtGui.QAction('add current data \n to classifier',self)
        self.addToClass.triggered.connect(lambda: classifier.add_to(self))
        self.addToClass.setEnabled(False)
        self.saveTrain = QtGui.QAction('Save training list', self)
        self.saveTrain.triggered.connect(lambda: classifier.save_list(self))
        self.saveTrain.setEnabled(False)
        class_menu = main_menu.addMenu('&Classifier')
        class_menu.addAction(self.loadClass)
        class_menu.addAction(self.loadTrain)
        class_menu.addAction(self.saveClass)
        class_menu.addAction(self.saveDefault)
        class_menu.addAction(self.addToClass)
        class_menu.addAction(self.saveTrain)
        # visualizations menuBar
        self.visualizations = QtGui.QAction('&Visualize selected cells', self)
        self.visualizations.triggered.connect(self.vis_window)
        self.visualizations.setEnabled(False)
        vis_menu = main_menu.addMenu('&Visualizations')
        vis_menu.addAction(self.visualizations)
        self.visualizations.setShortcut('Ctrl+V')

        #### --------- MAIN WIDGET LAYOUT --------- ####
        #pg.setConfigOption('background', 'w')
        #cwidget = EventWidget(self)
        cwidget = QtGui.QWidget()
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        # ROI CHECKBOX
        self.checkBox = QtGui.QCheckBox('ROIs &On')
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.stateChanged.connect(self.ROIs_on)
        self.checkBox.toggle()
        self.l0.addWidget(self.checkBox,0,0,1,1)
        # number of ROIs in each image
        self.lcell0 = QtGui.QLabel('')
        self.lcell0.setStyleSheet("color: white;")
        self.lcell0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(self.lcell0, 0,6,1,1)
        self.lcell1 = QtGui.QLabel('')
        self.lcell1.setStyleSheet("color: white;")
        self.l0.addWidget(self.lcell1, 0,10,1,1)
        self.selectbtn = [QtGui.QPushButton(' draw selection'),
                          QtGui.QPushButton(' draw selection')]
        for b in self.selectbtn:
            b.setCheckable(True)
            b.setStyleSheet(self.styleUnpressed)
        self.selectbtn[0].clicked.connect(lambda: self.ROI_selection(0))
        self.selectbtn[1].clicked.connect(lambda: self.ROI_selection(1))
        self.selectbtn[0].setEnabled(False)
        self.selectbtn[1].setEnabled(False)
        self.isROI=False
        self.ROIplot = 0
        self.l0.addWidget(self.selectbtn[0], 0,1,1,1)
        self.l0.addWidget(self.selectbtn[1], 0,15,1,1)
        # minimize view
        self.sizebtns = QtGui.QButtonGroup(self)
        b=0
        labels = ['cells','both','not cells']
        for l in labels:
            btn  = gui.SizeButton(b,l,self)
            self.sizebtns.addButton(btn,b)
            self.l0.addWidget(btn,0,7+b,1,1)
            btn.setEnabled(False)
            if b==1:
                btn.setEnabled(True)
            b+=1
        self.sizebtns.setExclusive(True)
        #### -------- MAIN PLOTTING AREA ---------- ####
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,1,1,38,15)
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
        self.win.ci.layout.setRowStretchFactor(0,2)
        layout.setColumnMinimumWidth(0,1)
        layout.setColumnMinimumWidth(1,1)
        layout.setHorizontalSpacing(20)
        self.p3.setMouseEnabled(x=True,y=False)
        self.p3.enableAutoRange(x=True,y=True)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        #self.key_on(self.win.scene().keyPressEvent)
        self.show()
        self.win.show()
        #### --------- VIEW AND COLOR BUTTONS ---------- ####
        self.views = ['Q: ROIs', 'W: mean img (enhanced)', 'E: mean img', 'R: correlation map']
        self.colors = ['A: random', 'S: skew', 'D: compact','F: footprint','G: aspect_ratio','H: classifier','J: correlations']
        b = 0
        boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        self.viewbtns = QtGui.QButtonGroup(self)
        vlabel = QtGui.QLabel(self)
        vlabel.setText("<font color='white'>Background</font>")
        vlabel.setFont(boldfont)
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
        clabel.setText("<font color='white'>Colors</font>")
        clabel.setFont(boldfont)
        self.l0.addWidget(clabel,b+3,0,1,1)
        nv = b+3
        b=0
        # colorbars for different statistics
        for names in self.colors:
            btn  = gui.ColorButton(b,'&'+names,self)
            self.colorbtns.addButton(btn,b)
            self.l0.addWidget(btn,nv+b+1,0,1,1)
            btn.setEnabled(False)
            self.colors[b] = self.colors[b][3:]
            b+=1
        self.bend = nv+b+4
        colorbarW = pg.GraphicsLayoutWidget()
        colorbarW.setMaximumHeight(60)
        colorbarW.setMaximumWidth(150)
        colorbarW.ci.layout.setRowStretchFactor(0,2)
        colorbarW.ci.layout.setContentsMargins(0,0,0,0)
        self.l0.addWidget(colorbarW, nv+b+2,0,1,1)
        self.colorbar = pg.ImageItem()
        cbar = colorbarW.addViewBox(row=0,col=0,colspan=3)
        cbar.setMenuEnabled(False)
        cbar.addItem(self.colorbar)
        self.clabel = [colorbarW.addLabel('0.0',color=[255,255,255],row=1,col=0),
                        colorbarW.addLabel('0.5',color=[255,255,255],row=1,col=1),
                        colorbarW.addLabel('1.0',color=[255,255,255],row=1,col=2)]
        #### ----- CLASSIFIER BUTTONS ------- ####
        applyclass = QtGui.QPushButton(' apply classifier')
        applyclass.resize(100,50)
        applyclass.clicked.connect(lambda: classifier.apply(self))
        cllabel = QtGui.QLabel("")
        cllabel.setFont(boldfont)
        cllabel.setText("<font color='white'>Classifier</font>")
        self.l0.addWidget(cllabel,self.bend,0,1,1)
        plabel = QtGui.QLabel('\t    cell probability')
        plabel.setStyleSheet("color: white;")
        self.l0.addWidget(plabel,self.bend+1,0,1,1)
        applyclass.setEnabled(False)
        applyclass.setStyleSheet(self.styleUnpressed)
        self.probedit = QtGui.QDoubleSpinBox(self)
        self.probedit.setDecimals(3)
        self.probedit.setMaximum(1.0)
        self.probedit.setMinimum(0.0)
        self.probedit.setSingleStep(0.01)
        self.probedit.setValue(0.5)
        self.probedit.setFixedWidth(55)
        self.l0.addWidget(self.probedit,self.bend+1,0,1,1)
        self.l0.addWidget(applyclass,self.bend+2,0,1,1)
        self.classbtns = QtGui.QButtonGroup(self)
        self.classbtns.addButton(applyclass,0)
        self.l0.addWidget(QtGui.QLabel(''), self.bend+3,0,1,1)
        #### ------ CELL STATS -------- ####
        # which stats
        self.bend = self.bend+4
        self.stats_to_show = ['med','npix','skew','compact','footprint',
                              'aspect_ratio']
        lilfont = QtGui.QFont("Arial", 8)
        qlabel = QtGui.QLabel(self)
        qlabel.setFont(boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel,self.bend,0,1,1)
        self.ROIedit = QtGui.QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0,10000))
        self.ROIedit.setText('0')
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, self.bend+1,0,1,1)
        self.ROIstats = []
        self.ROIstats.append(qlabel)
        for k in range(1,len(self.stats_to_show)+1):
            llabel = QtGui.QLabel(self.stats_to_show[k-1])
            self.ROIstats.append(llabel)
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].setStyleSheet("color: white;")
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.addWidget(self.ROIstats[k], self.bend+2+k,0,1,1)
        self.l0.addWidget(QtGui.QLabel(''), self.bend+3+k,0,1,1)
        self.l0.setRowStretch(self.bend+3+k, 1)
        # up/down arrows to resize view
        self.level = 1
        self.arrowButtons = [QtGui.QPushButton(u"  \u25b2"), QtGui.QPushButton(u"  \u25bc")]
        self.arrowButtons[0].clicked.connect(self.expand_trace)
        self.arrowButtons[1].clicked.connect(self.collapse_trace)
        b = 0
        for btn in self.arrowButtons:
            btn.setMaximumWidth(25)
            btn.setStyleSheet(self.styleUnpressed)
            self.l0.addWidget(btn, self.bend+4+k+b,0,1,1,QtCore.Qt.AlignRight)
            b+=1
        self.l0.addWidget(QtGui.QLabel(''), self.bend+6+k,0,1,1)
        # trace labels
        flabel = QtGui.QLabel(self)
        flabel.setText("<font color='blue'>Fluorescence</font>")
        nlabel = QtGui.QLabel(self)
        nlabel.setText("<font color='red'>Neuropil</font>")
        slabel = QtGui.QLabel(self)
        slabel.setText("<font color='gray'>Deconvolved</font>")
        flabel.setFont(boldfont)
        nlabel.setFont(boldfont)
        slabel.setFont(boldfont)
        flabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        nlabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        slabel.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(flabel, self.bend+7+k,0,1,1)
        self.l0.addWidget(nlabel, self.bend+8+k,0,1,1)
        self.l0.addWidget(slabel, self.bend+9+k,0,1,1)
        # classifier file to load
        self.classfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                         'classifiers/classifier_user.npy')
        #print(self.classfile)
        #self.fname = '/media/carsen/DATA2/Github/data3/stat.npy'
        #self.fname = 'C:/Users/carse/github/tiffs/suite2p/plane0/stat.npy'
        #self.load_proc()

    def keyPressEvent(self, event):
        if event.modifiers() !=  QtCore.Qt.ControlModifier:
            if event.key() == QtCore.Qt.Key_Return:
                merge=1
            elif event.key() == QtCore.Qt.Key_Escape:
                self.zoom_plot(1)
                self.show()
            elif event.key() == QtCore.Qt.Key_Delete:
                self.ROI_remove()
            elif event.key() == QtCore.Qt.Key_Shift:
                split=1
            elif event.key() == QtCore.Qt.Key_Q:
                self.viewbtns.button(0).setChecked(True)
                self.viewbtns.button(0).press(self, 0)
            elif event.key() == QtCore.Qt.Key_W:
                self.viewbtns.button(1).setChecked(True)
                self.viewbtns.button(1).press(self, 1)
            elif event.key() == QtCore.Qt.Key_E:
                self.viewbtns.button(2).setChecked(True)
                self.viewbtns.button(2).press(self, 2)
            elif event.key() == QtCore.Qt.Key_R:
                self.viewbtns.button(3).setChecked(True)
                self.viewbtns.button(3).press(self, 3)
            elif event.key() == QtCore.Qt.Key_O:
                self.checkBox.toggle()
            elif event.key() == QtCore.Qt.Key_A:
                self.colorbtns.button(0).setChecked(True)
                self.colorbtns.button(0).press(self, 0)
            elif event.key() == QtCore.Qt.Key_S:
                self.colorbtns.button(1).setChecked(True)
                self.colorbtns.button(1).press(self, 1)
            elif event.key() == QtCore.Qt.Key_D:
                self.colorbtns.button(2).setChecked(True)
                self.colorbtns.button(2).press(self, 2)
            elif event.key() == QtCore.Qt.Key_F:
                self.colorbtns.button(3).setChecked(True)
                self.colorbtns.button(3).press(self, 3)
            elif event.key() == QtCore.Qt.Key_G:
                self.colorbtns.button(4).setChecked(True)
                self.colorbtns.button(4).press(self, 4)
            elif event.key() == QtCore.Qt.Key_H:
                self.colorbtns.button(5).setChecked(True)
                self.colorbtns.button(5).press(self, 5)
            elif event.key() == QtCore.Qt.Key_J:
                self.colorbtns.button(6).setChecked(True)
                self.colorbtns.button(6).press(self, 6)
            #elif event.key() == QtCore.Qt.Key_K:
            #    self.colorbtns.button(7).press(self, 0)

    def expand_trace(self):
        self.level+=1
        self.level = np.minimum(5,self.level)
        self.win.ci.layout.setRowStretchFactor(1,self.level)

    def collapse_trace(self):
        self.level-=1
        self.level = np.maximum(1,self.level)
        self.win.ci.layout.setRowStretchFactor(1,self.level)

    def ROI_selection(self, wplot):
        if wplot==0:
            view = self.p1.viewRange()
        else:
            view = self.p2.viewRange()
        self.ROIplot = wplot
        if self.selectbtn[wplot].isChecked():
            self.selectbtn[1-wplot].setEnabled(False)
            self.selectbtn[wplot].setStyleSheet(self.stylePressed)
            self.selectbtn[1-wplot].setStyleSheet(self.styleUnpressed)
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx  = (view[0][1] - view[0][0]) / 4
            dy  = (view[1][1] - view[1][0]) / 4
            imx = imx - dx/2
            imy = imy - dy/2
            self.ROI = pg.RectROI([imx,imy],[dx,dy],pen='w',sideScalers=True)
            if wplot==0:
                self.p1.addItem(self.ROI)
            else:
                self.p2.addItem(self.ROI)
            self.ROI_position()
            self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
            self.isROI = True
        else:
            self.selectbtn[wplot].setStyleSheet(self.styleUnpressed)
            self.ROI_remove()

    def ROI_remove(self):
        if self.isROI:
            if self.ROIplot==0:
                self.p1.removeItem(self.ROI)
            else:
                self.p2.removeItem(self.ROI)
            self.isROI=False
            self.selectbtn[1-self.ROIplot].setEnabled(True)
            self.selectbtn[self.ROIplot].setChecked(False)
            for b in range(2): self.selectbtn[b].setStyleSheet(self.styleUnpressed)

    def ROI_position(self):
        pos0 = self.ROI.getSceneHandlePositions()
        if self.ROIplot==0:
            pos = self.p1.mapSceneToView(pos0[0][1])
        else:
            pos = self.p2.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex,sizey = self.ROI.size()
        xrange = (np.arange(-1*int(sizex),1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1*int(sizey),1) + int(posy)).astype(np.int32)
        xrange = xrange[xrange>=0]
        xrange = xrange[xrange<self.ops['Lx']]
        yrange = yrange[yrange>=0]
        yrange = yrange[yrange<self.ops['Ly']]
        ypix,xpix = np.meshgrid(yrange,xrange)
        self.select_cells(ypix,xpix)

    def number_chosen(self):
        if self.loaded:
            self.ichosen = int(self.ROIedit.text())
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.imerge = [self.ichosen]
            if self.ops_plot[2]==self.ops_plot[3].shape[1]:
                fig.corr_masks(self)
                fig.plot_colorbar(self, self.ops_plot[2])
            self.ichosen_stats()
            M = fig.draw_masks(self)
            fig.plot_masks(self,M)
            fig.plot_trace(self)
            self.show()

    def select_cells(self,ypix,xpix):
        i = self.ROIplot
        iROI0 = self.iROI[i,0,ypix,xpix]
        icells = np.unique(iROI0[iROI0>=0])
        self.imerge = []
        for n in icells:
            if (self.iROI[i,:,ypix,xpix]==n).sum()>0.6*self.stat[n]['npix']:
                self.imerge.append(n)
        if len(self.imerge)>0:
            self.ichosen = self.imerge[0]
            if self.ops_plot[2]==self.ops_plot[3].shape[1]:
                fig.corr_masks(self)
                fig.plot_colorbar(self, self.ops_plot[2])
            self.ichosen_stats()
            M = fig.draw_masks(self)
            fig.plot_masks(self,M)
            fig.plot_trace(self)
            self.show()

    def make_masks_and_buttons(self):
        self.ROI_remove()
        self.ops_plot[1] = 0
        self.ops_plot[2] = 0
        self.ops_plot[3] = []
        self.ops_plot[4] = []
        self.setWindowTitle(self.fname)
        # add boundaries to stat for ROI overlays
        ncells = len(self.stat)
        for n in range(0,ncells):
            ypix = self.stat[n]['ypix'].flatten()
            xpix = self.stat[n]['xpix'].flatten()
            iext = fig.boundary(ypix,xpix)
            self.stat[n]['yext'] = ypix[iext]
            self.stat[n]['xext'] = xpix[iext]
            ycirc,xcirc = fig.circle(self.stat[n]['med'],self.stat[n]['radius'])
            self.stat[n]['ycirc'] = ycirc
            self.stat[n]['xcirc'] = xcirc
        # enable buttons
        self.enable_views_and_classifier()
        # make color arrays for various views
        fig.make_colors(self)
        self.ichosen = int(0)
        self.imerge = [int(0)]
        self.iflip = int(0)
        self.ichosen_stats()
        # colorbar
        self.colormat = fig.make_colorbar()
        fig.plot_colorbar(self, self.ops_plot[2])
        fig.init_masks(self)
        fig.corr_masks(self)
        M = fig.draw_masks(self)
        fig.plot_masks(self,M)
        self.lcell1.setText('%d'%(ncells-self.iscell.sum()))
        self.lcell0.setText('%d'%(self.iscell.sum()))
        fig.init_range(self)
        fig.plot_trace(self)
        if (type(self.ops['diameter']) is not int) and (len(self.ops['diameter'])>1):
            self.xyrat = self.ops['diameter'][0] / self.ops['diameter'][1]
        else:
            self.xyrat = 1.0
        self.p1.setAspectLocked(lock=True, ratio=self.xyrat)
        self.p2.setAspectLocked(lock=True, ratio=self.xyrat)
        self.show()
        # default classifier always loaded
        classifier.load(self, self.classfile, False)

    def enable_views_and_classifier(self):
        for b in range(len(self.views)):
            self.viewbtns.button(b).setEnabled(True)
            #self.viewbtns.button(b).setShortcut(QtGui.QKeySequence('R'))
            if b==0:
                self.viewbtns.button(b).setChecked(True)
                self.viewbtns.button(b).setStyleSheet(self.stylePressed)
        for b in range(len(self.colors)):
            self.colorbtns.button(b).setEnabled(True)
            if b==0:
                self.colorbtns.button(b).setChecked(True)
                self.colorbtns.button(b).setStyleSheet(self.stylePressed)
        for btn in self.classbtns.buttons():
            btn.setEnabled(True)
        b = 0
        for btn in self.sizebtns.buttons():
            btn.setEnabled(True)
            if b==1:
                btn.setChecked(True)
                btn.setStyleSheet(self.stylePressed)
            b+=1
        for i in range(2):
            self.selectbtn[i].setEnabled(True)
        # enable classifier menu
        self.loadClass.setEnabled(True)
        self.loadTrain.setEnabled(True)
        self.saveClass.setEnabled(True)
        self.saveDefault.setEnabled(True)
        self.addToClass.setEnabled(True)
        self.saveTrain.setEnabled(True)
        self.visualizations.setEnabled(True)

    def ROIs_on(self,state):
        if state == QtCore.Qt.Checked:
            self.ops_plot[0] = True
        else:
            self.ops_plot[0] = False
        if self.loaded:
            M = fig.draw_masks(self)
            fig.plot_masks(self,M)

    def plot_clicked(self,event):
        '''left-click chooses a cell, right-click flips cell to other view'''
        flip = False
        choose = False
        zoom = False
        replot = False
        items = self.win.scene().items(event.scenePos())
        posx  = 0
        posy  = 0
        iplot = 0
        if self.loaded:
            #print(event.modifiers() == QtCore.Qt.ControlModifier)
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
                        flip   = False
                if choose:
                    merged = False
                    if event.modifiers() == QtCore.Qt.ControlModifier:
                        if self.iscell[self.imerge[0]] == self.iscell[ichosen]:
                            if ichosen not in self.imerge:
                                self.imerge.append(ichosen)
                                self.ichosen = ichosen
                                merged = True
                            elif ichosen in self.imerge and len(self.imerge)>1:
                                self.imerge.remove(ichosen)
                                self.ichosen = self.imerge[0]
                                merged = True
                    if not merged:
                        self.imerge = [ichosen]
                        self.ichosen = ichosen
                if flip:
                    if ichosen not in self.imerge:
                        self.imerge = [ichosen]
                        self.ichosen = ichosen
                    self.flip_plot(iplot)
                    if self.isROI:
                        self.ROI_remove()
                if choose or flip or replot:
                    if self.isROI:
                        self.ROI_remove()
                    if self.ops_plot[2]==self.ops_plot[3].shape[1]:
                        fig.corr_masks(self)
                        fig.plot_colorbar(self, self.ops_plot[2])
                    self.ichosen_stats()
                    M = fig.draw_masks(self)
                    fig.plot_masks(self,M)
                    fig.plot_trace(self)
                    self.show()
                    #print(time.time()-tic)

    def ichosen_stats(self):
        n = self.ichosen
        self.ROIedit.setText(str(self.ichosen))
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
        for n in self.imerge:
            iscell = int(self.iscell[n])
            self.iscell[n] = ~self.iscell[n]
            self.ichosen = n
            fig.flip_cell(self)
        np.save(self.basename+'/iscell.npy',
                np.concatenate((np.expand_dims(self.iscell,axis=1),
                np.expand_dims(self.probcell,axis=1)), axis=1))
        self.lcell0.setText('%d'%(self.iscell.sum()))
        self.lcell1.setText('%d'%(self.iscell.size-self.iscell.sum()))

    def zoom_plot(self,iplot):
        if iplot==1 or iplot==2 or iplot==4:
            self.p1.setXRange(0,self.ops['Lx'])
            self.p1.setYRange(0,self.ops['Ly'])
            self.p2.setXRange(0,self.ops['Lx'])
            self.p2.setYRange(0,self.ops['Ly'])
        else:
            self.p3.setXRange(0,self.Fcell.shape[1])
            self.p3.setYRange(self.fmin,self.fmax)
        self.show()

    def run_suite2p(self):
        RW = gui.RunWindow(self)
        RW.show()

    def vis_window(self):
        VW = visualize.VisWindow(self)
        VW.show()

    def load_dialog(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open stat.npy', filter='stat.npy')
        self.fname = name[0]
        self.load_proc()

    def load_proc(self):
        name = self.fname
        print(name)
        try:
            stat = np.load(name)
            ypix = stat[0]['ypix']
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print('ERROR: this is not a stat.npy file :( (needs stat[n]["ypix"]!)')
            stat = None
        if stat is not None:
            basename, fname = os.path.split(name)
            goodfolder = True
            try:
                Fcell = np.load(basename + '/F.npy')
                Fneu = np.load(basename + '/Fneu.npy')
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('ERROR: there are no fluorescence traces in this folder (F.npy/Fneu.npy)')
                goodfolder = False
            try:
                Spks = np.load(basename + '/spks.npy')
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('there are no spike deconvolved traces in this folder (spks.npy)')
            try:
                iscell = np.load(basename + '/iscell.npy')
                probcell = iscell[:,1]
                iscell = iscell[:,0].astype(np.bool)
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('no manual labels found (iscell.npy)')
            try:
                ops = np.load(basename + '/ops.npy')
                ops = ops.item()
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('ERROR: there is no ops file in this folder (ops.npy)')
                goodfolder = False
            if goodfolder:
                self.basename = basename
                self.stat = stat
                self.ops  = ops
                self.Fcell = Fcell
                self.Fneu = Fneu
                self.Spks = Spks
                self.iscell = iscell
                self.probcell = probcell
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
            classifier.load(self, name[0], True)
        else:
            print('no classifier')

    def class_default(self):
        classfile = os.path.join(os.path.dirname(__file__), 'classifier_user.npy')
        np.save(classfile, self.model)

    #def save_gui_data(self):
    #    gui_data = {
    #                'RGBall': self.RGBall,
    #                'RGBback': self.RGBback,
    #                'Vback': self.Vback,
    #                'iROI': self.iROI,
    #                'iExt': self.iExt,
    #                'Sroi': self.Sroi,
    #                'Sext': self.Sext,
    #                'Lam': self.Lam,
    #                'LamMean': self.LamMean,
    #                'wasloaded': True
    #               }
    #    np.save(self.basename+'/gui_data.npy', gui_data)

def run():
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     'logo/logo.png')
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16,16))
    app_icon.addFile(icon_path, QtCore.QSize(24,24))
    app_icon.addFile(icon_path, QtCore.QSize(32,32))
    app_icon.addFile(icon_path, QtCore.QSize(48,48))
    app_icon.addFile(icon_path, QtCore.QSize(96,96))
    app_icon.addFile(icon_path, QtCore.QSize(256,256))
    app.setWindowIcon(app_icon)
    GUI = MainW()
    ret = app.exec_()
    #GUI.save_gui_data()
    sys.exit(ret)

#run()
