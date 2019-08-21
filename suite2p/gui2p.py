import sys
import os
import shutil
import time
import numpy as np
from numpy import genfromtxt
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from suite2p import fig, gui, classifier, visualize, reggui, classgui, merge, drawroi
from pkg_resources import iter_entry_points
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy import io
import csv

def resample_frames(y, x, xt):
    ''' resample y (defined at x) at times xt '''
    ts = x.size / xt.size
    y = gaussian_filter1d(y, np.ceil(ts/2), axis=0)
    f = interp1d(x,y,fill_value="extrapolate")
    yt = f(xt)
    return yt

class MainW(QtGui.QMainWindow):
    def __init__(self, statfile=None):
        super(MainW, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(0, 0, 1500, 600)
        self.setWindowTitle("suite2p (run pipeline or load stat.npy)")
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "logo/logo.png"
        )
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(96, 96))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.loaded = False
        self.ops_plot = []
        ### first time running, need to check for user files
        # check for classifier file
        self.classfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "classifiers/classifier_user.npy",
        )
        self.classorig = os.path.join(os.path.abspath(os.path.dirname(__file__)),
            "classifiers/classifier.npy"
        )
        if not os.path.isfile(self.classfile):
            shutil.copy(self.classorig, self.classfile)
        # check for ops file (for running suite2p)
        self.opsorig = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                          'ops/ops.npy')
        self.opsfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                          'ops/ops_user.npy')
        if not os.path.isfile(self.opsfile):
            shutil.copy(self.opsorig, self.opsfile)


        # default plot options
        self.ops_plot.append(True)
        for k in range(6):
            self.ops_plot.append(0)

        # --------------- MENU BAR --------------------------
        # run suite2p from scratch
        runS2P = QtGui.QAction("&Run suite2p ", self)
        runS2P.setShortcut("Ctrl+R")
        runS2P.triggered.connect(self.run_suite2p)
        self.addAction(runS2P)

        # load processed data
        loadProc = QtGui.QAction("&Load processed data", self)
        loadProc.setShortcut("Ctrl+L")
        loadProc.triggered.connect(self.load_dialog)
        self.addAction(loadProc)

        # load a behavioral trace
        self.loadBeh = QtGui.QAction(
            "Load behavior or stim traces(.npy, .csv)", self
        )
        self.loadBeh.triggered.connect(self.load_behavior)
        self.loadBeh.setEnabled(False)
        self.addAction(self.loadBeh)

        # load processed data
        self.saveMat = QtGui.QAction("&Save to mat file (*.mat)", self)
        self.saveMat.setShortcut("Ctrl+S")
        self.saveMat.triggered.connect(self.save_mat)
        self.saveMat.setEnabled(False)
        self.addAction(self.saveMat)

        # export figure
        exportFig = QtGui.QAction("Export as image (svg)", self)
        exportFig.triggered.connect(self.export_fig)
        exportFig.setEnabled(True)
        self.addAction(exportFig)

        # export figure
        self.manual = QtGui.QAction("Manual labelling", self)
        self.manual.triggered.connect(self.manual_label)
        self.manual.setEnabled(False)

        # make mainmenu!
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        file_menu.addAction(runS2P)
        file_menu.addAction(loadProc)
        file_menu.addAction(self.loadBeh)
        file_menu.addAction(self.saveMat)
        file_menu.addAction(exportFig)
        file_menu.addAction(self.manual)
        # classifier menu
        self.trainfiles = []
        self.statlabels = None
        self.loadMenu = QtGui.QMenu("Load", self)
        self.loadClass = QtGui.QAction("from file", self)
        self.loadClass.triggered.connect(self.load_classifier)
        self.loadClass.setEnabled(False)
        self.loadMenu.addAction(self.loadClass)
        self.loadUClass = QtGui.QAction("default classifier", self)
        self.loadUClass.triggered.connect(self.load_default_classifier)
        self.loadUClass.setEnabled(False)
        self.loadMenu.addAction(self.loadUClass)
        self.loadSClass = QtGui.QAction("built-in classifier", self)
        self.loadSClass.triggered.connect(self.load_s2p_classifier)
        self.loadSClass.setEnabled(False)
        self.loadMenu.addAction(self.loadSClass)
        self.loadTrain = QtGui.QAction("Build", self)
        self.loadTrain.triggered.connect(lambda: classgui.load_list(self))
        self.loadTrain.setEnabled(False)
        self.saveDefault = QtGui.QAction("Save loaded as default", self)
        self.saveDefault.triggered.connect(self.class_default)
        self.saveDefault.setEnabled(False)
        self.resetDefault = QtGui.QAction("Reset default to built-in", self)
        self.resetDefault.triggered.connect(self.reset_default)
        self.resetDefault.setEnabled(True)
        class_menu = main_menu.addMenu("&Classifier")
        class_menu.addMenu(self.loadMenu)
        class_menu.addAction(self.loadTrain)
        class_menu.addAction(self.resetDefault)
        class_menu.addAction(self.saveDefault)

        # visualizations menuBar
        vis_menu = main_menu.addMenu("&Visualizations")
        self.visualizations = QtGui.QAction("&Visualize selected cells", self)
        self.visualizations.triggered.connect(self.vis_window)
        self.visualizations.setEnabled(False)
        vis_menu.addAction(self.visualizations)
        self.visualizations.setShortcut("Ctrl+V")
        self.custommask = QtGui.QAction("Load custom hue for ROIs (*.npy)", self)
        self.custommask.triggered.connect(self.load_custom_mask)
        self.custommask.setEnabled(False)
        vis_menu.addAction(self.custommask)

        # registration menuBar
        reg_menu = main_menu.addMenu("&Registration")
        self.reg = QtGui.QAction("View registered &binary", self)
        self.reg.triggered.connect(self.reg_window)
        self.reg.setShortcut("Ctrl+B")
        self.reg.setEnabled(True)
        self.regPC = QtGui.QAction("View registration &Metrics", self)
        self.regPC.triggered.connect(self.regPC_window)
        self.regPC.setShortcut("Ctrl+M")
        self.regPC.setEnabled(True)
        reg_menu.addAction(self.reg)
        reg_menu.addAction(self.regPC)

        # plugin menu
        self.plugins = {}
        plugin_menu = main_menu.addMenu('&Plugins')
        for entry_pt in iter_entry_points(group='suite2p.plugin', name=None):
            self.plugins[entry_pt.name] = QtGui.QAction(entry_pt.menu, self)
            self.plugins[entry_pt.name].triggered.connect(entry_pt.window)
            plugin_menu.addAction(self.plugins[entry_pt.name])

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QtGui.QWidget()
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        # ROI CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBox = QtGui.QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.stateChanged.connect(self.ROIs_on)
        self.checkBox.toggle()
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)
        # number of ROIs in each image
        self.lcell0 = QtGui.QLabel("")
        self.lcell0.setStyleSheet("color: white;")
        self.lcell0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.l0.addWidget(self.lcell0, 0, 12, 1, 2)
        self.lcell1 = QtGui.QLabel("")
        self.lcell1.setStyleSheet("color: white;")
        self.l0.addWidget(self.lcell1, 0, 20, 1, 2)
        # buttons to draw a square on view
        self.topbtns = QtGui.QButtonGroup()
        ql = QtGui.QLabel("select cells")
        ql.setStyleSheet("color: white;")
        ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(ql, 0, 2, 1, 2)
        pos = [2, 3, 4]
        for b in range(3):
            btn = gui.TopButton(b, self)
            btn.setFont(QtGui.QFont("Arial", 8))
            self.topbtns.addButton(btn, b)
            self.l0.addWidget(btn, 0, (pos[b]) * 2, 1, 2)
            btn.setEnabled(False)
        self.topbtns.setExclusive(True)
        self.isROI = False
        self.ROIplot = 0
        ql = QtGui.QLabel("n=")
        ql.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ql.setStyleSheet("color: white;")
        ql.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.l0.addWidget(ql, 0, 10, 1, 1)
        # self.l0.setColumnStretch(5,0.1)
        self.topedit = QtGui.QLineEdit(self)
        self.topedit.setValidator(QtGui.QIntValidator(0, 500))
        self.topedit.setText("40")
        self.ntop = 40
        self.topedit.setFixedWidth(35)
        self.topedit.setAlignment(QtCore.Qt.AlignRight)
        self.topedit.returnPressed.connect(self.top_number_chosen)
        self.l0.addWidget(self.topedit, 0, 11, 1, 1)
        # minimize view
        self.sizebtns = QtGui.QButtonGroup(self)
        b = 0
        labels = [" cells", " both", " not cells"]
        for l in labels:
            btn = gui.SizeButton(b, l, self)
            self.sizebtns.addButton(btn, b)
            self.l0.addWidget(btn, 0, 14 + 2 * b, 1, 2)
            btn.setEnabled(False)
            if b == 1:
                btn.setEnabled(True)
            b += 1
        self.sizebtns.setExclusive(True)

        ##### -------- MAIN PLOTTING AREA ---------- ####################
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, 41, 30)
        layout = self.win.ci.layout
        # --- cells image
        self.p1 = self.win.addViewBox(
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            row=0,
            col=0,
            invertY=True,
        )
        self.img1 = pg.ImageItem()
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        data = np.zeros((700, 512, 3))
        self.img1.setImage(data)
        self.p1.addItem(self.img1)
        # --- noncells image
        self.p2 = self.win.addViewBox(
            lockAspect=True,
            name="plot2",
            border=[100, 100, 100],
            row=0,
            col=1,
            invertY=True,
        )
        self.p2.setMenuEnabled(False)
        self.p2.scene().contextMenuItem = self.p2
        self.img2 = pg.ImageItem()
        self.img2.setImage(data)
        self.p2.addItem(self.img2)
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")
        # --- fluorescence trace plot
        self.p3 = self.win.addPlot(row=1, col=0, colspan=2)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.enableAutoRange(x=True, y=True)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # self.key_on(self.win.scene().keyPressEvent)
        self.show()
        self.win.show()

        ###### --------- QUADRANT AND VIEW AND COLOR BUTTONS ---------- #############
        self.views = [
            "Q: ROIs",
            "W: mean img",
            "E: mean img (enhanced)",
            "R: correlation map",
            "T: max projection",
            "Y: mean img (chan2, corrected)",
            "U: mean img (chan2)",
        ]
        self.colors = [
            "A: random",
            "S: skew",
            "D: compact",
            "F: footprint",
            "G: aspect_ratio",
            "H: chan2_prob",
            "J: classifier, cell prob=",
            "K: correlations, bin=",
        ]

        # quadrant buttons
        self.quadbtns = QtGui.QButtonGroup(self)
        #vlabel = QtGui.QLabel(self)
        #vlabel.setText("<font color='white'>Background</font>")
        #vlabel.setFont(boldfont)
        #vlabel.resize(vlabel.minimumSizeHint())
        #self.l0.addWidget(vlabel, 1, 0, 1, 1)
        for b in range(9):
            btn = gui.QuadButton(b, ' '+str(b+1), self)
            self.quadbtns.addButton(btn, b)
            self.l0.addWidget(btn, 0 + self.quadbtns.button(b).ypos, 29 + self.quadbtns.button(b).xpos, 1, 1)
            btn.setEnabled(False)
            b += 1
        self.quadbtns.setExclusive(True)

        # view buttons
        b = 0
        boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        self.viewbtns = QtGui.QButtonGroup(self)
        vlabel = QtGui.QLabel(self)
        vlabel.setText("<font color='white'>Background</font>")
        vlabel.setFont(boldfont)
        vlabel.resize(vlabel.minimumSizeHint())
        self.l0.addWidget(vlabel, 1, 0, 1, 1)
        for names in self.views:
            btn = gui.ViewButton(b, "&" + names, self)
            self.viewbtns.addButton(btn, b)
            self.l0.addWidget(btn, b + 2, 0, 1, 2)
            btn.setEnabled(False)
            b += 1
        self.viewbtns.setExclusive(True)
        # color buttons
        self.colorbtns = QtGui.QButtonGroup(self)
        clabel = QtGui.QLabel(self)
        clabel.setText("<font color='white'>Colors</font>")
        clabel.setFont(boldfont)
        self.l0.addWidget(clabel, b + 3, 0, 1, 2)
        nv = b + 3
        b = 0
        # colorbars for different statistics
        colorsAll = self.colors.copy()
        colorsAll.append("L: corr with 1D var, bin= ^^^")
        colorsAll.append("M: rastermap / custom")
        for names in colorsAll:
            btn = gui.ColorButton(b, "&" + names, self)
            self.colorbtns.addButton(btn, b)
            if b == len(self.colors) - 1 or b==5 or b==6:
                self.l0.addWidget(btn, nv + b + 1, 0, 1, 1)
            else:
                self.l0.addWidget(btn, nv + b + 1, 0, 1, 2)
            btn.setEnabled(False)
            if b < len(self.colors):
                self.colors[b] = self.colors[b][3:]
            b += 1
        self.chan2edit = QtGui.QLineEdit(self)
        self.chan2edit.setText("0.6")
        self.chan2edit.setFixedWidth(40)
        self.chan2edit.setAlignment(QtCore.Qt.AlignRight)
        self.chan2edit.returnPressed.connect(self.chan2_prob)
        self.l0.addWidget(self.chan2edit, nv + b - 4, 1, 1, 1)

        self.probedit = QtGui.QLineEdit(self)
        self.probedit.setText("0.5")
        self.probedit.setFixedWidth(40)
        self.probedit.setAlignment(QtCore.Qt.AlignRight)
        self.probedit.returnPressed.connect(
            lambda: classgui.apply(self)
        )
        self.l0.addWidget(self.probedit, nv + b - 3, 1, 1, 1)


        self.binedit = QtGui.QLineEdit(self)
        self.binedit.setValidator(QtGui.QIntValidator(0, 500))
        self.binedit.setText("0")
        self.binedit.setFixedWidth(40)
        self.binedit.setAlignment(QtCore.Qt.AlignRight)
        self.binedit.returnPressed.connect(
            lambda: self.mode_change(self.activityMode)
        )
        self.l0.addWidget(self.binedit, nv + b - 2, 1, 1, 1)
        self.bend = nv + b + 4
        colorbarW = pg.GraphicsLayoutWidget()
        colorbarW.setMaximumHeight(60)
        colorbarW.setMaximumWidth(150)
        colorbarW.ci.layout.setRowStretchFactor(0, 2)
        colorbarW.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.l0.addWidget(colorbarW, nv + b + 2, 0, 1, 2)
        self.colorbar = pg.ImageItem()
        cbar = colorbarW.addViewBox(row=0, col=0, colspan=3)
        cbar.setMenuEnabled(False)
        cbar.addItem(self.colorbar)
        self.clabel = [
            colorbarW.addLabel("0.0", color=[255, 255, 255], row=1, col=0),
            colorbarW.addLabel("0.5", color=[255, 255, 255], row=1, col=1),
            colorbarW.addLabel("1.0", color=[255, 255, 255], row=1, col=2),
        ]

        # ----- CLASSIFIER BUTTONS -------
        cllabel = QtGui.QLabel("")
        cllabel.setFont(boldfont)
        cllabel.setText("<font color='white'>Classifier</font>")
        self.classLabel = QtGui.QLabel("<font color='white'>not loaded (using prob from iscell.npy)</font>")
        self.classLabel.setFont(QtGui.QFont("Arial", 8))
        self.l0.addWidget(cllabel, self.bend + 1, 0, 1, 2)
        self.l0.addWidget(self.classLabel, self.bend + 2, 0, 1, 2)
        self.addtoclass = QtGui.QPushButton(" add current data to classifier")
        self.addtoclass.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addtoclass.clicked.connect(lambda: classgui.add_to(self))
        self.addtoclass.setStyleSheet(self.styleInactive)
        self.l0.addWidget(self.addtoclass, self.bend + 3, 0, 1, 2)
        # ------ CELL STATS --------
        # which stats
        self.bend = self.bend + 4
        self.stats_to_show = [
            "med",
            "npix",
            "skew",
            "compact",
            "footprint",
            "aspect_ratio"
        ]
        lilfont = QtGui.QFont("Arial", 8)
        qlabel = QtGui.QLabel(self)
        qlabel.setFont(boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, self.bend, 0, 1, 2)
        self.ROIedit = QtGui.QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, self.bend + 1, 0, 1, 1)
        self.ROIstats = []
        self.ROIstats.append(qlabel)
        for k in range(1, len(self.stats_to_show) + 1):
            llabel = QtGui.QLabel(self.stats_to_show[k - 1])
            self.ROIstats.append(llabel)
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].setStyleSheet("color: white;")
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.addWidget(self.ROIstats[k], self.bend + 2 + k, 0, 1, 2)
        self.l0.addWidget(QtGui.QLabel(""), self.bend + 3 + k, 0, 1, 2)
        self.l0.setRowStretch(self.bend + 3 + k, 1)
        # combo box to decide what kind of activity to view
        qlabel = QtGui.QLabel(self)
        qlabel.setText("<font color='white'>Activity mode:</font>")
        self.l0.addWidget(qlabel, self.bend + k + 4, 0, 1, 1)
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.setFixedWidth(100)
        self.l0.addWidget(self.comboBox, self.bend + k + 5, 0, 1, 1)
        self.comboBox.addItem("F")
        self.comboBox.addItem("Fneu")
        self.comboBox.addItem("F - 0.7*Fneu")
        self.comboBox.addItem("deconvolved")
        self.activityMode = 3
        self.comboBox.setCurrentIndex(self.activityMode)
        self.comboBox.currentIndexChanged.connect(self.mode_change)
        # up/down arrows to resize view
        self.level = 1
        self.arrowButtons = [
            QtGui.QPushButton(u" \u25b2"),
            QtGui.QPushButton(u" \u25bc"),
        ]
        self.arrowButtons[0].clicked.connect(self.expand_trace)
        self.arrowButtons[1].clicked.connect(self.collapse_trace)
        b = 0
        for btn in self.arrowButtons:
            btn.setMaximumWidth(22)
            btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
            btn.setStyleSheet(self.styleUnpressed)
            self.l0.addWidget(
                btn,
                self.bend + 4 + k + b, 1, 1, 1,
                QtCore.Qt.AlignRight
            )
            b += 1
        self.pmButtons = [QtGui.QPushButton(" +"), QtGui.QPushButton(" -")]
        self.pmButtons[0].clicked.connect(self.expand_scale)
        self.pmButtons[1].clicked.connect(self.collapse_scale)
        b = 0
        self.sc = 2
        for btn in self.pmButtons:
            btn.setMaximumWidth(22)
            btn.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
            btn.setStyleSheet(self.styleUnpressed)
            self.l0.addWidget(btn, self.bend + 4 + k + b, 1, 1, 1)
            b += 1
        # choose max # of cells plotted
        self.l0.addWidget(
            QtGui.QLabel("<font color='white'>max # plotted:</font>"),
            self.bend + 6 + k,
            0,
            1,
            1,
        )
        self.ncedit = QtGui.QLineEdit(self)
        self.ncedit.setValidator(QtGui.QIntValidator(0, 400))
        self.ncedit.setText("40")
        self.ncedit.setFixedWidth(35)
        self.ncedit.setAlignment(QtCore.Qt.AlignRight)
        self.ncedit.returnPressed.connect(self.nc_chosen)
        self.l0.addWidget(self.ncedit, self.bend + 7 + k, 0, 1, 1)
        # labels for traces
        self.traceLabel = [
            QtGui.QLabel(self),
            QtGui.QLabel(self),
            QtGui.QLabel(self)
        ]
        self.traceText = [
            "<font color='blue'>fluorescence</font>",
            "<font color='red'>neuropil</font>",
            "<font color='gray'>deconvolved</font>",
        ]
        for n in range(3):
            self.traceLabel[n].setText(self.traceText[n])
            self.traceLabel[n].setFont(boldfont)
            self.l0.addWidget(
                self.traceLabel[n],
                self.bend + 7 + k,
                4 + n * 2,
                1, 2
            )
        # initialize merges
        self.merged = []
        self.rastermap = False
        model = np.load(self.classorig, allow_pickle=True).item()
        self.default_keys = model["keys"]

        # load initial file
        statfile = '/groups/hackathon/data/guest3/sampleData/TSeries-06262017-1330-1minrec-002/suite2p/plane0/stat.npy'
        if statfile is not None:
            self.fname = statfile
            self.load_proc()
            #self.manual_label()


    def export_fig(self):
        self.win.scene().contextMenuItem = self.p1
        self.win.scene().showExportDialog()

    def mode_change(self, i):
        '''
            changes the activity mode that is used when multiple neurons are selected
            or in visualization windows like rastermap or for correlation computation!

            activityMode =
            0 : F
            1 : Fneu
            2 : F - 0.7 * Fneu (default)
            3 : spks

            uses binning set by self.bin
        '''
        self.activityMode = i
        if self.loaded:
            # activity used for correlations
            self.bin = int(self.binedit.text())
            nb = int(np.floor(float(self.Fcell.shape[1]) / float(self.bin)))
            if i == 0:
                f = self.Fcell
            elif i == 1:
                f = self.Fneu
            elif i == 2:
                f = self.Fcell - 0.7 * self.Fneu
            else:
                f = self.Spks
            ncells = len(self.stat)
            self.Fbin = f[:, : nb * self.bin].reshape(
                (ncells, nb, self.bin)
            ).mean(axis=2)

            self.Fbin = self.Fbin - self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin ** 2).sum(axis=1)
            self.trange = np.arange(0, self.Fcell.shape[1])
            # if in correlation-view, recompute
            if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                fig.corr_masks(self)
            elif self.ops_plot[2] == self.ops_plot[3].shape[1] + 1:
                fig.beh_masks(self)
            fig.plot_colorbar(self, self.ops_plot[2])
            M = fig.draw_masks(self)
            fig.plot_masks(self, M)
            fig.plot_trace(self)
            self.show()

    def keyPressEvent(self, event):
        if self.loaded:
            if event.modifiers() != QtCore.Qt.ControlModifier and event.modifiers() != QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Return:
                    if 0:
                        if len(self.imerge) > 1:
                            self.merge_cells()
                elif event.key() == QtCore.Qt.Key_Escape:
                    self.zoom_plot(1)
                    self.zoom_plot(3)
                    self.show()
                elif event.key() == QtCore.Qt.Key_Delete:
                    self.ROI_remove()
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
                elif event.key() == QtCore.Qt.Key_T:
                    self.viewbtns.button(4).setChecked(True)
                    self.viewbtns.button(4).press(self, 4)
                elif event.key() == QtCore.Qt.Key_U:
                    if "meanImg_chan2" in self.ops:
                        self.viewbtns.button(6).setChecked(True)
                        self.viewbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_Y:
                    if "meanImg_chan2_corrected" in self.ops:
                        self.viewbtns.button(5).setChecked(True)
                        self.viewbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_Space:
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
                    if self.hasred:
                        self.colorbtns.button(5).setChecked(True)
                        self.colorbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_J:
                    self.colorbtns.button(6).setChecked(True)
                    self.colorbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_K:
                    self.colorbtns.button(7).setChecked(True)
                    self.colorbtns.button(7).press(self, 7)
                elif event.key() == QtCore.Qt.Key_L:
                    if self.bloaded:
                        self.colorbtns.button(8).setChecked(True)
                        self.colorbtns.button(8).press(self, 8)
                elif event.key() == QtCore.Qt.Key_M:
                    if self.rastermap:
                        self.colorbtns.button(9).setChecked(True)
                        self.colorbtns.button(9).press(self, 9)
                elif event.key() == QtCore.Qt.Key_Left:
                    self.ichosen = (self.ichosen-1)%len(self.stat)
                    self.imerge = [self.ichosen]
                    self.ichosen_stats()
                    M = fig.draw_masks(self)
                    fig.plot_masks(self, M)
                    fig.plot_trace(self)
                    self.show()
                elif event.key() == QtCore.Qt.Key_Right:
                    self.ichosen = (self.ichosen+1)%len(self.stat)
                    self.imerge = [self.ichosen]
                    self.ichosen_stats()
                    M = fig.draw_masks(self)
                    fig.plot_masks(self, M)
                    fig.plot_trace(self)
                    self.show()


    def merge_cells(self):
        dm = QtGui.QMessageBox.question(
            self,
            "Merge cells",
            "Do you want to merge selected cells?",
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
        )
        if dm == QtGui.QMessageBox.Yes:
            nmerged = len(self.merged)
            merge.activity_stats(self)
            merge.fig_masks(self)
            M = fig.draw_masks(self)
            fig.plot_masks(self, M)
            fig.plot_trace(self)
            self.show()
            self.merged.append(self.imerge)
            print(self.merged)
            print('merged ROIs')

    def expand_scale(self):
        self.sc += 0.5
        self.sc = np.minimum(10, self.sc)
        fig.plot_trace(self)
        self.show()

    def collapse_scale(self):
        self.sc -= 0.5
        self.sc = np.maximum(0.5, self.sc)
        fig.plot_trace(self)
        self.show()

    def expand_trace(self):
        self.level += 1
        self.level = np.minimum(5, self.level)
        self.win.ci.layout.setRowStretchFactor(1, self.level)

    def collapse_trace(self):
        self.level -= 1
        self.level = np.maximum(1, self.level)
        self.win.ci.layout.setRowStretchFactor(1, self.level)

    def nc_chosen(self):
        if self.loaded:
            fig.plot_trace(self)
            self.show()

    def top_number_chosen(self):
        self.ntop = int(self.topedit.text())
        if self.loaded:
            if not self.sizebtns.button(1).isChecked():
                for b in [1, 2]:
                    if self.topbtns.button(b).isChecked():
                        self.top_selection(b)
                        self.show()

    def top_selection(self, bid):
        self.ROI_remove()
        draw = False
        ncells = len(self.stat)
        icells = np.minimum(ncells, self.ntop)
        if bid == 1:
            top = True
        elif bid == 2:
            top = False
        if self.sizebtns.button(0).isChecked():
            wplot = 0
            draw = True
        elif self.sizebtns.button(2).isChecked():
            wplot = 1
            draw = True
        if draw:
            if self.ops_plot[2] != 0:
                # correlation view
                if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                    istat = self.ops_plot[4]
                elif self.ops_plot[2] == self.ops_plot[3].shape[1] + 1:
                    istat = self.ops_plot[5]
                elif self.ops_plot[2] == self.ops_plot[3].shape[1] + 2:
                    istat = self.ops_plot[6]
                # statistics view
                else:
                    istat = self.ops_plot[3][:, self.ops_plot[2]]
                if wplot == 0:
                    icell = np.array(self.iscell.nonzero()).flatten()
                    istat = istat[self.iscell]
                else:
                    icell = np.array((~self.iscell).nonzero()).flatten()
                    istat = istat[~self.iscell]
                inds = istat.argsort()
                if top:
                    inds = inds[:icells]
                    self.ichosen = icell[inds[-1]]
                else:
                    inds = inds[-icells:]
                    self.ichosen = icell[inds[0]]
                self.imerge = []
                for n in inds:
                    self.imerge.append(icell[n])
                # draw choices
                if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                    fig.corr_masks(self)
                    fig.plot_colorbar(self, self.ops_plot[2])
                self.ichosen_stats()
                M = fig.draw_masks(self)
                fig.plot_masks(self, M)
                fig.plot_trace(self)
                self.show()

    def ROI_selection(self):
        draw = False
        if self.sizebtns.button(0).isChecked():
            wplot = 0
            view = self.p1.viewRange()
            draw = True
        elif self.sizebtns.button(2).isChecked():
            wplot = 1
            view = self.p2.viewRange()
            draw = True
        if draw:
            self.ROI_remove()
            self.topbtns.button(0).setStyleSheet(self.stylePressed)
            self.ROIplot = wplot
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, 300)
            dy = np.minimum(dy, 300)
            imx = imx - dx / 2
            imy = imy - dy / 2
            self.ROI = pg.RectROI(
                [imx, imy], [dx, dy],
                pen="w", sideScalers=True
            )
            if wplot == 0:
                self.p1.addItem(self.ROI)
            else:
                self.p2.addItem(self.ROI)
            self.ROI_position()
            self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
            self.isROI = True

    def ROI_remove(self):
        if self.isROI:
            if self.ROIplot == 0:
                self.p1.removeItem(self.ROI)
            else:
                self.p2.removeItem(self.ROI)
            self.isROI = False
        if self.sizebtns.button(1).isChecked():
            self.topbtns.button(0).setStyleSheet(self.styleInactive)
            self.topbtns.button(0).setEnabled(False)
        else:
            self.topbtns.button(0).setStyleSheet(self.styleUnpressed)

    def ROI_position(self):
        pos0 = self.ROI.getSceneHandlePositions()
        if self.ROIplot == 0:
            pos = self.p1.mapSceneToView(pos0[0][1])
        else:
            pos = self.p2.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        xrange = xrange[xrange >= 0]
        xrange = xrange[xrange < self.ops["Lx"]]
        yrange = yrange[yrange >= 0]
        yrange = yrange[yrange < self.ops["Ly"]]
        ypix, xpix = np.meshgrid(yrange, xrange)
        self.select_cells(ypix, xpix)

    def number_chosen(self):
        if self.loaded:
            self.ichosen = int(self.ROIedit.text())
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.imerge = [self.ichosen]
            if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                fig.corr_masks(self)
                fig.plot_colorbar(self, self.ops_plot[2])
            self.ichosen_stats()
            M = fig.draw_masks(self)
            fig.plot_masks(self, M)
            fig.plot_trace(self)
            self.show()

    def select_cells(self, ypix, xpix):
        i = self.ROIplot
        iROI0 = self.iROI[i, 0, ypix, xpix]
        icells = np.unique(iROI0[iROI0 >= 0])
        self.imerge = []
        for n in icells:
            if (self.iROI[i, :, ypix, xpix] == n).sum() > 0.6 * self.stat[n]["npix"]:
                self.imerge.append(n)
        if len(self.imerge) > 0:
            self.ichosen = self.imerge[0]
            if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                fig.corr_masks(self)
                fig.plot_colorbar(self, self.ops_plot[2])
            self.ichosen_stats()
            M = fig.draw_masks(self)
            fig.plot_masks(self, M)
            fig.plot_trace(self)
            self.show()

    def make_masks_and_buttons(self):
        self.loadBeh.setEnabled(True)
        self.saveMat.setEnabled(True)
        self.manual.setEnabled(True)
        self.bloaded = False
        self.ROI_remove()
        self.isROI = False
        self.ops_plot[1] = 0
        self.ops_plot[2] = 0
        self.ops_plot[3] = []
        self.ops_plot[4] = []
        self.ops_plot[5] = []
        self.ops_plot[6] = []
        self.setWindowTitle(self.fname)
        # set bin size to be 0.5s by default
        self.bin = int(self.ops["tau"] * self.ops["fs"] / 2)
        self.binedit.setText(str(self.bin))
        if "chan2_thres" not in self.ops:
            self.ops["chan2_thres"] = 0.6
        self.chan2prob = self.ops["chan2_thres"]
        self.chan2edit.setText(str(self.chan2prob))
        # add boundaries to stat for ROI overlays
        ncells = len(self.stat)
        for n in range(0, ncells):
            ypix = self.stat[n]["ypix"].flatten()
            xpix = self.stat[n]["xpix"].flatten()
            iext = fig.boundary(ypix, xpix)
            self.stat[n]["yext"] = ypix[iext]
            self.stat[n]["xext"] = xpix[iext]
            ycirc, xcirc = fig.circle(
                self.stat[n]["med"],
                self.stat[n]["radius"]
            )
            goodi = (
                (ycirc >= 0)
                & (xcirc >= 0)
                & (ycirc < self.ops["Ly"])
                & (xcirc < self.ops["Lx"])
            )
            self.stat[n]["ycirc"] = ycirc[goodi]
            self.stat[n]["xcirc"] = xcirc[goodi]
        # enable buttons
        self.enable_views_and_classifier()
        # make color arrays for various views
        fig.make_colors(self)
        self.ichosen = int(0)
        self.imerge = [int(0)]
        self.iflip = int(0)
        self.ichosen_stats()
        self.comboBox.setCurrentIndex(2)
        # colorbar
        self.colormat = fig.make_colorbar()
        fig.plot_colorbar(self, self.ops_plot[2])
        tic = time.time()
        fig.init_masks(self)
        print(time.time() - tic)
        M = fig.draw_masks(self)
        fig.plot_masks(self, M)
        self.lcell1.setText("%d" % (ncells - self.iscell.sum()))
        self.lcell0.setText("%d" % (self.iscell.sum()))
        fig.init_range(self)
        fig.plot_trace(self)
        if 'aspect' in self.ops:
            self.xyrat = self.ops['aspect']
        elif 'diameter' in self.ops and (type(self.ops["diameter"]) is not int) and (len(self.ops["diameter"]) > 1):
            self.xyrat = self.ops["diameter"][0] / self.ops["diameter"][1]
        else:
            self.xyrat = 1.0
        self.p1.setAspectLocked(lock=True, ratio=self.xyrat)
        self.p2.setAspectLocked(lock=True, ratio=self.xyrat)
        self.loaded = True
        self.mode_change(2)
        self.show()
        # no classifier loaded
        classgui.activate(self, False)

    def enable_views_and_classifier(self):
        for b in range(9):
            self.quadbtns.button(b).setEnabled(True)
            self.quadbtns.button(b).setStyleSheet(self.styleUnpressed)
        for b in range(len(self.views)):
            self.viewbtns.button(b).setEnabled(True)
            self.viewbtns.button(b).setStyleSheet(self.styleUnpressed)
            # self.viewbtns.button(b).setShortcut(QtGui.QKeySequence('R'))
            if b == 0:
                self.viewbtns.button(b).setChecked(True)
                self.viewbtns.button(b).setStyleSheet(self.stylePressed)
        # check for second channel
        if "meanImg_chan2_corrected" not in self.ops:
            self.viewbtns.button(5).setEnabled(False)
            self.viewbtns.button(5).setStyleSheet(self.styleInactive)
            if "meanImg_chan2" not in self.ops:
                self.viewbtns.button(6).setEnabled(False)
                self.viewbtns.button(6).setStyleSheet(self.styleInactive)

        for b in range(len(self.colors)):
            if b==5:
                if self.hasred:
                    self.colorbtns.button(b).setEnabled(True)
                    self.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
            elif b==0:
                self.colorbtns.button(b).setEnabled(True)
                self.colorbtns.button(b).setChecked(True)
                self.colorbtns.button(b).setStyleSheet(self.stylePressed)
            else:
                self.colorbtns.button(b).setEnabled(True)
                self.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
        #self.applyclass.setStyleSheet(self.styleUnpressed)
        #self.applyclass.setEnabled(True)
        b = 0
        for btn in self.sizebtns.buttons():
            btn.setStyleSheet(self.styleUnpressed)
            btn.setEnabled(True)
            if b == 1:
                btn.setChecked(True)
                btn.setStyleSheet(self.stylePressed)
            b += 1
        for b in range(3):
            self.topbtns.button(b).setEnabled(False)
            self.topbtns.button(b).setStyleSheet(self.styleInactive)
        # enable classifier menu
        self.loadClass.setEnabled(True)
        self.loadTrain.setEnabled(True)
        self.loadUClass.setEnabled(True)
        self.loadSClass.setEnabled(True)
        self.resetDefault.setEnabled(True)
        self.visualizations.setEnabled(True)
        self.custommask.setEnabled(True)
        # self.p1.scene().showExportDialog()

    def ROIs_on(self, state):
        if state == QtCore.Qt.Checked:
            self.ops_plot[0] = True
        else:
            self.ops_plot[0] = False
        if self.loaded:
            M = fig.draw_masks(self)
            fig.plot_masks(self, M)

    def plot_clicked(self, event):
        """left-click chooses a cell, right-click flips cell to other view"""
        flip = False
        choose = False
        zoom = False
        replot = False
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        if self.loaded:
            # print(event.modifiers() == QtCore.Qt.ControlModifier)
            for x in items:
                if x == self.img1:
                    pos = self.p1.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 1
                elif x == self.img2:
                    pos = self.p2.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 2
                elif x == self.p3:
                    iplot = 3
                elif (
                    (x == self.p1 or x == self.p2) and
                    x != self.img1 and
                    x != self.img2
                ):
                    iplot = 4
                    if event.double():
                        zoom = True
                if iplot == 1 or iplot == 2:
                    if event.button() == 2:
                        flip = True
                    elif event.button() == 1:
                        if event.double():
                            zoom = True
                        else:
                            choose = True
                if iplot == 3 and event.double():
                    zoom = True
                posy = int(posy)
                posx = int(posx)
                if zoom:
                    self.zoom_plot(iplot)
                if (choose or flip) and (iplot == 1 or iplot == 2):
                    ichosen = int(self.iROI[iplot - 1, 0, posx, posy])
                    if ichosen < 0:
                        choose = False
                        flip = False
                if choose:
                    merged = False
                    if event.modifiers() == QtCore.Qt.ShiftModifier or event.modifiers() == QtCore.Qt.ControlModifier:
                        if self.iscell[self.imerge[0]] == self.iscell[ichosen]:
                            if ichosen not in self.imerge:
                                self.imerge.append(ichosen)
                                self.ichosen = ichosen
                                merged = True
                            elif ichosen in self.imerge and len(self.imerge) > 1:
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
                if choose or flip or replot:
                    if self.isROI:
                        self.ROI_remove()
                    if not self.sizebtns.button(1).isChecked():
                        for btn in self.topbtns.buttons():
                            if btn.isChecked():
                                btn.setStyleSheet(self.styleUnpressed)
                    if self.ops_plot[2] == self.ops_plot[3].shape[1]:
                        fig.corr_masks(self)
                        fig.plot_colorbar(self, self.ops_plot[2])
                    self.ichosen_stats()
                    M = fig.draw_masks(self)
                    fig.plot_masks(self, M)
                    fig.plot_trace(self)
                    self.show()
                elif event.button() == 2:
                    if iplot == 1:
                        event.acceptedItem = self.p1
                        self.p1.raiseContextMenu(event)
                    elif iplot == 2:
                        event.acceptedItem = self.p2
                        self.p2.raiseContextMenu(event)

    def ichosen_stats(self):
        n = self.ichosen
        self.ROIedit.setText(str(self.ichosen))
        for k in range(1, len(self.stats_to_show) + 1):
            key = self.stats_to_show[k - 1]
            ival = self.stat[n][key]
            if k == 1:
                self.ROIstats[k].setText(
                    key + ": [%d, %d]" % (ival[0], ival[1])
                )
            elif k == 2:
                self.ROIstats[k].setText(key + ": %d" % (ival))
            else:
                self.ROIstats[k].setText(key + ": %2.2f" % (ival))

    def flip_plot(self, iplot):
        self.iflip = self.ichosen
        for n in self.imerge:
            iscell = int(self.iscell[n])
            self.iscell[n] = ~self.iscell[n]
            self.ichosen = n
            fig.flip_cell(self)
        np.save(
            self.basename + "/iscell.npy",
            np.concatenate(
                (
                    np.expand_dims(self.iscell, axis=1),
                    np.expand_dims(self.probcell, axis=1),
                ),
                axis=1,
            ),
        )
        self.lcell0.setText("%d" % (self.iscell.sum()))
        self.lcell1.setText("%d" % (self.iscell.size - self.iscell.sum()))

    def chan2_prob(self):
        chan2prob = float(self.chan2edit.text())
        if abs(self.chan2prob - chan2prob) > 1e-3:
            self.chan2prob = chan2prob
            self.redcell = self.probredcell > self.chan2prob
            fig.chan2_masks(self)
            M = fig.draw_masks(self)
            fig.plot_masks(self,M)
            np.save(os.path.join(self.basename, 'redcell.npy'),
                    np.concatenate((np.expand_dims(self.redcell,axis=1),
                    np.expand_dims(self.probredcell,axis=1)), axis=1))


    def zoom_plot(self, iplot):
        if iplot == 1 or iplot == 2 or iplot == 4:
            self.p1.setXRange(0, self.ops["Lx"])
            self.p1.setYRange(0, self.ops["Ly"])
            self.p2.setXRange(0, self.ops["Lx"])
            self.p2.setYRange(0, self.ops["Ly"])
        else:
            self.p3.setXRange(0, self.Fcell.shape[1])
            self.p3.setYRange(self.fmin, self.fmax)
        self.show()

    def run_suite2p(self):
        RW = gui.RunWindow(self)
        RW.show()

    def manual_label(self):
        MW = drawroi.ROIDraw(self)
        MW.show()

    def vis_window(self):
        VW = visualize.VisWindow(self)
        VW.show()

    def reg_window(self):
        RW = reggui.BinaryPlayer(self)
        RW.show()

    def regPC_window(self):
        RW = reggui.PCViewer(self)
        RW.show()

    def load_dialog(self):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Open stat.npy", filter="stat.npy"
        )
        self.fname = name[0]
        self.load_proc()

    def load_proc(self):
        name = self.fname
        print(name)
        try:
            stat = np.load(name, allow_pickle=True)
            ypix = stat[0]["ypix"]
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print('ERROR: this is not a stat.npy file :( '
                  '(needs stat[n]["ypix"]!)')
            stat = None
        if stat is not None:
            basename, fname = os.path.split(name)
            goodfolder = True
            try:
                Fcell = np.load(basename + "/F.npy")
                Fneu = np.load(basename + "/Fneu.npy")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print(
                    "ERROR: there are no fluorescence traces in this folder "
                    "(F.npy/Fneu.npy)"
                )
                goodfolder = False
            try:
                Spks = np.load(basename + "/spks.npy")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("there are no spike deconvolved traces in this folder "
                      "(spks.npy)")
                goodfolder = False
            try:
                ops = np.load(basename + "/ops.npy", allow_pickle=True).item()
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("ERROR: there is no ops file in this folder (ops.npy)")
                goodfolder = False
            try:
                iscell = np.load(basename + "/iscell.npy")
                probcell = iscell[:, 1]
                iscell = iscell[:, 0].astype(np.bool)
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("no manual labels found (iscell.npy)")
                if goodfolder:
                    NN = Fcell.shape[0]
                    iscell = np.ones((NN,), np.bool)
                    probcell = np.ones((NN,), np.float32)
            try:
                redcell = np.load(basename + "/redcell.npy")
                probredcell = redcell[:,1].copy()
                redcell = redcell[:,0].astype(np.bool)
                self.hasred = True
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print("no channel 2 labels found (redcell.npy)")
                self.hasred = False
                if goodfolder:
                    NN = Fcell.shape[0]
                    redcell = np.zeros((NN,), np.bool)
                    probredcell = np.zeros((NN,), np.float32)

            if goodfolder:
                self.basename = basename
                self.stat = stat
                self.ops = ops
                self.Fcell = Fcell
                self.Fneu = Fneu
                self.Spks = Spks
                self.iscell = iscell
                self.probcell = probcell
                self.redcell = redcell
                self.probredcell = probredcell
                for n in range(len(self.stat)):
                    self.stat[n]['chan2_prob'] = self.probredcell[n]
                self.make_masks_and_buttons()
                self.loaded = True
            else:
                print("stat.npy found, but other files not in folder")
                Text = ("stat.npy found, but other files missing, "
                        "choose another?")
                self.load_again(Text)
        else:
            Text = "Incorrect file, not a stat.npy, choose another?"
            self.load_again(Text)

    def load_behavior(self):
        """
        Loads behavior one or multi dimension data in the GUI.
        Valid file formats: .npy - numpy arrays
                            .csv - files, comma separated with one line header
                            which will be skipped. For now...
        """

        name = QtGui.QFileDialog.getOpenFileName(
            self, filter="File types (*.csv *.npy)")

        name = name[0]
        pathTo, ext = os.path.splitext(name)
        if ext == '.npy':
            name = np.load(name)
        if ext == '.csv':
            with open(name) as f:
                reader = csv.reader(f)
                columns = next(reader)
                colmap = dict(zip(columns, range(len(columns))))
            arr = np.matrix(np.loadtxt(name, delimiter=",", skiprows=1))
            name = np.array(arr)
        if ext != '.csv' or '.npy':
            print('---- Input only can be .csv or .npy ----')

        # debugging statements
        # name = '/groups/hackathon/data/guest3/sampleData/TSeries-06262017-1330-1minrec-002/suite2p/fakeData.csv'
        # print(name)
        # print(ext)
        # print(name[0])
        # print(type(name))
        # print(name.shape)

        bloaded = False
        try:
            beh = name
            # set_trace()
            if beh.ndim==1:
                beh=beh[:,np.newaxis]
            if beh.shape[0] == self.Fcell.shape[1]:
                self.bloaded = True
                beh_time = np.arange(0, self.Fcell.shape[1])
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print("ERROR: the behavioral data does not have same sampling frequency")
        if self.bloaded:
            for ind in range(beh.shape[1]):
                beh[:,ind] -= beh[:,ind].min()
                beh[:,ind] /= beh[:,ind].max()
            self.beh = beh
            self.beh_time=beh_time
            b = len(self.colors)
            self.colorbtns.button(b).setEnabled(True)
            self.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
            fig.beh_masks(self)
            fig.plot_trace(self)
            self.show()
        else:
            print("ERROR: this is not a 1D array with length of data")

    def save_mat(self):
        matpath = os.path.join(self.basename,'Fall.mat')
        io.savemat(matpath, {'stat': self.stat,
                             'ops': self.ops,
                             'F': self.Fcell,
                             'Fneu': self.Fneu,
                             'spks': self.Spks,
                             'iscell': np.concatenate((self.iscell[:,np.newaxis],
                                                       self.probcell[:,np.newaxis]), axis=1)
                             })
        print('saving to mat')

    def load_custom_mask(self):
        name = QtGui.QFileDialog.getOpenFileName(
            self, "Open *.npy", filter="*.npy"
        )
        name = name[0]
        cloaded = False
        try:
            mask = np.load(name)
            mask = mask.flatten()
            if mask.size == self.Fcell.shape[0]:
                self.cloaded = True
        except (ValueError, KeyError, OSError,
                RuntimeError, TypeError, NameError):
            print("ERROR: this is not a 1D array with length of data")
        if self.cloaded:
            self.custom_mask = mask
            fig.custom_masks(self)
            M = fig.draw_masks(self)
            b = len(self.colors)+1
            self.colorbtns.button(b).setEnabled(True)
            self.colorbtns.button(b).setStyleSheet(self.styleUnpressed)
            self.colorbtns.button(b).setChecked(True)
            self.colorbtns.button(b).press(self,b)
            #fig.plot_masks(self,M)
            #fig.plot_colorbar(self,bid)
            self.show()
        else:
            print("ERROR: this is not a 1D array with length of # of ROIs")


    def load_again(self, Text):
        tryagain = QtGui.QMessageBox.question(
            self, "ERROR", Text, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
        )
        if tryagain == QtGui.QMessageBox.Yes:
            self.load_dialog()

    def load_classifier(self):
        name = QtGui.QFileDialog.getOpenFileName(self, "Open File")
        if name:
            classgui.load(self, name[0])
            self.class_activated()
        else:
            print("no classifier")

    def load_s2p_classifier(self):
        classgui.load(self, self.classorig)
        self.class_file()
        self.saveDefault.setEnabled(True)

    def load_default_classifier(self):
        classgui.load(
            self,
            os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "classifiers/classifier_user.npy",
            ),
        )
        self.class_activated()

    def class_file(self):
        if self.classfile == os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "classifiers/classifier_user.npy",
        ):
            cfile = "default classifier"
        elif self.classfile == os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "classifiers/classifier.npy"
        ):
            cfile = "suite2p classifier"
        else:
            cfile = self.classfile
        cstr = "<font color='white'>" + cfile + "</font>"
        self.classLabel.setText(cstr)

    def class_activated(self):
        self.class_file()
        self.saveDefault.setEnabled(True)
        self.addtoclass.setStyleSheet(self.styleUnpressed)
        self.addtoclass.setEnabled(True)

    def class_default(self):
        dm = QtGui.QMessageBox.question(
            self,
            "Default classifier",
            "Are you sure you want to overwrite your default classifier?",
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
        )
        if dm == QtGui.QMessageBox.Yes:
            classfile = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "classifiers/classifier_user.npy",
            )
            classgui.save_model(classfile, self.model.stats, self.model.iscell, self.model.keys)

    def reset_default(self):
        dm = QtGui.QMessageBox.question(
            self,
            "Default classifier",
            ("Are you sure you want to reset the default classifier "
             "to the built-in suite2p classifier?"),
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
        )
        if dm == QtGui.QMessageBox.Yes:
            classfile = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "classifiers/classifier_user.npy",
            )
            shutil.copy(self.classorig, classfile)

    # def save_gui_data(self):
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


def run(statfile=None):
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "logo/logo.png"
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainW(statfile=statfile)
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
