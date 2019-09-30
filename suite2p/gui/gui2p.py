import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import warnings
import __main__
from . import menus, io, merge, views, buttons, classgui, traces, graphics, masks
import sys
sys.path.append("..")
from EnsemblePursuit.EnsemblePursuitModule.EnsemblePursuitNumpyFast import EnsemblePursuitNumpyFast

def resample_frames(y, x, xt):
    ''' resample y (defined at x) at times xt '''
    ts = x.size / xt.size
    y = gaussian_filter1d(y, np.ceil(ts/2), axis=0)
    f = interp1d(x,y,fill_value="extrapolate")
    yt = f(xt)
    return yt

class MainWindow(QtGui.QMainWindow):
    def __init__(self, statfile=None):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("suite2p (run pipeline or load stat.npy)")
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__main__.__file__)), "logo/logo.png"
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
        self.classfile = os.path.join(os.path.abspath(os.path.dirname(__main__.__file__)),
            "classifiers/classifier_user.npy",
        )
        self.classorig = os.path.join(os.path.abspath(os.path.dirname(__main__.__file__)),
            "classifiers/classifier.npy"
        )
        if not os.path.isfile(self.classfile):
            shutil.copy(self.classorig, self.classfile)
        # check for ops file (for running suite2p)
        self.opsorig = os.path.join(os.path.abspath(os.path.dirname(__main__.__file__)),
                                          'ops/ops.npy')
        self.opsfile = os.path.join(os.path.abspath(os.path.dirname(__main__.__file__)),
                                          'ops/ops_user.npy')
        if not os.path.isfile(self.opsfile):
            shutil.copy(self.opsorig, self.opsfile)

        menus.mainmenu(self)
        menus.classifier(self)
        menus.visualizations(self)
        menus.registration(self)
        menus.mergebar(self)
        menus.plugins(self)

        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)

        # default plot options
        self.ops_plot = {'ROIs_on': True, 'color': 0, 'view': 0,
                         'opacity': [127,255], 'saturation': [0, 255],
                         'colormap': 'hsv'}
        self.rois = {'iROI':0, 'Sroi':0, 'Lam':0, 'LamMean':0, 'LamNorm':0}
        self.colors = {'RGB':0, 'cols':0, 'colorbar':[]}

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QtGui.QWidget()
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        b0 = self.make_buttons()
        self.make_graphics(b0)
        # so they're on top of plot, draw last
        buttons.make_quadrants(self)

        #EnsemblePursuitButtons
        epbuttons=buttons.ensemble_pursuit_buttons(self)

        # initialize merges
        self.merged = []
        self.imerge = [0]
        self.ichosen = 0
        self.rastermap = False
        model = np.load(self.classorig, allow_pickle=True).item()
        self.default_keys = model["keys"]

        # load initial file
        statfile = '/home/flora/Documents/plane0/stat.npy'
        #statfile = '/home/flora/Documents/TX39/stat.npy'
        #statfile = '/media/carsen/DATA1/TIFFS/auditory_cortex/suite2p/plane0/stat.npy'
        if statfile is not None:
            self.fname = statfile
            io.load_proc(self)
            #self.manual_label()

        self.show()
        self.win.show()

    def make_buttons(self):
        # ROI CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBox = QtGui.QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.toggle()
        self.checkBox.stateChanged.connect(self.ROIs_on)
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)

        buttons.make_selection(self)
        buttons.make_cellnotcell(self)
        b0=views.make_buttons(self) # b0 says how many
        b0=masks.make_buttons(self,b0)
        masks.make_colorbar(self, b0)
        b0+=1
        b0=classgui.make_buttons(self, b0)
        b0+=1

        # ------ CELL STATS / ROI SELECTION --------
        # which stats
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
        qlabel.setFont(self.boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, b0, 0, 1, 1)
        self.ROIedit = QtGui.QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, b0, 1, 1, 1)
        b0+=1
        self.ROIstats = []
        self.ROIstats.append(qlabel)
        for k in range(1, len(self.stats_to_show) + 1):
            llabel = QtGui.QLabel(self.stats_to_show[k - 1])
            self.ROIstats.append(llabel)
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].setStyleSheet("color: white;")
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.addWidget(self.ROIstats[k], b0, 0, 1, 2)
            b0+=1
        self.l0.addWidget(QtGui.QLabel(""), b0 , 0, 1, 2)
        self.l0.setRowStretch(b0, 1)
        b0+=2
        b0 = traces.make_buttons(self, b0)

        # zoom to cell CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBoxz = QtGui.QCheckBox("zoom to cell")
        self.checkBoxz.setStyleSheet("color: white;")
        self.zoomtocell = False
        self.checkBoxz.stateChanged.connect(self.zoom_cell)
        self.l0.addWidget(self.checkBoxz,
        b0,15,
        1, 2)
        return b0

    def zoom_cell(self, state):
        if self.loaded:
            if state == QtCore.Qt.Checked:
                self.zoomtocell = True
            else:
                self.zoomtocell = False
            self.update_plot()

    def make_graphics(self, b0):
        ##### -------- MAIN PLOTTING AREA ---------- ####################
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, b0-1, 30)
        layout = self.win.ci.layout
        # --- cells image
        self.p1 = graphics.ViewBox(parent=self, lockAspect=True, name="plot1", border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p1, 0, 0)
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.view1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view1.autoDownsample = False
        self.color1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color1.autoDownsample = False
        self.p1.addItem(self.view1)
        self.p1.addItem(self.color1)
        self.view1.setLevels([0,255])
        self.color1.setLevels([0,255])
        #self.view1.setImage(np.random.rand(500,500,3))
        #x = np.arange(0,500)
        #img = np.concatenate((np.zeros((500,500,3)), 127*(1+np.tile(np.sin(x/100)[:,np.newaxis,np.newaxis],(1,500,1)))),axis=-1)
        #self.color1.setImage(img)
        # --- noncells image
        self.p2 = graphics.ViewBox(parent=self, lockAspect=True, name="plot2", border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p2, 0, 1)
        self.p2.setMenuEnabled(False)
        self.p2.scene().contextMenuItem = self.p2
        self.view2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view2.autoDownsample = False
        self.color2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color2.autoDownsample = False
        self.p2.addItem(self.view2)
        self.p2.addItem(self.color2)
        self.view2.setLevels([0,255])
        self.color2.setLevels([0,255])

        # LINK TWO VIEWS!
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")

        # --- fluorescence trace plot
        self.p3 = graphics.TraceBox(parent=self, invertY=False)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.enableAutoRange(x=True, y=True)
        self.win.addItem(self.p3, row=1, col=0, colspan=2)
        #self.p3 = pg.PlotItem()
        #self.v3.addItem(self.p3)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        layout = self.win.ci.layout
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)
        #self.win.scene().sigMouseClicked.connect(self.plot_clicked)

    def keyPressEvent(self, event):
        if self.loaded:
            if event.modifiers() != QtCore.Qt.ControlModifier and event.modifiers() != QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Return:
                    if event.modifiers()==QtCore.Qt.AltModifier:
                        if len(self.imerge) > 1:
                            merge.do_merge(self)
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
                #Agus
                elif event.key() == QtCore.Qt.Key_N:
                    self.checkBoxd.toggle()
                elif event.key() == QtCore.Qt.Key_B:
                    self.checkBoxn.toggle()
                elif event.key() == QtCore.Qt.Key_V:
                    self.checkBoxt.toggle()
                #
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
                    ctype = self.iscell[self.ichosen]
                    while -1:
                        self.ichosen = (self.ichosen-1)%len(self.stat)
                        if self.iscell[self.ichosen] is ctype:
                            break
                    self.imerge = [self.ichosen]
                    self.ROI_remove()
                    self.update_plot()

                elif event.key() == QtCore.Qt.Key_Right:
                ##Agus
                    self.ROI_remove()
                    ctype = self.iscell[self.ichosen]
                    while 1:
                        self.ichosen = (self.ichosen+1)%len(self.stat)
                        if self.iscell[self.ichosen] is ctype:
                            break
                    self.imerge = [self.ichosen]
                    self.update_plot()
                    self.show()
                ##Agus
                elif event.key() == QtCore.Qt.Key_Up:
                    masks.flip_plot(self)
                    self.ROI_remove()


    def update_plot(self):
        if self.ops_plot['color'] == 7:
            masks.corr_masks(self)
        masks.plot_colorbar(self)
        self.ichosen_stats()
        views.plot_views(self)
        M = masks.draw_masks(self)
        masks.plot_masks(self, M)
        traces.plot_trace(self)
        if self.zoomtocell:
            self.zoom_to_cell()
        self.win.show()
        self.show()

    def export_fig(self):
        self.win.scene().contextMenuItem = self.p1
        self.win.scene().showExportDialog()

    def mode_change(self, i):
        """

            changes the activity mode that is used when multiple neurons are selected
            or in visualization windows like rastermap or for correlation computation!

            activityMode =
            0 : F
            1 : Fneu
            2 : F - 0.7 * Fneu (default)
            3 : spks

            uses binning set by self.bin

        """
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

            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin ** 2).mean(axis=1)**0.5
            self.trange = np.arange(0, self.Fcell.shape[1])
            # if in behavior-view, recompute
            if self.ops_plot['color'] == 8:
                masks.beh_masks(self)
                masks.plot_colorbar(self)
            self.update_plot()

    def top_number_chosen(self):
        self.ntop = int(self.topedit.text())
        if self.loaded:
            if not self.sizebtns.button(1).isChecked():
                for b in [1, 2]:
                    if self.topbtns.button(b).isChecked():
                        self.topbtns.button(b).top_selection(parent)
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

    def select_cells(self, ypix, xpix):
        i = self.ROIplot
        iROI0 = self.rois['iROI'][i, 0, ypix, xpix]
        icells = np.unique(iROI0[iROI0 >= 0])
        self.imerge = []
        for n in icells:
            if (self.rois['iROI'][i, :, ypix, xpix] == n).sum() > 0.6 * self.stat[n]["npix"]:
                self.imerge.append(n)
        if len(self.imerge) > 0:
            self.ichosen = self.imerge[0]
            self.update_plot()
            self.show()

    def number_chosen(self):
        if self.loaded:
            self.ichosen = int(self.ROIedit.text())
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.imerge = [self.ichosen]
            self.update_plot()
            self.show()



    def ROIs_on(self, state):
        if state == QtCore.Qt.Checked:
            self.ops_plot['ROIs_on'] = True
            self.p1.addItem(self.color1)
            self.p2.addItem(self.color2)
        else:
            self.ops_plot['ROIs_on'] = False
            self.p1.removeItem(self.color1)
            self.p2.removeItem(self.color2)
        self.win.show()
        self.show()

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
                    self.update_plot()
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

    def zoom_to_cell(self):
        irange = 0.1 * np.array([self.Ly, self.Lx]).max()
        if len(self.imerge) > 1:
            apix = np.zeros((0,2))
            for i,k in enumerate(self.imerge):
                apix = np.append(apix,
                        np.concatenate((self.stat[k]['ypix'].flatten()[:,np.newaxis],
                                        self.stat[k]['xpix'].flatten()[:,np.newaxis]), axis=1),
                        axis=0)

            imin = apix.min(axis=0)
            imax = apix.max(axis=0)
            icent = apix.mean(axis=0)
            imin[0] = min(icent[0]-irange, imin[0])
            imin[1] = min(icent[1]-irange, imin[1])
            imax[0] = max(icent[0]+irange, imax[0])
            imax[1] = max(icent[1]+irange, imax[1])
        else:
            icent = np.array(self.stat[self.ichosen]['med'])
            imin = icent - irange
            imax = icent + irange
        self.p1.setYRange(imin[0], imax[0])
        self.p1.setXRange(imin[1], imax[1])
        self.p2.setYRange(imin[0], imax[0])
        self.p2.setXRange(imin[1], imax[1])
        self.win.show()
        self.show()

    def add_neuron_to_ensemble(self):
        current_v=np.mean(self.Fbin[self.imerge,:],axis=0)
        cost_vector=np.clip(self.Fbin@current_v,a_min=0,a_max=None)**2
        cost_vector[self.imerge]=-1000
        cell_inds=np.nonzero(self.iscell==True)
        non_cell_inds=np.nonzero(self.iscell==False)
        if self.ichosen in non_cell_inds[0]:
            cost_vector[cell_inds]=-1000
        if self.ichosen in cell_inds[0]:
            cost_vector[non_cell_inds]=-1000
        new_neuron=np.argmax(cost_vector)
        self.imerge.append(new_neuron)
        self.update_plot()
        self.show()

    def mapping_sel_to_entire_arr(self,cel_inds_to_sel):
        i=0
        ix_dict={}
        for j in range(self.Fbin.shape[0]):
            if j in cel_inds_to_sel:
                ix_dict[i]=j
                i+=1
        return ix_dict



    def fit_one_ensemble(self):
        starting_v=np.mean(self.Fbin[self.imerge,:],axis=0)
        cell_inds=np.nonzero(self.iscell==True)
        non_cell_inds=np.nonzero(self.iscell==False)
        if self.ichosen in non_cell_inds[0]:
            cell_inds_not_to_select=np.sort(list(cell_inds[0])+self.imerge)
            #print(cell_inds_not_to_select)
            cel_inds_to_sel=[i for i in range(self.Fbin.shape[0]) if i not in cell_inds_not_to_select]
            #print(cel_inds_to_sel)
            X=self.Fbin[cel_inds_to_sel,:]
            ix_dict=self.mapping_sel_to_entire_arr(cel_inds_to_sel)
        if self.ichosen in cell_inds[0]:
            cell_inds_not_to_select=np.sort(list(non_cell_inds[0])+self.imerge)
            #print(cell_inds_not_to_select)
            cel_inds_to_sel=[i for i in range(self.Fbin.shape[0]) if i not in cell_inds_not_to_select]
            X=self.Fbin[cel_inds_to_sel,:]
            #print(cel_inds_to_sel)
            ix_dict=self.mapping_sel_to_entire_arr(cel_inds_to_sel)
        print(X.shape)
        np.save('Fbin.npy',X)
        starting_v=np.mean(self.Fbin[self.imerge,:],axis=0)
        options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
        ep_np=EnsemblePursuitNumpyFast(n_ensembles=1,lambd=0.01,options_dict=options_dict)
        starting_v=np.mean(self.Fbin[self.imerge,:],axis=0)
        C=X@X.T
        selected_neurons=ep_np.fit_one_ensemble_suite2p(X,C,starting_v)
        print(selected_neurons)
        sel=np.nonzero(selected_neurons==1)
        cells_to_draw=[]
        print(list(sel[0]))
        for cell in list(sel[0]):
            cells_to_draw.append(ix_dict[cell])
        self.imerge=self.imerge+cells_to_draw
        self.update_plot()
        self.show()

    def compute_seed(self):
        cell_inds=np.nonzero(self.iscell==True)
        non_cell_inds=np.nonzero(self.iscell==False)
        if self.ichosen in non_cell_inds[0]:
            cell_inds_not_to_select=np.sort(list(cell_inds[0]))
            #print(cell_inds_not_to_select)
            cel_inds_to_sel=[i for i in range(self.Fbin.shape[0]) if i not in cell_inds_not_to_select]
            #print(cel_inds_to_sel)
            X=self.Fbin[cel_inds_to_sel,:]
            ix_dict=self.mapping_sel_to_entire_arr(cel_inds_to_sel)
        if self.ichosen in cell_inds[0]:
            cell_inds_not_to_select=np.sort(list(non_cell_inds[0]))
            #print(cell_inds_not_to_select)
            cel_inds_to_sel=[i for i in range(self.Fbin.shape[0]) if i not in cell_inds_not_to_select]
            X=self.Fbin[cel_inds_to_sel,:]
            #print(cel_inds_to_sel)
            ix_dict=self.mapping_sel_to_entire_arr(cel_inds_to_sel)
        options_dict={'seed_neuron_av_nr':100,'min_assembly_size':8}
        ep_np=EnsemblePursuitNumpyFast(n_ensembles=1,lambd=0.01,options_dict=options_dict)
        C=X@X.T
        seed=ep_np.repeated_seed(C,-1)
        print('seed',seed)
        self.imerge=[ix_dict[seed]]
        self.update_plot()
        self.show()






def run(statfile=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__main__.__file__)), "logo/logo.png"
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainWindow(statfile=statfile)
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
