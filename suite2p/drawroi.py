from PyQt5 import QtGui, QtCore
from suite2p import roiextract
import pyqtgraph as pg
import os
import sys
import numpy as np
from matplotlib.colors import hsv_to_rgb

class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleUnpressed)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in range(len(parent.views)):
            if parent.viewbtns.button(b).isEnabled():
                parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        im0 = np.zeros((parent.Ly, parent.Lx), np.float32)
        if bid==0:
            parent.img0.setImage(parent.ops['meanImg'])
        elif bid==1:
            parent.img0.setImage(parent.ops['meanImgE'])
        elif bid==2:
            im0[parent.ops['yrange'][0]:parent.ops['yrange'][1],
                parent.ops['xrange'][0]:parent.ops['xrange'][1]] = parent.ops['Vcorr']
            parent.img0.setImage(im0)
        else:
            im0[parent.ops['yrange'][0]:parent.ops['yrange'][1],
                parent.ops['xrange'][0]:parent.ops['xrange'][1]] = parent.ops['max_proj']
            parent.img0.setImage(im0)
        parent.win.show()
        parent.show()


class ROIDraw(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ROIDraw, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(70,70,1400,800)
        self.setWindowTitle('extract ROI activity')
        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")

        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win,3,0,13,14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot(row=0,col=1)
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        self.p1.scene().sigMouseMoved.connect(self.mouse_moved)

        self.p0 = self.win.addViewBox(name='plot1',lockAspect=True,row=0,col=0,invertY=True)
        self.img0=pg.ImageItem()
        self.p0.addItem(self.img0)
        self.img0.setImage(parent.ops['meanImg'])
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.instructions = QtGui.QLabel("Add ROI: button / SHIFT+CLICK")
        self.instructions.setStyleSheet("color: white;")
        self.l0.addWidget(self.instructions,0,0,1,4)
        self.instructions = QtGui.QLabel("Remove last clicked ROI: DELETE")
        self.instructions.setStyleSheet("color: white;")
        self.l0.addWidget(self.instructions,1,0,1,4)

        self.addROI = QtGui.QPushButton("add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(lambda: self.add_ROI(pos=None))
        self.addROI.setEnabled(True)
        self.addROI.setFixedWidth(60)
        self.addROI.setStyleSheet(self.styleUnpressed)
        self.l0.addWidget(self.addROI,2,0,1,1)
        lbl = QtGui.QLabel('diameter:')
        lbl.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        lbl.setStyleSheet("color: white;")
        lbl.setFixedWidth(60)
        self.l0.addWidget(lbl,2,1,1,1)
        self.diam = QtGui.QLineEdit(self)
        self.diam.setValidator(QtGui.QIntValidator(0, 10000))
        self.diam.setText("12")
        self.diam.setFixedWidth(35)
        self.diam.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.diam, 2,2,1,1)
        self.procROI = QtGui.QPushButton("extract ROIs")
        self.procROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.procROI.clicked.connect(lambda: self.proc_ROI(parent))
        self.procROI.setStyleSheet(self.styleUnpressed)
        self.procROI.setEnabled(True)
        self.l0.addWidget(self.procROI,3,0,1,3)
        self.l0.addWidget(QtGui.QLabel(""), 4, 0, 1, 3)
        self.l0.setRowStretch(4, 1)

        # view buttons
        self.views = ["W: mean img",
                      "E: mean img (enhanced)",
                      "R: correlation map",
                      "T: max projection"]
        b = 0
        self.viewbtns = QtGui.QButtonGroup(self)
        for names in self.views:
            btn = ViewButton(b, "&" + names, self)
            self.viewbtns.addButton(btn, b)
            self.l0.addWidget(btn, b, 3, 1, 2)
            btn.setEnabled(True)
            b += 1
        b=0
        self.viewbtns.button(b).setChecked(True)
        self.viewbtns.button(b).setStyleSheet(self.stylePressed)

        self.l0.addWidget(QtGui.QLabel("neuropil"), 13,13,1,1)

        self.ops = parent.ops
        self.Ly = self.ops['Ly']
        self.Lx = self.ops['Lx']
        self.ROIs = []
        self.cell_pos = []
        self.extracted = False

    def mouse_moved(self, pos):
        if self.extracted:
            if self.p1.sceneBoundingRect().contains(pos):
                x = self.p1.vb.mapSceneToView(pos).x()
                y = self.p1.vb.mapSceneToView(pos).y()
                self.ineuron = self.nROIs - y + 1
                #print(self.ineuron)


    def keyPressEvent(self, event):
        if event.modifiers() != QtCore.Qt.ControlModifier and event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Delete:
                self.ROIs[self.iROI].remove(self)
            elif event.key() == QtCore.Qt.Key_W:
                self.viewbtns.button(0).setChecked(True)
                self.viewbtns.button(0).press(self, 0)
            elif event.key() == QtCore.Qt.Key_E:
                self.viewbtns.button(1).setChecked(True)
                self.viewbtns.button(1).press(self, 1)
            elif event.key() == QtCore.Qt.Key_R:
                self.viewbtns.button(2).setChecked(True)
                self.viewbtns.button(2).press(self, 2)
            elif event.key() == QtCore.Qt.Key_T:
                self.viewbtns.button(3).setChecked(True)
                self.viewbtns.button(3).press(self, 3)

    def add_ROI(self, pos=None):
        self.iROI = len(self.ROIs)
        self.nROIs = len(self.ROIs)
        self.ROIs.append(sROI(iROI=self.nROIs, parent=self, pos=pos, diameter=int(self.diam.text())))
        self.ROIs[-1].position(self)
        self.nROIs += 1

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
        for x in items:
            if x == self.img0:
                pos = self.p0.mapSceneToView(event.scenePos())
                posy = pos.x()
                posx = pos.y()
                if event.modifiers() == QtCore.Qt.ShiftModifier:
                    self.add_ROI(pos=np.array([posx-5,posy-5,
                                int(self.diam.text()),int(self.diam.text())]))
                if event.double():
                    self.p0.setXRange(0, self.Lx)
                    self.p0.setYRange(0, self.Ly)
            elif x==self.p1:
                if event.double():
                    self.p1.setXRange(0,self.trange.size)
                    self.p1.setYRange(self.fmin, self.fmax)

    def proc_ROI(self, parent):
        stat0 = []
        if self.extracted:
            for t,s in zip(self.scatter, self.tlabel):
                self.p0.removeItem(s)
                self.p0.removeItem(t)
        self.scatter = []
        self.tlabel = []
        for n in range(self.nROIs):
            ellipse = self.ROIs[n].ellipse
            yrange = self.ROIs[n].yrange
            xrange = self.ROIs[n].xrange
            x,y = np.meshgrid(xrange, yrange)
            ypix = y[ellipse].flatten()
            xpix = x[ellipse].flatten()
            lam = np.ones(ypix.shape)
            stat0.append({'ypix': ypix, 'xpix': xpix, 'lam': lam, 'npix': ypix.size})
            self.tlabel.append(pg.TextItem(str(n), self.ROIs[n].color, anchor=(0, 0)))
            self.tlabel[-1].setPos(xpix.mean(), ypix.mean())
            self.p0.addItem(self.tlabel[-1])
            self.scatter.append(pg.ScatterPlotItem([xpix.mean()], [ypix.mean()],
                                                pen=self.ROIs[n].color, symbol='+'))
            self.p0.addItem(self.scatter[-1])
        F, Fneu, F_chan2, Fneu_chan2, ops, stat = roiextract.masks_and_traces(parent.ops, stat0)
        self.Fcell = F
        self.Fneu = Fneu
        self.plot_trace()
        self.extracted = True

    def plot_trace(self):
        self.trange = np.arange(0, self.Fcell.shape[1])
        self.p1.clear()
        kspace = 1.0
        ax = self.p1.getAxis('left')
        favg = 0
        k = self.nROIs - 1
        ttick = list()
        for n in range(self.nROIs):
            f = self.Fcell[n,:]
            fneu = self.Fneu[n,:]
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            f = (f - fmin) / (fmax - fmin)
            fneu = (fneu - fmin) / (fmax - fmin)
            rgb = self.ROIs[n].color
            self.p1.plot(self.trange,f+k*kspace,pen=rgb)
            self.p1.plot(self.trange,fneu+k*kspace,pen='r')
            ttick.append((k*kspace+f.mean(), str(n)))
            k-=1
        self.fmax = (self.nROIs-1)*kspace + 1
        self.fmin = 0
        ax.setTicks([ttick])
        self.p1.setXRange(0, self.Fcell.shape[1])
        self.p1.setYRange(self.fmin, self.fmax)

class sROI():
    def __init__(self, iROI, parent=None, pos=None, diameter=None, color=None,
                 yrange=None, xrange=None):
        # what type of ROI it is

        self.iROI = iROI
        self.xrange = xrange
        self.yrange = yrange
        if color is None:
            self.color = hsv_to_rgb(np.array([np.random.rand() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * self.color)
        else:
            self.color = color
        if pos is None:
            view = parent.p0.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            if diameter is None:
                dx = np.maximum(3, np.minimum(dx, parent.Ly*0.2))
                dy = np.maximum(3, np.minimum(dy, parent.Lx*0.2))
            else:
                dx = diameter
                dy = diameter
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = pos[0]
            imx = pos[1]
            dy = pos[2]
            dx = pos[3]

        self.draw(parent, imy, imx, dy, dx)
        self.position(parent)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI([imx, imy], [dx, dy], pen=roipen, removable=True)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        for i in range(len(parent.ROIs)):
            if i > self.iROI:
                parent.ROIs[i].iROI -= 1
        del parent.ROIs[self.iROI]
        parent.iROI = min(len(parent.ROIs)-1, max(0, parent.iROI))
        parent.nROIs -= 1
        parent.win.show()
        parent.show()

    def position(self, parent):
        parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        pos = parent.p0.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        yrange += int(np.floor(sizey/2)) + 1
        # what is ellipse circling?
        br = self.ROI.boundingRect()
        ellipse = np.zeros((yrange.size, xrange.size), np.bool)
        x,y = np.meshgrid(np.arange(0,xrange.size,1), np.arange(0,yrange.size,1))
        ellipse = ((y - br.center().y())**2 / (br.height()/2)**2 +
                    (x - br.center().x())**2 / (br.width()/2)**2) <= 1


        ellipse = ellipse[:,np.logical_and(xrange >= 0, xrange < parent.Lx)]
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.Lx)]
        ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.Ly), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.Ly)]

        # ellipse = lambda x,y: (((x+0.5)/(w/2.)-1)**2+ ((y+0.5)/(h/2.)-1)**2)**0.5 < 1, (w, h))
        self.ellipse = ellipse
        self.xrange = xrange
        self.yrange = yrange
