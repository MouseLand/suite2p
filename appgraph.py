from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import pyqtgraph as pg
import sys
import numpy as np

def newwindow():
    print('meanimg')
    LoadW = QtGui.QWindow()
    LoadW.show()



def imageHoverEvent(event, data):
    """Show the position, pixel, and value under the mouse cursor.
    """
    if event.isExit():
        p1.setTitle("")
        return
    pos = event.pos()
    i, j = pos.y(), pos.x()
    i = int(np.clip(i, 0, data.shape[0] - 1))
    j = int(np.clip(j, 0, data.shape[1] - 1))
    val = data[i, j]
    ppos = img.mapToParent(pos)
    x, y = ppos.x(), ppos.y()
    p1.setTitle("pos: (%0.1f, %0.1f)  pixel: (%d, %d)  value: %g" % (x, y, i, j, val))

def mouseMoved(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if p3.sceneBoundingRect().contains(pos):
        mousePoint = p3.vb.mapSceneToView(evt[0])
        print(mousePoint)
        #index = int(mousePoint.x())
        #if index > 0 and index < len(data1):
        #       label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        #vLine.setPos(mousePoint.x())
        #hLine.setPos(mousePoint.y())
        #print(mousePoint.x())
        #print(mousePoint.y())

class MainW(QtGui.QMainWindow):
    resized = QtCore.pyqtSignal()
    def __init__(self):
        super(MainW, self).__init__()
        self.setGeometry(50,50,1600,1000)
        self.setWindowTitle('suite2p')
        #self.setStyleSheet("QMainWindow {background: 'black';}")
        self.selectionMode = False
        self.masks = np.random.random((512,512,3))
        self.resized.connect(self.windowsize)
        ### menu bar options
        # load processed data
        loadProc = QtGui.QAction('&Load processed data', self)
        loadProc.setShortcut('Ctrl+L')
        loadProc.setStatusTip('load processed data in suite2p format')
        loadProc.triggered.connect(self.load_proc)
        # load masks
        loadMask = QtGui.QAction('&Load masks and extract traces', self)
        loadMask.setShortcut('Ctrl+L')
        loadMask.setStatusTip('load mask pixels in suite2p format')
        # save file
        saveFile = QtGui.QAction('&Save', self)
        saveFile.setShortcut('Ctrl+S')
        saveFile.triggered.connect(self.file_save)
        # make menuBar!
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu('&File')
        file_menu.addAction(loadProc)
        file_menu.addAction(loadMask)
        file_menu.addAction(saveFile)

        x = np.linspace(-50, 50, 1000)
        y = np.sin(x) / x

        ### main widget with plots
        self.win = pg.GraphicsView()
        self.setCentralWidget(self.win)
        l = pg.GraphicsLayout(border=(100,100,100))
        self.win.setCentralItem(l)
        self.win.show()
        l.addLabel('Buttons for views',row=0,col=0)
        p0 = l.addLabel('F*.npy',row=0,col=1,colspan=2)
        # cells image
        self.p1 = l.addViewBox(lockAspect=True,name='plot1',row=1,col=1)
        self.img1 = pg.ImageItem()
        data = np.random.random((512,512,3))
        self.img1.setImage(data)
        self.p1.addItem(self.img1)
        # noncells image
        self.p2 = l.addViewBox(lockAspect=True,name='plot2',row=1,col=2)
        self.img2 = pg.ImageItem()
        self.img2.setImage(data)
        self.p2.addItem(self.img2)
        self.p2.autoRange()
        self.p2.setXLink('plot1')
        self.p2.setYLink('plot1')
        # fluorescence trace plot
        p3 = l.addPlot(row=2,col=1,colspan=2)
        self.trace = p3.plot(x,y,pen='y')
        p3.setMouseEnabled(x=True,y=False)
        p3.enableAutoRange(x=False,y=True)
        # cell clicking enabled in either cell or noncell image
        self.win.scene().sigMouseClicked.connect(self.plotClicked)

        ### checkboxes
        checkBox = QtGui.QCheckBox('ROIs on')
        checkBox.move(30,100)
        checkBox.stateChanged.connect(self.plot_neuropil)
        checkBox.toggle()

        ### buttons for different views
        self.mask_view()

    def plotClicked(self,event):
        flip = 0
        items = self.win.scene().items(event.scenePos())
        posx=0
        posy=0
        goodclick=False
        for x in items:
            if x==self.img1:
                pos = self.p1.mapSceneToView(event.scenePos())
                posx = pos.x()
                posy = pos.y()
                goodclick = True
            elif x==self.img2:
                pos = self.p2.mapSceneToView(event.scenePos())
                posx = pos.x()
                posy = pos.y()
                goodclick = True
        if goodclick:
            if event.double():
                print('double click')
                flip = 1
        print(posx,posy,flip)
        print("Plots:", [x for x in items if isinstance(x, pg.ImageItem)])
        return posx,posy,flip

    def windowsize(self):
        print(10)
    #def resizeEvent(self,event):
    #    self.resized.emit()
    #    return super(MainW,self).resizeEvent(event)

    def plot_neuropil(self,state):
        if state == QtCore.Qt.Checked:
            print('yay')
        else:
            print('boo')

    def load_proc(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        #print(name)
        masks = np.load(name[0])

        if masks.ndim == 2 | (masks.ndim==3 & masks.shape[2]==3):
            self.masks = masks
            self.img1.setImage(masks)
            self.img2.setImage(masks)
            #pg.image(masks)
            self.selectionMode = True
        else:
            tryagain = QtGui.QMessageBox.question(self, 'error',
                                                  'Incorrect file, choose another?',
                                                  QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
            if tryagain == QtGui.QMessageBox.Yes:
                self.load_proc(self)
            else:
                pass

    def file_save(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Save File')
        file = open(name,'w')
        file.write('boop')
        file.close()

    # different mask views
    def mask_view(self):
        btn = QtGui.QPushButton('mean image (M)', self)
        btn.setShortcut('M')
        btn.clicked.connect(newwindow)
        btn.resize(btn.minimumSizeHint())
        btn.move(10,60)
        self.show()

class PlotCells(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        #self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setFocus()
        #self.mpl_toolbar = NavigationToolbar(self, parent)
        FigureCanvas.setSizePolicy(self,
                                   QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # add axes
        self.figure.patch.set_facecolor('black')
        self.ax1 = self.figure.add_axes([0.01,0.2,0.47,0.73])
        self.ax1.axis('off')
        self.ax2 = self.figure.add_axes([0.52,0.2,0.47,0.73], sharex=self.ax1, sharey=self.ax1)
        self.ax2.axis('off')
        self.ax3 = self.figure.add_axes([0.01,0.02,0.98,0.2])
        self.ax3.set_facecolor('black')

    def plot(self, masks1, masks2):
        self.ax1.imshow(masks1)
        self.ax2.imshow(masks2)
        self.ax3.plot(np.random.random((1000,)))

        #ax.set_title('PyQt Matplotlib Example')
        #self.figure.tight_layout()

        self.draw()
        self.show()

    def onclick(self,event):
        if event.inaxes:
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            if self.ax1 is event.inaxes:
                print('plot1')


def run():
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    GUI = MainW()
    #plot = pg.PlotWidget()
    #GUI.setCentralWidget(plot)
    #plot.resize(400,400)
    #plot.sigPointsClicked.connect(plot,meclick)
    sys.exit(app.exec_())

run()

## Define a top-level widget to hold everything
#w = QtGui.QWidget()
#w.setGeometry(0,0,900,500)
#w.setWindowTitle('suite2p')

## Create some widgets to be placed inside
#btn = QtGui.QPushButton('press me')
#text = QtGui.QLineEdit('enter text')
#listw = QtGui.QListWidget()
#plot = pg.PlotWidget()

## Create a grid layout to manage the widgets size and position
#layout = QtGui.QGridLayout()
#w.setLayout(layout)

## Add widgets to the layout in their proper positions
#layout.addWidget(btn, 0, 0)   # button goes in upper-left
#layout.addWidget(text, 1, 0)   # text edit goes in middle-left
#layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
#layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows

## Display the widget as a new window
#w.show()

## Start the Qt event loop
#app.exec_()
