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

### custom QPushButton class that plots image when clicked
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows up to 1 button to pressed at a time
class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, data1, data2, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid, data1, data2))
        self.show()
    def press(self, parent, bid, data1, data2):
        ischecked  = parent.viewbtns.checkedId()
        waschecked = parent.btnstate[bid]
        for n in range(len(parent.btnstate)):
            parent.btnstate[n] = False
        if ischecked==bid and not waschecked:
            parent.viewbtns.setExclusive(True)
            parent.img1.setImage(data1)
            parent.img2.setImage(data2)
            parent.btnstate[bid]=True
        elif ischecked==bid and waschecked:
            parent.viewbtns.setExclusive(False)
            parent.btnstate[bid]=False
        else:
            parent.viewbtns.setExclusive(True)
            parent.img1.setImage(data1)
            parent.img2.setImage(data2)
            parent.btnstate[bid] = True
        self.setChecked(parent.btnstate[bid])

### Changes colors of ROIs
# button group is exclusive (at least one color is always chosen)
class ColorButton(QtGui.QPushButton):
    def __init__(self, bid, Text, data1, data2, parent=None):
        super(ColorButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid, data1, data2))
        self.show()
    def press(self, parent, bid, data1, data2):
        ischecked  = self.isChecked()
        if ischecked:
            parent.img1.setImage(data1)
            parent.img2.setImage(data2)




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
        # main widget
        cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)
        # ROI CHECKBOX
        checkBox = QtGui.QCheckBox('ROIs on')
        checkBox.move(30,100)
        checkBox.stateChanged.connect(self.plot_neuropil)
        checkBox.toggle()
        self.l0.addWidget(checkBox,0,0,1,1)
        # MAIN PLOTTING AREA
        self.win = pg.GraphicsView()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,0,1,8,12)
        l = pg.GraphicsLayout(border=(100,100,100))
        self.win.setCentralItem(l)
        self.p0 = l.addLabel('stat*.npy',row=0,col=0,colspan=2)
        # cells image
        self.p1 = l.addViewBox(lockAspect=True,name='plot1',row=1,col=0)
        self.img1 = pg.ImageItem()
        self.p1.setMenuEnabled(False)
        data = np.random.random((512,512,3))
        self.img1.setImage(data)
        self.p1.addItem(self.img1)
        # noncells image
        self.p2 = l.addViewBox(lockAspect=True,name='plot2',row=1,col=1)
        self.p2.setMenuEnabled(False)
        self.img2 = pg.ImageItem()
        self.img2.setImage(data)
        self.p2.addItem(self.img2)
        self.p2.autoRange()
        self.p2.setXLink('plot1')
        self.p2.setYLink('plot1')
        # fluorescence trace plot
        p3 = l.addPlot(row=2,col=0,colspan=2)
        x = np.arange(0,20000)
        y = np.random.random((20000,))
        self.trace = p3.plot(x,y,pen='y')
        p3.setMouseEnabled(x=True,y=False)
        p3.enableAutoRange(x=False,y=True)
        # cell clicking enabled in either cell or noncell image
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.show()
        self.win.show()

    def make_masks_and_buttons(self, name):
        data = np.load(name)
        self.p0.setText(name)
        views = ['mean img', 'correlation map','red channel']
        colors = ['random','skewness', 'compactness','aspect ratio','classifier']
        n = 0
        self.viewbtns = QtGui.QButtonGroup(self)
        blabel = QtGui.QLabel(self)
        blabel.setText('Background')
        blabel.resize(blabel.minimumSizeHint())
        blabel.setGeometry(100,10,10,10)
        #blabel.setAlignment(QtCore.AlignLeft)
        #self.l0.addWidget(blabel,1,0,1,1)
        self.btnstate = []
        for names in views:
            btn  = ViewButton(n,names,data,data,self)
            self.viewbtns.addButton(btn,n)
            self.l0.addWidget(btn,n+2,0,1,1)
            self.btnstate.append(False)
            n+=1
        self.colorbtns = QtGui.QButtonGroup(self)
        self.l0.addWidget(QtGui.QLabel('Mask Colors'),5,0,1,1)
        n=0
        for names in colors:
            btn  = ColorButton(n,names,data,data,self)
            if n==0:
                btn.setChecked(True)
            self.colorbtns.addButton(btn,n)
            self.l0.addWidget(btn,n+6,0,1,1)
            self.btnstate.append(False)
            n+=1

    def plot_clicked(self,event):
        flip = False
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
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
        if iplot > 0:
            if event.button()==2:
                flip = True
        if flip:
            print(posx,posy,flip)
        self.posx = posx
        self.posy = posy
        self.flip = flip

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
        if name:
            print(name[0])
            try:
                masks = np.load(name[0])
            except (OSError, RuntimeError, TypeError, NameError):
                print('this is not an npy file :(')
                masks = np.zeros((0,))

            if masks.ndim > 1:
                self.masks = masks
                self.img1.setImage(masks)
                self.img2.setImage(masks)
                self.make_masks_and_buttons(name[0])
                self.selectionMode = True
            else:
                tryagain = QtGui.QMessageBox.question(self, 'error',
                                                    'Incorrect file, choose another?',
                                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
                if tryagain == QtGui.QMessageBox.Yes:
                    self.load_proc()
                else:
                    pass


    def file_save(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Save File')
        file = open(name,'w')
        file.write('boop')
        file.close()

    # different mask views
    #def mask_view(self):



    #self.btn = QtGui.QPushButton('mean image (M)', self)
    #self.meanBtn.setCheckable(True)
    #self.meanBtn.clicked.connect(newwindow)
    #self.meanBtn.resize(btn.minimumSizeHint())
    #btn.move(10,60)
    #self.show()

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
