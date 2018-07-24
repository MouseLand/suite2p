from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os

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
            parent.ops_plot[1] = bid
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)
            parent.btnstate[bid]=True
        elif ischecked==bid and waschecked:
            parent.viewbtns.setExclusive(False)
            parent.btnstate[bid]=False
            parent.ops_plot[1] = -1
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)
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
            parent.ops_plot[2] = bid
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)

class MainW(QtGui.QMainWindow):
    resized = QtCore.pyqtSignal()
    def __init__(self):
        super(MainW, self).__init__()
        self.setGeometry(50,50,1600,1000)
        self.setWindowTitle('suite2p')
        #self.setStyleSheet("QMainWindow {background: 'black';}")
        self.loaded = False
        self.ops_plot = []
        self.ops_plot.append(True)
        self.ops_plot.append(-1)
        self.ops_plot.append(0)

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
        self.p0 = l.addLabel('stat*.npy',row=0,col=0,colspan=2)
        # cells image
        self.p1 = l.addViewBox(lockAspect=True,name='plot1',row=1,col=0)
        self.img1 = pg.ImageItem()
        self.p1.setMenuEnabled(False)
        data = np.zeros((512,512,3))
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
        #self.p2.setLimits(minXRange=-50,maxXRange=552,
        #                      minYRange=-50,maxYRange=552)
        #self.p2.setXRange(0,512,padding=0.25)
        #self.p2.setYRange(0,512,padding=0.25)
        # fluorescence trace plot
        self.p3 = l.addPlot(row=2,col=0,colspan=2)
        x = np.arange(0,20000)
        y = np.zeros((20000,))
        self.p3.plot(x,y,pen='b')
        self.p3.setMouseEnabled(x=True,y=False)
        self.p3.enableAutoRange(x=False,y=True)
        # cell clicking enabled in either cell or noncell image


        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.show()
        self.win.show()

    def make_masks_and_buttons(self, name):
        randcols = np.random.random(len(self.stat,))
        ops_plot.append(randcols)
        self.p0.setText(name)
        views = ['mean img', 'correlation map','red channel']
        colors = ['random','skewness', 'compactness','aspect ratio','classifier']
        n = 0
        self.viewbtns = QtGui.QButtonGroup(self)
        vlabel = QtGui.QLabel(self)
        vlabel.setText('Background')
        vlabel.resize(vlabel.minimumSizeHint())
        self.l0.addWidget(vlabel,1,0,1,1)
        self.btnstate = []
        for names in views:
            btn  = ViewButton(n,names,data,data,self)
            self.viewbtns.addButton(btn,n)
            self.l0.addWidget(btn,n+2,0,1,1)
            self.btnstate.append(False)
            n+=1
        self.colorbtns = QtGui.QButtonGroup(self)
        clabel = QtGui.QLabel(self)
        clabel.setText('Colors')
        clabel.resize(clabel.minimumSizeHint())
        self.l0.addWidget(clabel,n+2,0,1,1)
        nv = n+2
        n=0
        for names in colors:
            btn  = ColorButton(n,names,data,data,self)
            if n==0:
                btn.setChecked(True)
                self.color = 0
            self.colorbtns.addButton(btn,n)
            self.l0.addWidget(btn,nv+n+1,0,1,1)
            self.btnstate.append(False)
            n+=1
        self.show()

    def ROIs_on(self,state):
        if state == QtCore.Qt.Checked:
            self.ops_plot[0] = True
            if self.loaded:
                M = fig.draw_masks(self.ops, self.stat, self.ops_plot
                                    self.iscell, self.ichosen)
                self.plot_masks(M)
        else:
            self.ops_plot[0] = False
    def plot_masks(self,M):
        self.img1.setImage(M[0])
        self.img2.setImage(M[1])

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
                self.stat = np.load(name[0])
            except (OSError, RuntimeError, TypeError, NameError):
                print('this is not an npy file :(')
                self.stat=[0]

            if 'ipix' in self.stat[0]:
                self.stat = np.load(name)
                basename, fname = os.path.split(name)

                self.make_masks_and_buttons(name[0])
                self.loaded = True
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

def run():
    ## Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    GUI = MainW()
    sys.exit(app.exec_())

run()
