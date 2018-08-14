from PyQt5 import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from scipy.ndimage import filters
from scipy.ndimage import gaussian_filter
from scipy import ndimage
import math
from matplotlib.colors import hsv_to_rgb

### custom QDialog which allows user to fill in ops and run suite2p!
class VisWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(VisWindow, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(50,50,1100,600)
        self.setWindowTitle('Visualize deconvolved data')
        self.win = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.win.setLayout(self.l0)
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.addItem("PC")
        self.comboBox.addItem("embed")
        self.comboBox.activated[str].connect(self.sorting)
        self.l0.addWidget(self.comboBox,0,0,1,2)
        self.PCedit = QtGui.QLineEdit(self)
        self.PCedit.setValidator(QtGui.QIntValidator(0,10000))
        self.PCedit.setText('0')
        self.PCedit.setFixedWidth(35)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.sorting)
        self.l0.addWidget(QtGui.QLabel('PC: '),1,1,1,1)
        self.l0.addWidget(self.PCedit,1,2,1,1)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.p0 = pg.ImageView(self, name='image')
        self.l0.addWidget(self.p0,0,3,10,10)
        self.p0.show()

    def sorting(self):
        print('sort')
