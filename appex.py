from PyQt5 import QtGui
import sys

import pyqtgraph as pg
import numpy as np

class ViewData(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ViewData, self).__init__(parent)
        self.widget = QtGui.QWidget()
        self.widget.setLayout(QtGui.QHBoxLayout())

        imv = pg.ImageView()
        imagedata = np.random.rand(256,256)
        imv.setImage(imagedata)

        self.widget.layout().addWidget(imv)
        self.setCentralWidget(self.widget)
        self.show()


def main():
    app = QtGui.QApplication(sys.argv)
    vd = ViewData()
    vd.show()
    app.exec_()

if __name__ == '__main__':
    main()
