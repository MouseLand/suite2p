"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from qtpy import QtGui

"""        QToolTip { 
                            background-color: black; 
                            color: white; 
                            border: black solid 1px
                            }"""

def stylesheet():
    return """
        QToolTip {font-size: 10pt; font: Arial;}
        QLineEdit {border: 1px solid rgb(80, 80, 80); font-size: 10pt; font: Arial;}
        QLabel {font-size: 10pt; font: Arial;}
        QPushButton {font-size: 9pt; font: Arial; font-weight: bold;}
        QComboBox {font-size: 10pt; font: Arial;}
        QCheckBox {font-size: 10pt; font: Arial;}
        QPushButton:pressed {Text-align: center; 
                             background-color: rgb(150,50,150); 
                             border-color: white;
                             color:white;}
                            QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:!pressed {Text-align: center; 
                               background-color: rgb(50,50,50);
                                border-color: white;
                               color:white;}
                                QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
        QPushButton:disabled {Text-align: center; 
                             background-color: rgb(30,30,30);
                             border-color: white;
                              color:rgb(80,80,80);}
                               QToolTip { 
                           background-color: black; 
                           color: white; 
                           border: black solid 1px
                           }
                        
        """


class DarkPalette(QtGui.QPalette):
    """Class that inherits from pyqtgraph.QtGui.QPalette and renders dark colours for the application.
    (from pykilosort/kilosort4)
    """

    def __init__(self):
        QtGui.QPalette.__init__(self)
        self.setup()

    def setup(self):
        self.setColor(QtGui.QPalette.Window, QtGui.QColor(50, 50, 50))
        self.setColor(QtGui.QPalette.WindowText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Base, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Text, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.Button, QtGui.QColor(40, 40, 40))
        self.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(255, 255, 255))
        self.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
        self.setColor(QtGui.QPalette.Link, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        self.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,
                      QtGui.QColor(128, 128, 128))
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.ButtonText,
            QtGui.QColor(128, 128, 128),
        )
        self.setColor(
            QtGui.QPalette.Disabled,
            QtGui.QPalette.WindowText,
            QtGui.QColor(128, 128, 128),
        )

def boundary(ypix, xpix):
    """ returns pixels of mask that are on the exterior of the mask """
    ypix = np.expand_dims(ypix.flatten(), axis=1)
    xpix = np.expand_dims(xpix.flatten(), axis=1)
    npix = ypix.shape[0]
    if npix > 0:
        msk = np.zeros((np.ptp(ypix) + 6, np.ptp(xpix) + 6), "bool")
        msk[ypix - ypix.min() + 3, xpix - xpix.min() + 3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)
        k = np.ones((3, 3), dtype=int)  # for 4-connected
        k = np.zeros((3, 3), dtype=int)
        k[1] = 1
        k[:, 1] = 1  # for 8-connected
        out = binary_dilation(msk == 0, k) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext + ypix.min() - 3, xext + xpix.min() - 3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))
    return yext, xext


def circle(med, r):
    """ returns pixels of circle with radius 1.25x radius of cell (r) """
    theta = np.linspace(0.0, 2 * np.pi, 100)
    x = r * 1.25 * np.cos(theta) + med[0]
    y = r * 1.25 * np.sin(theta) + med[1]
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    return x, y
