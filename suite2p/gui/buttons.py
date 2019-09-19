from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np


# minimize view
def make_cellnotcell(parent):
    parent.sizebtns = QtGui.QButtonGroup(parent)
    b = 0
    labels = [" cells", " both", " not cells"]
    for l in labels:
        btn = SizeButton(b, l, parent)
        parent.sizebtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0, 14 + 2 * b, 1, 2)
        btn.setEnabled(False)
        if b == 1:
            btn.setEnabled(True)
        b += 1
    parent.sizebtns.setExclusive(True)

###### --------- QUADRANT AND VIEW AND COLOR BUTTONS ---------- #############
# quadrant buttons
def make_quadrants(parent):
    parent.quadbtns = QtGui.QButtonGroup(parent)
    for b in range(9):
        btn = QuadButton(b, ' '+str(b+1), parent)
        parent.quadbtns.addButton(btn, b)
        parent.l0.addWidget(btn, 0 + parent.quadbtns.button(b).ypos, 29 + parent.quadbtns.button(b).xpos, 1, 1)
        btn.setEnabled(False)
        b += 1
    parent.quadbtns.setExclusive(True)


### custom QPushButton class for quadrant plotting
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows only 1 button to pressed at a time
class QuadButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(QuadButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)
        self.xpos = bid%3
        self.ypos = int(np.floor(bid/3))
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in range(9):
            if parent.quadbtns.button(b).isEnabled():
                parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        self.xrange = np.array([self.xpos-.15, self.xpos+1.15]) * parent.ops['Lx']/3
        self.yrange = np.array([self.ypos-.15, self.ypos+1.15]) * parent.ops['Ly']/3
        # change the zoom
        parent.p1.setXRange(self.xrange[0], self.xrange[1])
        parent.p1.setYRange(self.yrange[0], self.yrange[1])
        parent.p2.setXRange(self.xrange[0], self.xrange[1])
        parent.p2.setYRange(self.yrange[0], self.yrange[1])
        parent.p2.setXLink("plot1")
        parent.p2.setYLink("plot1")
        parent.show()


# size of view
class SizeButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(SizeButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in parent.sizebtns.buttons():
            b.setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        ts = 100
        if bid==0:
            parent.p2.linkView(parent.p2.XAxis,view=None)
            parent.p2.linkView(parent.p2.YAxis,view=None)
            parent.win.ci.layout.setColumnStretchFactor(0,ts)
            parent.win.ci.layout.setColumnStretchFactor(1,0)
        elif bid==1:
            parent.win.ci.layout.setColumnStretchFactor(0,ts)
            parent.win.ci.layout.setColumnStretchFactor(1,ts)
            parent.p2.setXLink('plot1')
            parent.p2.setYLink('plot1')
        elif bid==2:
            parent.p2.linkView(parent.p2.XAxis,view=None)
            parent.p2.linkView(parent.p2.YAxis,view=None)
            parent.win.ci.layout.setColumnStretchFactor(0,0)
            parent.win.ci.layout.setColumnStretchFactor(1,ts)
        # only enable selection buttons when not in 'both' view
        if bid!=1:
            if parent.ops_plot['color']!=0:
                for btn in parent.topbtns.buttons():
                    btn.setStyleSheet(parent.styleUnpressed)
                    btn.setEnabled(True)
            else:
                parent.topbtns.button(0).setStyleSheet(parent.styleUnpressed)
                parent.topbtns.button(0).setEnabled(True)
        else:
            parent.ROI_remove()
            for btn in parent.topbtns.buttons():
                btn.setEnabled(False)
                btn.setStyleSheet(parent.styleInactive)
        parent.win.show()
        parent.show()

# selection of top neurons
class TopButton(QtGui.QPushButton):
    def __init__(self, bid, parent=None):
        super(TopButton,self).__init__(parent)
        text = [' draw selection', ' select top n', ' select bottom n']
        self.setText(text[bid])
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        if not parent.sizebtns.button(1).isChecked():
            if parent.ops_plot['color']==0:
                for b in [1,2]:
                    parent.topbtns.button(b).setEnabled(False)
                    parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
            else:
                for b in [1,2]:
                    parent.topbtns.button(b).setEnabled(True)
                    parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
        else:
            for b in range(3):
                parent.topbtns.button(b).setEnabled(False)
                parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
        self.setStyleSheet(parent.stylePressed)
        if bid==0:
            parent.ROI_selection()
        else:
            parent.top_selection(bid)
