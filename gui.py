from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os
import pickle
import fig

### custom QDialog which allows user to fill in ops
class OpsValues(QtGui.QDialog):
    def __init__(self, ops_file, parent=None):
        super(OpsValues, self).__init__(parent)
        self.setGeometry(100,100,600,500)
        self.setWindowTitle('Choose run options')
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.win.setLayout(layout)
        # initial ops values
        pkl_file = open(ops_file,'rb')
        ops = pickle.load(pkl_file)
        pkl_file.close()
        k = 0
        for key in ops:
            lops = 1
            try:
                lops = len(ops[key])
            except (TypeError):
                lops = 1
            if lops==1:
                qedit = QtGui.QLineEdit()
                qedit.setInputMask('%5.2f'%float(ops[key]))
                layout.addWidget(QtGui.QLabel(key),k%18,2*np.floor(float(k)/18),1,1)
                layout.addWidget(qedit,k%18,2*np.floor(float(k)/18)+1,1,1)

                k+=1

### custom QDialog which makes a list of items you can include/exclude
class ListChooser(QtGui.QDialog):
    def __init__(self, Text, parent=None):
        super(ListChooser, self).__init__(parent)
        self.setGeometry(300,300,650,320)
        self.setWindowTitle(Text)
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        self.win.setLayout(layout)
        #self.setCentralWidget(self.win)
        loadtext = QtGui.QPushButton('Load txt file')
        loadtext.resize(loadtext.minimumSizeHint())
        loadtext.clicked.connect(self.load_text)
        layout.addWidget(loadtext,0,0,1,1)
        self.leftlist = QtGui.QListWidget(parent)
        self.rightlist = QtGui.QListWidget(parent)
        layout.addWidget(QtGui.QLabel('INCLUDE'),1,0,1,1)
        layout.addWidget(QtGui.QLabel('EXCLUDE'),1,3,1,1)
        layout.addWidget(self.leftlist,2,0,5,1)
        sright = QtGui.QPushButton('-->')
        sright.resize(sright.minimumSizeHint())
        sleft = QtGui.QPushButton('<--')
        sleft.resize(sleft.minimumSizeHint())
        sright.clicked.connect(self.move_right)
        sleft.clicked.connect(self.move_left)
        layout.addWidget(sright,3,1,1,1)
        layout.addWidget(sleft,4,1,1,1)
        layout.addWidget(self.rightlist,2,3,5,1)
        done = QtGui.QPushButton('OK')
        done.resize(done.minimumSizeHint())
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done,7,1,1,1)
    def move_right(self):
        currentRow = self.leftlist.currentRow()
        if self.leftlist.item(currentRow) is not None:
            self.rightlist.addItem(self.leftlist.item(currentRow).text())
            self.leftlist.takeItem(currentRow)
    def move_left(self):
        currentRow = self.rightlist.currentRow()
        if self.rightlist.item(currentRow) is not None:
            self.leftlist.addItem(self.rightlist.item(currentRow).text())
            self.rightlist.takeItem(currentRow)
    def load_text(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if name:
            try:
                txtfile = open(name[0], 'r')
                files = txtfile.read()
                txtfile.close()
                files = files.splitlines()
                for f in files:
                    self.leftlist.addItem(f)
            except (OSError, RuntimeError, TypeError, NameError):
                print('not a good list')
    def exit_list(self, parent):
        parent.trainfiles = []
        for n in range(len(self.leftlist)):
            if self.leftlist.item(n) is not None:
                parent.trainfiles.append(self.leftlist.item(n).text())
        self.accept()

### custom QPushButton class that plots image when clicked
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows up to 1 button to pressed at a time
class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        ischecked  = parent.viewbtns.checkedId()
        waschecked = parent.btnstate[bid]
        for n in range(len(parent.btnstate)):
            parent.btnstate[n] = False
        if ischecked==bid and not waschecked:
            parent.viewbtns.setExclusive(True)
            parent.ops_plot[1] = bid
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot,
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)
            parent.btnstate[bid]=True
        elif ischecked==bid and waschecked:
            parent.viewbtns.setExclusive(False)
            parent.btnstate[bid]=False
            parent.ops_plot[1] = -1
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot,
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)
        self.setChecked(parent.btnstate[bid])

### Changes colors of ROIs
# button group is exclusive (at least one color is always chosen)
class ColorButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ColorButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        ischecked  = self.isChecked()
        if ischecked:
            parent.ops_plot[2] = bid
            M = fig.draw_masks(parent.ops, parent.stat, parent.ops_plot,
                                parent.iscell, parent.ichosen)
            parent.plot_masks(M)
            parent.plot_colorbar(bid)

def load_trainfiles(trainfiles, statclass):
    traindata = np.zeros((0,len(statclass)+1),np.float32)
    trainfiles_good = []
    if trainfiles is not None:
        for fname in trainfiles:
            badfile = False
            basename, bname = os.path.split(fname)
            try:
                iscell = np.load(fname)
                ncells = iscell.shape[0]
            except (OSError, RuntimeError, TypeError, NameError):
                print(fname+': not a numpy array of booleans')
                badfile = True
            if not badfile:
                basename, bname = os.path.split(fname)
                lstat = 0
                try:
                    pkl_file = open(basename+'/stat.pkl', 'rb')
                    stat = pickle.load(pkl_file)
                    pkl_file.close()
                    ypix = stat[0]['ypix']
                    lstat = len(stat) - 1
                except (KeyError, OSError, RuntimeError, TypeError, NameError, pickle.UnpicklingError):
                    print('\t'+'basename+': incorrect or missing stat.pkl file :(')
                if lstat != ncells:
                    print('\t'+basename+': stat.pkl is not the same length as iscell.npy')
                else:
                    # add iscell and stat to classifier
                    print('\t'+fname+' was added to classifier')
                    iscell = iscell.astype(np.float32)
                    nall = np.zeros((ncells, len(statclass)+1),np.float32)
                    nall[:,0] = iscell
                    k=0
                    for key in statclass:
                        k+=1
                        for n in range(0,ncells):
                            nall[n,k] = stat[n][key]
                    traindata = np.concatenate((traindata,nall),axis=0)
                    trainfiles_good.append(fname)
    return traindata, trainfiles_good
