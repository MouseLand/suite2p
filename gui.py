from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
import sys
import numpy as np
import os
import pickle
import fig
import suite2p

def default_ops():
    ops = {
        'diameter':12, # this is the main parameter for cell detection
        'tau':  1., # this is the main parameter for deconvolution
        'fs': 10.,  # sampling rate (total across planes)
        'nplanes' : 1, # each tiff has these many planes in sequence
        'nchannels' : 1, # each tiff has these many channels per plane
        'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
        'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
        'look_one_level_down': False,
        'baseline': 'maximin', # baselining mode
        'win_baseline': 60., # window for maximin
        'sig_baseline': 10., # smoothing constant for gaussian filter
        'prctile_baseline': 8.,# smoothing constant for gaussian filter
        'neucoeff': .7,  # neuropil coefficient
        'neumax': 1.,  # maximum neuropil coefficient (not implemented)
        'niterneu': 5, # number of iterations when the neuropil coefficient is estimated (not implemented)
        'maxregshift': 0.,
        'subpixel' : 10,
        'batch_size': 200, # number of frames per batch
        'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
        'nimg_init': 200, # subsampled frames for finding reference image
        'navg_frames_svd': 5000,
        'nsvd_for_roi': 1000,
        'ratio_neuropil': 5,
        'tile_factor': 1,
        'threshold_scaling': 1,
        'Vcorr': [],
        'allow_overlap': False,
        'inner_neuropil_radius': 2,
        'outer_neuropil_radius': np.inf,
        'min_neuropil_pixels': 350,
        'ratio_neuropil_to_cell': 3,
        'nframes': 1,
        'diameter': 12
      }
    return ops

class OpsLabel(QtGui.QLabel):
    def __init__(self, text, parent=None):
        super(OpsLabel, self).__init__(parent)
        self.setAutoFillBackground(True)
        #p = self.palette()
        #p.setColor(self.backgroundRole(), QtGui.QColor(223, 230, 248))
        self.setPalette(p)
        self.setMouseTracking(True)
        self.setText(text)

    def mouseMoveEvent(self, event):
        print("On Hover") # event.pos().x(), event.pos().y()

    def mousePressEvent(self, event):
        print(event)

class OutLog:
    def __init__(self, edit, out=None, color=None):
        """(edit, out=None, color=None) -> can write stdout, stderr to a
        QTextEdit.
        edit = QTextEdit
        out = alternate stream ( can be the original sys.stdout )
        color = alternate color (i.e. color stderr a different color)
        """
        self.edit = edit
        self.out = None
        self.color = color

    def write(self, m):
        if self.color:
            tc = self.edit.textColor()
            self.edit.setTextColor(self.color)

        self.edit.moveCursor(QtGui.QTextCursor.End)
        self.edit.insertPlainText( m )

        if self.color:
            self.edit.setTextColor(tc)

        if self.out:
            self.out.write(m)

### custom QDialog which allows user to fill in ops
class OpsValues(QtGui.QDialog):
    def __init__(self, ops_file, parent=None):
        super(OpsValues, self).__init__(parent)
        self.setGeometry(50,50,900,900)
        self.setWindowTitle('Choose run options')
        self.win = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.win.setLayout(self.layout)
        # initial ops values
        self.ops = suite2p.default_ops()
        self.data_path = []
        tifkeys = ['nplanes','nchannels','fs','num_workers']
        regkeys = ['nimg_init', 'batch_size', 'subpixel', 'maxregshift', 'align_by_chan']
        cellkeys = ['diameter','navg_frames_svd','nsvd_for_roi','threshold_scaling', 'allow_overlap']
        neukeys = ['ratio_neuropil_to_cell','inner_neuropil_radius','outer_neuropil_radius','min_neuropil_pixels']
        deconvkeys = ['tau','win_baseline','sig_baseline','prctile_baseline','neucoeff']
        keys = [[],tifkeys, regkeys, cellkeys, neukeys, deconvkeys]
        labels = ['Filepaths','Main settings','Registration','Cell detection','Neuropil','Deconvolution']
        l=0
        self.keylist = []
        self.editlist = []
        for lkey in keys:
            k = 0
            qlabel = QtGui.QLabel(labels[l])
            bigfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
            qlabel.setFont(bigfont)
            self.layout.addWidget(qlabel,0,l,1,1)
            for key in lkey:
                lops = 1
                if self.ops[key] or (self.ops[key] == 0):
                    qedit = QtGui.QLineEdit()
                    qlabel = QtGui.QLabel(key)
                    qlabel.setToolTip('yo')
                    qedit.setText(str(self.ops[key]))
                    self.layout.addWidget(qlabel,k*2+1,l,1,1)
                    self.layout.addWidget(qedit,k*2+2,l,1,1)
                    self.keylist.append(key)
                    self.editlist.append(qedit)
                k+=1
            l+=1
        btiff = QtGui.QPushButton('Add directory to data_path')
        #btiff.resize(btiff.minimumSizeHint())
        btiff.clicked.connect(self.get_folders)
        self.layout.addWidget(btiff,0,0,1,1)
        qlabel = QtGui.QLabel('data_path')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,1,0,1,1)
        runbtn = QtGui.QPushButton('RUN SUITE2P')
        runbtn.clicked.connect(self.run_suite2p)
        self.layout.addWidget(runbtn,11,0,1,1)
        self.textEdit = QtGui.QTextEdit()
        self.layout.addWidget(self.textEdit, 12,0,25,l)
        print('Connecting process')
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdoutReady)
        self.process.readyReadStandardError.connect(self.stderrReady)
        self.process.started.connect(lambda: print('Started!'))
        self.process.finished.connect(lambda: print('Finished!'))

    def append(self, text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        #self.output.ensureCursorVisible()

    def stdoutReady(self):
        text = str(self.process.readAllStandardOutput())
        print(text.strip())
        self.append(text)

    def stderrReady(self):
        text = str(self.process.readAllStandardError())
        print(text.strip())
        self.append(text)


    def run_suite2p(self):
        k=0
        for key in self.keylist:
            if type(self.ops[key]) is float:
                self.ops[key] = float(self.editlist[k].text())
                print(key,'\t\t', float(self.editlist[k].text()))
            elif type(self.ops[key]) is int:
                self.ops[key] = int(self.editlist[k].text())
                print(key,'\t\t', int(self.editlist[k].text()))
            elif type(self.ops[key]) is bool:
                self.ops[key] = bool(self.editlist[k].text())
                print(key,'\t\t', bool(self.editlist[k].text()))
            k+=1
        self.ops['data_path'] = self.data_path
        self.ops['subfolders'] = []
        print('running suite2p')
        print('Starting process')
        np.save('ops.npy', self.ops)
        self.process.start('python', ['main.py'])

    def get_folders(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Add directory to data path")
        self.data_path.append(name)
        self.layout.addWidget(QtGui.QLabel(name),
                              len(self.data_path)+1,0,1,1)



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
        #self.leftlist.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
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
