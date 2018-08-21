from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import console
import sys
import numpy as np
import os
import pickle
from suite2p import fig
from suite2p import run_s2p

# this file contains helper functions for GUI and run dialog

### custom QDialog which allows user to fill in ops and run suite2p!
class RunWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(50,50,1200,800)
        self.setWindowTitle('Choose run options')
        self.win = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout()
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)
        # initial ops values
        self.ops = run_s2p.default_ops()
        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        tifkeys = ['nplanes','nchannels','functional_chan','diameter','tau','fs']
        parkeys = ['num_workers','num_workers_roi','combined']
        regkeys = ['nimg_init', 'batch_size', 'maxregshift', 'align_by_chan', 'reg_tif']
        cellkeys = ['max_overlap','threshold_scaling','max_iterations','navg_frames_svd','nsvd_for_roi','tile_factor']
        neukeys = ['ratio_neuropil_to_cell','inner_neuropil_radius','outer_neuropil_radius','min_neuropil_pixels']
        deconvkeys = ['win_baseline','sig_baseline','prctile_baseline','neucoeff']
        keys = [[],tifkeys, parkeys, regkeys, cellkeys, neukeys, deconvkeys]
        labels = ['Filepaths','Main settings','Parallel','Registration','ROI detection','Neuropil','Deconvolution']
        tooltips = ['each tiff has this many planes in sequence',
                    'each tiff has this many channels per plane',
                    'this channel is used to extract functional ROIs (1-based)',
                    'approximate diameter of ROIs in pixels (can input two numbers separated by a comma for elongated ROIs)',
                    'timescale of sensor in deconvolution (in seconds)',
                    'sampling rate (per plane)',
                    '0 to select num_cores, -1 to disable parallelism, N to enforce value',
                    'ROI detection parallelism: 0 to select number of planes, -1 to disable parallelism, N to enforce value',
                    'combine results across planes in separate folder "combined" at end of processing',
                    '# of subsampled frames for finding reference image',
                    'number of frames per batch',
                    'max allowed registration shift, as a fraction of frame max(width and height)',
                    'when multi-channel, you can align by non-functional channel (1-based)',
                    'if 1, registered tiffs are saved',
                    'ROIs with greater than this overlap as a fraction of total pixels will be discarded',
                    'adjust the automatically determined threshold by this scalar multiplier',
                    'maximum number of iterations for ROI detection',
                    'max number of binned frames for the SVD',
                    'max number of SVD components to keep for ROI detection',
                    'tile factor',
                    'minimum ratio between neuropil radius and cell radius',
                    'number of pixels between ROI and neuropil donut',
                    'maximum neuropil radius',
                    'minimum number of pixels in the neuropil',
                    'window for maximin',
                    'smoothing constant for gaussian filter',
                    'smoothing constant for gaussian filter',
                    'neuropil coefficient']
        l=0
        self.keylist = []
        self.editlist = []
        kk=0
        for lkey in keys:
            k = 0
            qlabel = QtGui.QLabel(labels[l])
            bigfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
            qlabel.setFont(bigfont)
            self.layout.addWidget(qlabel,0,2*l,1,2)
            for key in lkey:
                lops = 1
                if self.ops[key] or (self.ops[key] == 0):
                    qedit = QtGui.QLineEdit()
                    qlabel = QtGui.QLabel(key)
                    qlabel.setToolTip(tooltips[kk])
                    if key == 'diameter':
                        if (type(self.ops[key]) is not int) and (len(self.ops[key])>1):
                            dstr = str(int(self.ops[key][0])) + ', ' + str(int(self.ops[key][1]))
                        else:
                            dstr = str(int(self.ops[key]))
                    else:
                        if type(self.ops[key]) is not bool:
                            dstr = str(self.ops[key])
                        else:
                            dstr = str(int(self.ops[key]))
                    qedit.setText(dstr)
                    qedit.setFixedWidth(105)
                    self.layout.addWidget(qlabel,k*2+1,2*l,1,2)
                    self.layout.addWidget(qedit,k*2+2,2*l,1,2)
                    self.keylist.append(key)
                    self.editlist.append(qedit)
                k+=1
                kk+=1
            l+=1
        # data_path
        key = 'look_one_level_down'
        qlabel = QtGui.QLabel(key)
        qlabel.setToolTip('whether to look in all subfolders when searching for tiffs')
        self.layout.addWidget(qlabel,1,0,1,1)
        qedit = QtGui.QLineEdit()
        self.layout.addWidget(qedit,2,0,1,1)
        qedit.setText(str(int(self.ops[key])))
        qedit.setFixedWidth(105)
        self.keylist.append(key)
        self.editlist.append(qedit)
        btiff = QtGui.QPushButton('Add directory to data_path')
        btiff.clicked.connect(self.get_folders)
        self.layout.addWidget(btiff,3,0,1,1)
        qlabel = QtGui.QLabel('data_path')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,4,0,1,1)
        # save_path0
        bsave = QtGui.QPushButton('Add save_path (default is 1st data_path)')
        bsave.clicked.connect(self.save_folder)
        self.layout.addWidget(bsave,15,0,1,1)
        self.savelabel = QtGui.QLabel('')
        self.layout.addWidget(self.savelabel,16,0,1,1)
        # fast_disk
        bbin = QtGui.QPushButton('Add fast_disk (default is save_path)')
        bbin.clicked.connect(self.bin_folder)
        self.layout.addWidget(bbin,17,0,1,1)
        self.binlabel = QtGui.QLabel('')
        self.layout.addWidget(self.binlabel,18,0,1,1)
        self.runButton = QtGui.QPushButton('RUN SUITE2P')
        self.runButton.clicked.connect(lambda: self.run_S2P(parent))
        self.layout.addWidget(self.runButton,20,0,1,1)
        self.textEdit = QtGui.QTextEdit()
        self.layout.addWidget(self.textEdit, 21,0,30,2*l)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(lambda: self.finished(parent))
        # stop process
        self.stopButton = QtGui.QPushButton('STOP')
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, 20,2,1,2)
        self.stopButton.clicked.connect(self.stop)

    def stop(self):
        self.finish = False
        self.process.kill()

    def started(self):
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def finished(self, parent):
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        if self.finish and not self.error:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Opening in GUI (can close this window)\n')
            parent.fname = os.path.join(self.save_path, 'suite2p', 'plane0','stat.npy')
            parent.load_proc()
        elif not self.error:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Interrupted by user (not finished)\n')
        else:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Interrupted by error (not finished)\n')

    def stdout_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(str(self.process.readAllStandardOutput(), 'utf-8'))
        self.textEdit.ensureCursorVisible()

    def stderr_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText('>>>ERROR<<<\n')
        cursor.insertText(str(self.process.readAllStandardError(), 'utf-8'))
        self.textEdit.ensureCursorVisible()
        self.error = True

    def run_S2P(self, parent):
        self.finish = True
        self.error = False
        k=0
        for key in self.keylist:
            if key=='diameter':
                diams = self.editlist[k].text().replace(' ','').split(',')
                if len(diams)>1:
                    self.ops[key] = [int(diams[0]), int(diams[1])]
                else:
                    self.ops[key] = int(diams[0])
            else:
                if type(self.ops[key]) is float:
                    self.ops[key] = float(self.editlist[k].text())
                    #print(key,'\t\t', float(self.editlist[k].text()))
                elif type(self.ops[key]) is int or bool:
                    self.ops[key] = int(self.editlist[k].text())
                    #print(key,'\t\t', int(self.editlist[k].text()))
            k+=1
        self.db = {}
        self.db['data_path'] = self.data_path
        self.db['subfolders'] = []
        if len(self.save_path)==0:
            fpath = self.db['data_path'][0]
            self.save_path = fpath
        self.db['save_path0'] = self.save_path
        if len(self.fast_disk)==0:
            self.fast_disk = self.save_path
        self.db['fast_disk'] = self.fast_disk
        print('Running suite2p!')
        print('starting process')
        np.save('ops.npy', self.ops)
        np.save('db.npy', self.db)
        self.process.start('python -u -W ignore -m suite2p --ops ops.npy --db db.npy')

    def get_folders(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Add directory to data path")
        self.data_path.append(name)
        self.layout.addWidget(QtGui.QLabel(name),
                              len(self.data_path)+4,0,1,1)

    def save_folder(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Save folder for data")
        self.save_path = name
        self.savelabel.setText(name)

    def bin_folder(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Folder for binary file")
        self.fast_disk = name
        self.binlabel.setText(name)

# custom vertical label
class VerticalLabel(QtGui.QWidget):
    def __init__(self, text=None):
        super(self.__class__, self).__init__()
        self.text = text

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.translate(0, 0)
        painter.rotate(90)
        if self.text:
            painter.drawText(0, 0, self.text)
        painter.end()

### custom QPushButton class that plots image when clicked
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows only 1 button to pressed at a time
class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in parent.viewbtns.buttons():
            b.setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot[1] = bid
        if parent.ops_plot[2] == parent.ops_plot[3].shape[1]:
            fig.draw_corr(parent)
        M = fig.draw_masks(parent)
        fig.plot_masks(parent,M)

### Changes colors of ROIs
# button group is exclusive (at least one color is always chosen)
class ColorButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ColorButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in parent.colorbtns.buttons():
            b.setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot[2] = bid
        # disable top click
        if bid==0:
            for b in [1,2,4,5]:
                parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
                parent.topbtns.button(b).setEnabled(False)
        else:
            for b in [1,2,4,5]:
                parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
                parent.topbtns.button(b).setEnabled(True)
        if bid==6:
            fig.corr_masks(parent)
        M = fig.draw_masks(parent)
        fig.plot_masks(parent,M)
        fig.plot_colorbar(parent,bid)

# size of view
class SizeButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(SizeButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
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
        parent.zoom_plot(1)
        parent.win.show()
        parent.show()
        parent.zoom_plot(1)
        parent.win.show()
        parent.show()

# selection of top neurons
class TopButton(QtGui.QPushButton):
    def __init__(self, bid, parent=None):
        super(TopButton,self).__init__(parent)
        text = [' draw selection', ' select top', ' select bottom',
                ' draw selection', ' select top', ' select bottom']
        self.setText(text[bid])
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in parent.topbtns.buttons():
            b.setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        if bid==0 or bid==3:
            parent.ROI_selection(bid)
        else:
            parent.top_selection(bid)
