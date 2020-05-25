import os
import shutil

import numpy as np
from PyQt5 import QtGui

from . import masks
from .. import classification


def make_buttons(parent,b0):
    # ----- CLASSIFIER BUTTONS -------
    cllabel = QtGui.QLabel("")
    cllabel.setFont(parent.boldfont)
    cllabel.setText("<font color='white'>Classifier</font>")
    parent.classLabel = QtGui.QLabel("<font color='white'>not loaded (using prob from iscell.npy)</font>")
    parent.classLabel.setFont(QtGui.QFont("Arial", 8))
    parent.l0.addWidget(cllabel, b0, 0, 1, 2)
    b0+=1
    parent.l0.addWidget(parent.classLabel, b0, 0, 1, 2)
    parent.addtoclass = QtGui.QPushButton(" add current data to classifier")
    parent.addtoclass.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
    parent.addtoclass.clicked.connect(lambda: add_to(parent))
    parent.addtoclass.setStyleSheet(parent.styleInactive)
    b0+=1
    parent.l0.addWidget(parent.addtoclass, b0, 0, 1, 2)
    return b0

def load_classifier(parent):
    name = QtGui.QFileDialog.getOpenFileName(parent, "Open File")
    if name:
        load(parent, name[0])
        class_activated(parent)
    else:
        print("no classifier")

def load_s2p_classifier(parent):
    load(parent, parent.classorig)
    class_file(parent)
    parent.saveDefault.setEnabled(True)

def load_default_classifier(parent):
    load(parent, parent.classuser)
    class_activated(parent)

def class_file(parent):
    if parent.classfile == parent.classuser:
        cfile = "default classifier"
    elif parent.classfile == parent.classorig:
        cfile = "suite2p classifier"
    else:
        cfile = parent.classfile
    cstr = "<font color='white'>" + cfile + "</font>"
    parent.classLabel.setText(cstr)

def class_activated(parent):
    class_file(parent)
    parent.saveDefault.setEnabled(True)
    parent.addtoclass.setStyleSheet(parent.styleUnpressed)
    parent.addtoclass.setEnabled(True)

def class_default(parent):
    dm = QtGui.QMessageBox.question(
        parent,
        "Default classifier",
        "Are you sure you want to overwrite your default classifier?",
        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
    )
    if dm == QtGui.QMessageBox.Yes:
        classfile = parent.classuser
        save_model(classfile, parent.model.stats, parent.model.iscell, parent.model.keys)

def reset_default(parent):
    dm = QtGui.QMessageBox.question(
        parent,
        "Default classifier",
        ("Are you sure you want to reset the default classifier "
         "to the built-in suite2p classifier?"),
        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
    )
    if dm == QtGui.QMessageBox.Yes:
        shutil.copy(parent.classorig, parent.classuser)

def load(parent, name):
    print('loading classifier ', name)
    parent.classfile = name
    parent.model = classification.Classifier(classfile=name)
    if parent.model.loaded:
        activate(parent, True)

def save_model(name, train_stats, train_iscell, keys):
    model = {}
    model['stats']  = train_stats
    model['iscell'] = train_iscell
    model['keys']   = keys
    print('saving classifier in ' + name)
    np.save(name, model)

def load_list(parent):
    # will return
    LC = ListChooser('classifier training files', parent)
    result = LC.exec_()

def load_data(parent,keys,trainfiles):
    train_stats = np.zeros((0,len(keys)),np.float32)
    train_iscell = np.zeros((0,),np.float32)
    trainfiles_good = []
    loaded = False
    if trainfiles is not None:
        for fname in trainfiles:
            badfile = False
            basename, bname = os.path.split(fname)
            try:
                iscells = np.load(fname)
                ncells = iscells.shape[0]
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                print('\t'+fname+': not a numpy array of booleans')
                badfile = True
            if not badfile:
                basename, bname = os.path.split(fname)
                lstat = 0
                try:
                    stat = np.load(basename+'/stat.npy', allow_pickle=True)
                    ypix = stat[0]['ypix']
                    lstat = len(stat)
                except (KeyError, OSError, RuntimeError, TypeError, NameError):
                    print('\t'+basename+': incorrect or missing stat.npy file :(')
                if lstat != ncells:
                    print('\t'+basename+': stat.npy is not the same length as iscell.npy')
                else:
                    # add iscell and stat to classifier
                    print('\t'+fname+' was added to classifier')
                    iscell = iscells[:,0].astype(np.float32)
                    stats = np.reshape(np.array([stat[j][k] for j in range(len(stat)) for k in parent.default_keys]),
                                (len(stat),-1))
                    train_stats = np.concatenate((train_stats,stats),axis=0)
                    train_iscell = np.concatenate((train_iscell,iscell),axis=0)
                    trainfiles_good.append(fname)
    if len(trainfiles_good) > 0:
        classfile, saved = save(parent,train_stats,train_iscell,keys)
        if saved:
            parent.classfile = classfile
            loaded = True
        else:
            msg = QtGui.QMessageBox.information(parent,'Incorrect file path',
                                                'Incorrect save path for classifier, classifier not built.')
    else:
        msg = QtGui.QMessageBox.information(parent,'Incorrect files',
                                            'No valid datasets chosen to build classifier, classifier not built.')
    return loaded

def add_to(parent):
    fname = parent.basename+'/iscell.npy'
    print('Adding current dataset to classifier')
    if parent.classfile == parent.classuser:
        cfile = 'the default classifier'
    else:
        cfile = parent.classfile
    dm = QtGui.QMessageBox.question(parent,'Default classifier',
                                    'Current classifier is '+cfile+'. Add to this classifier?',
                                    QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
    if dm == QtGui.QMessageBox.Yes:
        stats = np.reshape(np.array([parent.stat[j][k] for j in range(len(parent.stat)) for k in parent.model.keys]),
                                (len(parent.stat),-1))
        parent.model.stats = np.concatenate((parent.model.stats,stats),axis=0)
        parent.model.iscell = np.concatenate((parent.model.iscell,parent.iscell),axis=0)
        save_model(parent.classfile, parent.model.stats, parent.model.iscell, parent.model.keys)
        activate(parent, True)
        msg = QtGui.QMessageBox.information(parent,'Classifier saved and loaded',
                                            'Current dataset added to classifier, and cell probabilities computed and in GUI')


def save(parent, train_stats, train_iscell, keys):
    name = QtGui.QFileDialog.getSaveFileName(parent,'Classifier name (*.npy)')
    name = name[0]
    saved = False
    if name:
        try:
            save_model(name, train_stats, train_iscell, keys)
            saved = True
        except (OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
            print('ERROR: incorrect filename for saving')
    return name, saved

def save_list(parent):
    name = QtGui.QFileDialog.getSaveFileName(parent,'Save list of iscell.npy')
    if name:
        try:
            with open(name[0],'w') as fid:
                for f in parent.trainfiles:
                    fid.write(f)
                    fid.write('\n')
        except (ValueError, OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
            print('ERROR: incorrect filename for saving')

def activate(parent, inactive):
    if inactive:
        parent.probcell = parent.model.predict_proba(parent.stat)
    class_masks(parent)
    parent.update_plot()

def disable(parent):
    parent.classbtn.setEnabled(False)
    parent.saveClass.setEnabled(False)
    parent.saveTrain.setEnabled(False)
    for btns in parent.classbtns.buttons():
        btns.setEnabled(False)

### custom QDialog which makes a list of items you can include/exclude
class ListChooser(QtGui.QDialog):
    def __init__(self, Text, parent=None):
        super(ListChooser, self).__init__(parent)
        self.setGeometry(300,300,500,320)
        self.setWindowTitle(Text)
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        self.win.setLayout(layout)
        #self.setCentralWidget(self.win)
        loadcell = QtGui.QPushButton('Load iscell.npy')
        loadcell.resize(200,50)
        loadcell.clicked.connect(self.load_cell)
        layout.addWidget(loadcell,0,0,1,1)
        loadtext = QtGui.QPushButton('Load txt file list')
        loadtext.clicked.connect(self.load_text)
        layout.addWidget(loadtext,0,1,1,1)
        layout.addWidget(QtGui.QLabel('(select multiple using ctrl)'),1,0,1,1)
        self.list = QtGui.QListWidget(parent)
        layout.addWidget(self.list,2,0,5,4)
        #self.list.resize(450,250)
        self.list.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        save = QtGui.QPushButton('build classifier')
        save.clicked.connect(lambda: self.build_classifier(parent))
        layout.addWidget(save,8,0,1,1)
        self.apply = QtGui.QPushButton('load in GUI')
        self.apply.clicked.connect(lambda: self.apply_class(parent))
        self.apply.setEnabled(False)
        layout.addWidget(self.apply,8,1,1,1)
        self.saveasdefault = QtGui.QPushButton('save as default')
        self.saveasdefault.clicked.connect(lambda: self.save_default(parent))
        self.saveasdefault.setEnabled(False)
        layout.addWidget(self.saveasdefault,8,2,1,1)
        done = QtGui.QPushButton('close')
        done.clicked.connect(self.exit_list)
        layout.addWidget(done,8,3,1,1)

    def load_cell(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open iscell.npy file',filter='iscell.npy')
        if name:
            try:
                iscell = np.load(name[0])
                badfile = True
                if iscell.shape[0] > 0:
                    if iscell[0,0]==0 or iscell[0,0]==1:
                        badfile = False
                        self.list.addItem(name[0])
                if badfile:
                    QtGui.QMessageBox.information(self, 'iscell.npy should be 0/1')
            except (OSError, RuntimeError, TypeError, NameError):
                QtGui.QMessageBox.information(self, 'iscell.npy should be 0/1')
        else:
            QtGui.QMessageBox.information(self, 'iscell.npy should be 0/1')

    def load_text(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open *.txt file', filter='text file (*.txt)')
        if name:
            try:
                txtfile = open(name[0], 'r')
                files = txtfile.read()
                txtfile.close()
                files = files.splitlines()
                for f in files:
                    self.list.addItem(f)
            except (OSError, RuntimeError, TypeError, NameError):
                QtGui.QMessageBox.information(self, 'not a text file')
                print('not a good list')

    def build_classifier(self, parent):
        parent.trainfiles = []
        i=0
        for item in self.list.selectedItems():
            parent.trainfiles.append(item.text())
            i+=1
        if i==0:
            for r in range(self.list.count()):
                parent.trainfiles.append(self.list.item(r).text())
        if len(parent.trainfiles)>0:
            print('Populating classifier:')
            keys = parent.default_keys
            loaded = load_data(parent, keys, parent.trainfiles)
            if loaded:
                msg = QtGui.QMessageBox.information(parent,'Classifier saved',
                                                    'Classifier built from valid files and saved.')
                self.apply.setEnabled(True)
                self.saveasdefault.setEnabled(True)

    def apply_class(self, parent):
        parent.model = classification.Classifier(classfile=parent.classfile)
        activate(parent, True)

    def save_default(self, parent):
        dm = QtGui.QMessageBox.question(self,'Default classifier',
                                        'Are you sure you want to overwrite your default classifier?',
                                        QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if dm == QtGui.QMessageBox.Yes:
            shutil.copy(parent.classfile, parent.classuser)

    def exit_list(self):
        self.accept()


def class_masks(parent):
    c = 6
    istat = parent.probcell
    parent.colors['colorbar'][c] = [istat.min(), (istat.max()-istat.min())/2, istat.max()]
    istat = istat - istat.min()
    istat = istat / istat.max()
    col = masks.istat_transform(istat, parent.ops_plot['colormap'])
    parent.colors['cols'][c] = col
    parent.colors['istat'][c] = istat.flatten()

    masks.rgb_masks(parent, col, c)