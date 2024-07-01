"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import glob, json, os, shutil, pathlib, sys, copy
from datetime import datetime

import numpy as np

from qtpy import QtGui, QtCore 
from qtpy.QtWidgets import (QDialog, QLineEdit, QLabel, QPushButton, QWidget, 
                            QGridLayout, QMainWindow, QComboBox, QTextEdit, 
                            QFileDialog, QAction, QToolButton, QStyle,
                            QListWidget, QCheckBox, QScrollArea)
from superqt import QCollapsible 

from cellpose.models import get_user_models, model_path, MODEL_NAMES

from . import io, utils
from .. import default_ops, default_db, parameters

### ---- this file contains helper functions for GUI and the RUN window ---- ###

def list_to_str(l):
    return ", ".join(str(l0) for l0 in l)

def create_input(key, OPS, ops_gui):
    qlabel = QLabel(OPS[key]["gui_name"])
    qlabel.setToolTip(OPS[key]["description"])
    qlabel.setFixedWidth(160)
    qlabel.setAlignment(QtCore.Qt.AlignRight)
    if OPS[key]["type"] == bool:
        ops_gui[key] = QCheckBox()
        ops_gui[key].setChecked(OPS[key]["default"])
    else:
        ops_gui[key] = QLineEdit()
        ops_gui[key].setFixedWidth(80)
        if OPS[key]["default"] is not None:
            if OPS[key]["type"] == list or OPS[key]["type"] == tuple:
                ops_gui[key].setText(list_to_str(OPS[key]["default"]))
            else:
                ops_gui[key].setText(str(OPS[key]["default"]))
    ops_gui[key].setToolTip(OPS[key]["description"])
    return qlabel 

### custom QMainWindow which allows user to fill in ops and run suite2p!
class RunWindow(QMainWindow):

    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(65, 65, 1100, 900)
        self.setWindowTitle(
            "Choose run options (hold mouse over parameters to see descriptions)")
        self.parent = parent
        cwidget = QWidget()
        self.layout = QGridLayout()
        cwidget.setLayout(self.layout)
        self.setCentralWidget(cwidget)
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        # initial ops values
        self.setStyleSheet(utils.stylesheet())
        self.setPalette(utils.DarkPalette())
        self.opsfile = parent.opsuser
        self.ops_path = os.fspath(
            pathlib.Path.home().joinpath(".suite2p").joinpath("ops").absolute())
        try:
            self.reset_ops()
            print("loaded default ops")
        except Exception as e:
            print("ERROR: %s" % e)
            print("could not load default ops, using built-in ops settings")
            self.ops = default_ops()

        # remove any remaining ops files
        fs = glob.glob("ops*.npy")
        for f in fs:
            os.remove(f)
        fs = glob.glob("db*.npy")
        for f in fs:
            os.remove(f)

        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        self.loadOps = QAction("&Load ops file")
        self.loadOps.triggered.connect(self.load_ops)
        self.loadOps.setEnabled(True)
        file_menu.addAction(self.loadOps)
        
        self.saveDef = QAction("&Save ops as default")
        self.saveDef.triggered.connect(self.save_default_ops)
        self.saveDef.setEnabled(True)
        file_menu.addAction(self.saveDef)
        
        self.revertDef = QAction("Revert default ops to built-in")
        self.revertDef.triggered.connect(self.revert_default_ops)
        self.revertDef.setEnabled(True)
        file_menu.addAction(self.revertDef)

        self.saveOps = QAction("Save current ops to file")
        self.saveOps.triggered.connect(self.save_ops)
        self.saveOps.setEnabled(True)
        file_menu.addAction(self.saveOps)

        file_submenu = file_menu.addMenu("Load example ops")
        self.onePOps = QAction("1P imaging")
        self.onePOps.triggered.connect(lambda: self.load_ops("ops_1P.npy"))
        file_submenu.addAction(self.onePOps)
        self.dendriteOps = QAction("dendrites / axons")
        self.dendriteOps.triggered.connect(lambda: self.load_ops("ops_dendrite.npy"))
        file_submenu.addAction(self.dendriteOps)

        batch_menu = main_menu.addMenu("&Batch processing")
        self.addToBatch = QAction("Save db+ops to batch list")
        self.addToBatch.triggered.connect(self.add_batch)
        self.addToBatch.setEnabled(False)
        batch_menu.addAction(self.addToBatch)

        self.batchList = QAction("View/edit batch list")
        self.batchList.triggered.connect(self.remove_ops)
        self.batchList.setEnabled(False)
        batch_menu.addAction(self.batchList)
        
        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        self.batch = False
        self.f = 0

        self.OPS = parameters.OPS
        self.ops = default_ops()
        parameters.add_descriptions(self.OPS)
        self.DB = parameters.DB
        self.db = default_db()
        parameters.add_descriptions(self.DB, dstr="db")
        
        self.create_db_ops_inputs()

    def reset_ops(self):
        self.ops = np.load(self.opsfile, allow_pickle=True).item()
        ops0 = default_ops()
        self.ops = {**ops0, **self.ops}
        if hasattr(self, "editlist"):
            for k in range(len(self.editlist)):
                self.editlist[k].set_text(self.ops)

    def create_db_ops_inputs(self):
        self.db_gui = {}
        self.ops_gui = {}

        XLfont = "QLabel {font-size: 12pt; font: Arial; font-weight: bold;}"
        bigfont = "QLabel {font-size: 10pt; font: Arial; font-weight: bold;}"
        qlabel = QLabel("File paths")
        qlabel.setStyleSheet(XLfont)
        self.layout.addWidget(qlabel, 0, 0, 1, 1)
        self.layout.addWidget(QLabel(""), 4, 4, 1, 2) 

        # data_path
        key = "input_format"
        qlabel = QLabel(key)
        qlabel.setStyleSheet(bigfont)
        qlabel.setToolTip("File format (selects which parser to use)")
        self.layout.addWidget(qlabel, 1, 0, 1, 1)
        self.db_gui["input_format"] = QComboBox()
        [
            self.db_gui["input_format"].addItem(f)
            for f in ["tif", "bruker", "sbx", "h5", "movie", "nd2", "mesoscan", "raw", "dcimg"]
        ]
        self.layout.addWidget(self.db_gui["input_format"], 1, 1, 1, 1)
        self.db_gui["look_one_level_down"] = QCheckBox("look one level down")
        self.db_gui["look_one_level_down"].setLayoutDirection(QtCore.Qt.RightToLeft) 
        self.db_gui["look_one_level_down"].setChecked(False)
        self.db_gui["look_one_level_down"].setToolTip(self.DB["look_one_level_down"]["description"])
        self.layout.addWidget(self.db_gui["look_one_level_down"], 1, 2, 1, 1)

        cw = 5
        qlabel = QLabel("data_path")
        qlabel.setStyleSheet(bigfont)
        qlabel.setToolTip(self.DB["data_path"]["description"])
        self.layout.addWidget(qlabel, 2, 0, 1, 1)
        self.db_gui["data_path"] = QLabel() #QListWidget()
        self.layout.addWidget(self.db_gui["data_path"], 3, 0, 8, cw)
        self.addDataPath = QPushButton("add folder")
        self.addDataPath.setFixedWidth(100)
        self.addDataPath.setToolTip(self.DB["data_path"]["description"])
        self.addDataPath.clicked.connect(self.get_folders)
        self.layout.addWidget(self.addDataPath, 2, 1, 1, 1)
        #self.removeDataPath = QPushButton("remove selected")
        #self.removeDataPath.setFixedWidth(100)
        #self.remove_data_path.clicked.connect(self.remove_data_path)
        #self.layout.addWidget(self.removeDataPath, 2, 2, 1, 1)

        self.bsave = QPushButton("add save_path0 (default is 1st data_path)")
        self.bsave.clicked.connect(self.save_folder)
        self.layout.addWidget(self.bsave, 13, 0, 1, cw)
        self.savelabel = QLabel("")
        self.layout.addWidget(self.savelabel, 14, 0, 1, cw)
        # fast_disk
        self.bbin = QPushButton("add fast_disk (default is save_path0)")
        self.bbin.clicked.connect(self.bin_folder)
        self.layout.addWidget(self.bbin, 15, 0, 1, cw)
        self.binlabel = QLabel("")
        self.layout.addWidget(self.binlabel, 16, 0, 1, cw)
        n0 = 17
        self.runButton = QPushButton("RUN SUITE2P")
        self.runButton.clicked.connect(self.run_S2P)
        self.layout.addWidget(self.runButton, n0, 0, 1, 1)
        self.runButton.setEnabled(False)
        self.textEdit = QTextEdit()
        self.layout.addWidget(self.textEdit, n0 + 1, 0, 16, cw+2)
        #self.textEdit.setFixedHeight(300)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(self.finished)
        # stop process
        self.stopButton = QPushButton("STOP")
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, n0, 1, 1, 1)
        self.stopButton.clicked.connect(self.stop)
        

        self.dbkeys = [
             "keep_movie_raw", "nplanes", "nchannels",
            "functional_chan", "ignore_flyback", "h5py_key", 
            "nwb_series", "force_sktiff"
        ]
        qlabel = QLabel("file settings")
        qlabel.setStyleSheet(bigfont)
        qlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(qlabel, 0, cw, 1, 2)
        b = 1
        for key in self.dbkeys:
            qlabel = create_input(key, self.DB, self.db_gui)
            self.layout.addWidget(qlabel, b, cw, 1, 1)
            self.layout.addWidget(self.db_gui[key], b, cw+1, 1, 1)
            b+=1

        self.genkeys = [
            "torch_device", "tau", "fs", "diameter"
        ]
        qlabel = QLabel("general settings")
        qlabel.setAlignment(QtCore.Qt.AlignCenter)
        qlabel.setStyleSheet(bigfont)
        self.layout.addWidget(qlabel, b+1, cw, 1, 2)
        b+=2
        for key in self.genkeys:
            qlabel = create_input(key, self.OPS, self.ops_gui)
            self.layout.addWidget(qlabel, b, cw, 1, 1)
            self.layout.addWidget(self.ops_gui[key], b, cw+1, 1, 1)
            b+=1

        scrollArea = QScrollArea()
        scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scrollArea.setStyleSheet("""QScrollArea { border: none }""")
        scrollArea.setWidgetResizable(True)
        swidget = QWidget(self)
        scrollArea.setWidget(swidget)
        layoutScroll = QGridLayout()
        swidget.setLayout(layoutScroll)
        self.layout.addWidget(scrollArea, 0, cw+2, n0+17, 4)


        labels = ["run", "io", "registration", "detection", "extraction", 
                  "dcnv_preprocess"]
        b = 0
        for label in labels:
            qbox = QCollapsible(f"{label} settings")
            qbox._toggle_btn.setStyleSheet(bigfont)
            qboxG = QGridLayout()
            _content = QWidget()
            _content.setLayout(qboxG)
            _content.setMaximumHeight(0)
            _content.setMinimumHeight(0)
            bl = 0
            self.ops_gui[label] = {}
            for key in self.OPS[label].keys():
                if "gui_name" in self.OPS[label][key]:
                    qlabel = create_input(key, self.OPS[label], self.ops_gui[label])
                    qboxG.addWidget(qlabel, bl, 0, 1, 1)
                    qboxG.addWidget(self.ops_gui[label][key], bl, 1, 1, 1)
                    bl+=1
            qbox.setContent(_content)
            if label=="run":
                qbox.expand()
            layoutScroll.addWidget(qbox, b, 0, 1, 1)
            b+=2
        layoutScroll.addWidget(QLabel(""), b, 0, 1, 1)
        
            

        
        # l = 0
        # self.keylist = []
        # self.editlist = []
        # kk = 0
        # wk = 0
        # for lkey in keys:
        #     k = 0
        #     kl = 0
        #     labs = [self.OPS[lkey]["gui_name"]]
        #     keyl = [lkey]
        #     for label in labs:
        #         qlabel = QLabel(label)
        #         qlabel.setStyleSheet(bigfont)
        #         self.layout.addWidget(qlabel, k * 2, 2 * (l + 4), 1, 2)
        #         k += 1
        #         for key in keyl:
        #             lops = 1
        #             if self.OPS[key]: #or (self.ops[key] == 0) or len(self.ops[key]) == 0:
        #                 qedit = LineEdit(wk, key, self)
        #                 qlabel = QLabel(key)
        #                 qlabel.setToolTip(self.OPS[key]["description"])
        #                 qedit.set_text(self.OPS[key])
        #                 qedit.setToolTip(self.OPS[key]["description"])
        #                 qedit.setFixedWidth(90)
        #                 self.layout.addWidget(qlabel, k * 2 - 1, 2 * (l + 4), 1, 2)
        #                 self.layout.addWidget(qedit, k * 2, 2 * (l + 4), 1, 2)
        #                 self.keylist.append(key)
        #                 self.editlist.append(qedit)
        #                 wk += 1
        #             k += 1
        #             kk += 1
        #         kl += 1
        #     l += 1

       
        
    def remove_ops(self):
        L = len(self.opslist)
        if L == 1:
            self.batch = False
            self.opslist = []
            self.removeOps.setEnabled(False)
        else:
            del self.opslist[L - 1]
        self.odata[L - 1].setText("")
        self.odata[L - 1].setToolTip("")
        self.f = 0

    def add_batch(self):
        self.add_ops()
        L = len(self.opslist)
        self.odata[L].setText(self.datastr)
        self.odata[L].setToolTip(self.datastr)

        # clear file fields
        self.db = {}
        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        for n in range(self.n_batch):
            self.qdata[n].setText("")
        self.savelabel.setText("")
        self.binlabel.setText("")

        # clear all ops
        # self.reset_ops()

        # enable all the file loaders again
        self.btiff.setEnabled(True)
        self.bsave.setEnabled(True)
        self.bbin.setEnabled(True)
        # and enable the run button
        self.runButton.setEnabled(True)
        self.removeOps.setEnabled(True)
        self.listOps.setEnabled(False)

    def add_ops(self):
        self.f = 0
        self.compile_ops_db()
        L = len(self.opslist)
        np.save(os.path.join(self.ops_path, "ops%d.npy" % L), self.ops)
        np.save(os.path.join(self.ops_path, "db%d.npy" % L), self.db)
        self.opslist.append("ops%d.npy" % L)
        if hasattr(self, "h5_key") and len(self.h5_key) > 0:
            self.db["h5py_key"] = self.h5_key

    def compile_ops_db(self):
        for k, key in enumerate(self.keylist):
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys,
                                                      self.stringkeys)
        self.db = {}
        self.db["data_path"] = self.data_path
        self.db["subfolders"] = []
        self.datastr = self.data_path[0]

        # add data type specific keys
        if hasattr(self, "h5_key") and len(self.h5_key) > 0:
            self.db["h5py_key"] = self.h5_key
        elif self.inputformat.currentText() == "sbx":
            self.db["sbx_ndeadcols"] = -1

        # add save_path0 and fast_disk
        if len(self.save_path) == 0:
            self.save_path = self.db["data_path"][0]
        self.db["save_path0"] = self.save_path
        if len(self.fast_disk) == 0:
            self.fast_disk = self.save_path
        self.db["fast_disk"] = self.fast_disk
        self.db["input_format"] = self.inputformat.currentText()

    def run_S2P(self):
        if len(self.opslist) == 0:
            self.add_ops()
        # pre-download model
        pretrained_model_string = self.ops.get("pretrained_model", "cyto")
        pretrained_model_string = pretrained_model_string if pretrained_model_string is not None else "cyto"
        pretrained_model_path = model_path(pretrained_model_string, 0)
        self.finish = True
        self.error = False
        ops_file = os.path.join(self.ops_path, "ops.npy")
        db_file = os.path.join(self.ops_path, "db.npy")
        shutil.copy(os.path.join(self.ops_path, "ops%d.npy" % self.f), ops_file)
        shutil.copy(os.path.join(self.ops_path, "db%d.npy" % self.f), db_file)
        self.db = np.load(db_file, allow_pickle=True).item()
        print(self.db)
        print("Running suite2p with command:")
        cmd = f"-u -W ignore -m suite2p --ops {ops_file} --db {db_file}"
        print("python " + cmd)
        self.process.start(sys.executable, cmd.split(" "))

        #self.process.start('python -u -W ignore -m suite2p --ops "%s" --db "%s"' %
        #                   (ops_file, db_file))

    def stop(self):
        self.finish = False
        self.logfile.close()
        self.process.kill()

    def started(self):
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.cleanButton.setEnabled(False)
        save_folder = os.path.join(self.db["save_path0"], "suite2p/")
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self.logfile = open(os.path.join(save_folder, "run.log"), "a")
        dstring = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.logfile.write("\n >>>>> started run at %s" % dstring)

    def finished(self):
        self.logfile.close()
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        if self.finish and not self.error:
            self.cleanButton.setEnabled(True)
            if len(self.opslist) == 1:
                self.parent.fname = os.path.join(self.db["save_path0"], "suite2p",
                                                 "plane0", "stat.npy")
                if os.path.exists(self.parent.fname):
                    cursor.insertText("Opening in GUI (can close this window)\n")
                    io.load_proc(self.parent)
                else:
                    cursor.insertText("not opening plane in GUI (no ROIs)\n")
            else:
                cursor.insertText("BATCH MODE: %d more recordings remaining \n" %
                                  (len(self.opslist) - self.f - 1))
                self.f += 1
                if self.f < len(self.opslist):
                    self.run_S2P()
        elif not self.error:
            cursor.insertText("Interrupted by user (not finished)\n")
        else:
            cursor.insertText("Interrupted by error (not finished)\n")

        # remove current ops from processing list
        if len(self.opslist) == 1:
            del self.opslist[0]

    def save_ops(self):
        name = QFileDialog.getSaveFileName(self, "Ops name (*.npy)")
        name = name[0]
        self.save_text()
        if name:
            np.save(name, self.ops)
            print("saved current settings to %s" % (name))

    def save_default_ops(self):
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = default_ops()
        self.save_text()
        np.save(name, self.ops)
        self.ops = ops
        print("saved current settings in GUI as default ops")

    def revert_default_ops(self):
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = default_ops()
        np.save(name, self.ops)
        self.load_ops(name)
        print("reverted default ops to built-in ops")

    def save_text(self):
        for k in range(len(self.editlist)):
            key = self.keylist[k]
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys,
                                                      self.stringkeys)

    def load_ops(self, name=None):
        print("loading ops")
        if not (isinstance(name, str) and len(name) > 0):
            name = QFileDialog.getOpenFileName(self, "Open ops file (npy or json)")
            name = name[0]

        if len(name) > 0:
            ext = os.path.splitext(name)[1]
            try:
                if ext == ".npy":
                    ops = np.load(name, allow_pickle=True).item()
                elif ext == ".json":
                    with open(name, "r") as f:
                        ops = json.load(f)
                ops0 = default_ops()
                ops = {**ops0, **ops}
                for key in ops:
                    if key != "data_path" and key != "save_path" and key != "fast_disk" and key != "cleanup" and key != "save_path0" and key != "h5py":
                        if key in self.keylist:
                            self.editlist[self.keylist.index(key)].set_text(ops)
                        self.ops[key] = ops[key]
                if not "input_format" in self.ops.keys():
                    self.ops["input_format"] = "tif"
                if "data_path" in ops and len(ops["data_path"]) > 0:
                    self.data_path = ops["data_path"]
                    for n in range(9):
                        if n < len(self.data_path):
                            self.qdata[n].setText(self.data_path[n])
                        else:
                            self.qdata[n].setText("")
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(True)
                    self.listOps.setEnabled(True)
                if "h5py_key" in ops and len(ops["h5py_key"]) > 0:
                    self.h5_key = ops["h5py_key"]
                self.inputformat.currentTextChanged.connect(lambda x: x)
                self.inputformat.setCurrentText(self.ops["input_format"])
                self.inputformat.currentTextChanged.connect(self.parse_inputformat)
                if self.ops["input_format"] == "sbx":
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(False)
                    self.listOps.setEnabled(True)

                if "save_path0" in ops and len(ops["save_path0"]) > 0:
                    self.save_path = ops["save_path0"]
                    self.savelabel.setText(self.save_path)
                if "fast_disk" in ops and len(ops["fast_disk"]) > 0:
                    self.fast_disk = ops["fast_disk"]
                    self.binlabel.setText(self.fast_disk)
                if "clean_script" in ops and len(ops["clean_script"]) > 0:
                    self.ops["clean_script"] = ops["clean_script"]
                    self.cleanLabel.setText(ops["clean_script"])

            except Exception as e:
                print("could not load ops file")
                print(e)

    def load_db(self):
        print("loading db")

    def stdout_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        output = str(self.process.readAllStandardOutput(), "utf-8")
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        #self.logfile = open(os.path.join(self.save_path, "suite2p/run.log"), "a")
        self.logfile.write(output)
        #self.logfile.close()

    def stderr_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(">>>ERROR<<<\n")
        output = str(self.process.readAllStandardError(), "utf-8")
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        self.error = True
        #self.logfile = open(os.path.join(self.save_path, "suite2p/run.log"), "a")
        self.logfile.write(">>>ERROR<<<\n")
        self.logfile.write(output)

    def clean_script(self):
        name = QFileDialog.getOpenFileName(self, "Open clean up file", filter="*.py")
        name = name[0]
        if name:
            self.cleanup = True
            self.cleanScript = name
            self.cleanLabel.setText(name)
            self.ops["clean_script"] = name

    def get_folders(self):
        name = QFileDialog.getExistingDirectory(self, "Add directory to data path")
        if len(name) > 0:
            self.data_path.append(name)
            self.qdata[len(self.data_path) - 1].setText(name)
            self.qdata[len(self.data_path) - 1].setToolTip(name)
            self.runButton.setEnabled(True)
            self.listOps.setEnabled(True)
            #self.loadDb.setEnabled(False)

    def get_h5py(self):
        # used to choose file, now just choose key
        TC = TextChooser(self)
        result = TC.exec_()
        if result:
            self.h5_key = TC.h5_key
        else:
            self.h5_key = "data"

    def parse_inputformat(self):
        inputformat = self.inputformat.currentText()
        print("Input format: " + inputformat)
        if inputformat == "h5":
            # replace functionality of "old" button
            self.get_h5py()
        else:
            pass

    def save_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Save folder for data")
        if len(name) > 0:
            self.save_path = name
            self.savelabel.setText(name)
            self.savelabel.setToolTip(name)

    def bin_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Folder for binary file")
        self.fast_disk = name
        self.binlabel.setText(name)
        self.binlabel.setToolTip(name)


class LineEdit(QLineEdit):

    def __init__(self, k, key, parent=None):
        super(LineEdit, self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self, intkeys, boolkeys, stringkeys):
        key = self.key
        if key == "diameter" or key == "block_size":
            diams = self.text().replace(" ", "").split(",")
            if len(diams) > 1:
                okey = [int(diams[0]), int(diams[1])]
            else:
                okey = int(diams[0])
        elif key == "ignore_flyback":
            okey = self.text().replace(" ", "").split(",")
            for i in range(len(okey)):
                okey[i] = int(okey[i])
            if len(okey) == 1 and okey[0] == -1:
                okey = []
        else:
            if key in intkeys:
                okey = int(float(self.text()))
            elif key in boolkeys:
                okey = bool(int(self.text()))
            elif key in stringkeys:
                okey = self.text()
            else:
                okey = float(self.text())
        return okey

    def set_text(self, odict):
        if odict["type"] == list:
            self.setText([str(o)+", " if k < len(odict["default"])-1 else str(o) 
                        for k, o in enumerate(odict["default"]) 
                          ])
        else:
            self.setText(str(odict["default"]))
        

class OpsButton(QPushButton):

    def __init__(self, bid, Text, parent=None):
        super(OpsButton, self).__init__(parent)
        self.setText(Text)
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        try:
            opsdef = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                  "../ops/ops_%s.npy" % parent.opsname[bid])
            ops = np.load(opsdef, allow_pickle=True).item()
            for key in ops:
                if key in parent.keylist:
                    parent.editlist[parent.keylist.index(key)].set_text(ops)
                    parent.ops[key] = ops[key]
        except Exception as e:
            print("could not load ops file")
            print(e)


# custom vertical label
class VerticalLabel(QWidget):

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
