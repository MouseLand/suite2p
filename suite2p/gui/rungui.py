"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import glob, json, os, shutil, pathlib, sys, copy
from datetime import datetime
from pathlib import Path
import numpy as np

from qtpy import QtGui, QtCore 
from qtpy.QtWidgets import (QDialog, QLineEdit, QLabel, QPushButton, QWidget, 
                            QGridLayout, QMainWindow, QComboBox, QTextEdit, 
                            QFileDialog, QAction, QToolButton, QStyle,
                            QListWidget, QCheckBox, QScrollArea, QAbstractItemView)
from superqt import QCollapsible 

from . import io, utils
from .. import default_ops, user_ops, OPS_FOLDER, default_db, parameters

### ---- this file contains helper functions for GUI and the RUN window ---- ###

def list_to_str(l):
    return ", ".join(str(l0) for l0 in l)

FILE_KEYS = ["data_path", "save_path0", "fast_disk"]
COMBO_KEYS = ["input_format", "algorithm", "img"]





### custom QDialog with an editable list 
class BatchView(QDialog):
    def __init__(self, parent=None):
        super(BatchView, self).__init__(parent)
        self.setGeometry(300, 300, 600, 320)
        self.setWindowTitle("Batch list")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        layout.addWidget(QLabel('(select multiple using ctrl)'), 0, 0, 1, 1)
        self.list = QListWidget(parent)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.list, 1, 0, 10, 1)
        self.list.addItems(parent.batch_list)

        remove = QPushButton('remove selected')
        remove.clicked.connect(lambda: self.remove_selected(parent))
        layout.addWidget(done, 11, 0, 1, 1)

        done = QPushButton('close')
        done.clicked.connect(self.exit_list)
        layout.addWidget(done, 11, 1, 1, 1)

    def remove_selected(self, parent):
        for item in self.list.selectedItems():
            parent.batch_list.removeItem(item.text())
            print("removed", item.text())
        parent.list.clear()
        parent.list.addItems(parent.batch_list)

    def exit_list(self):
        self.accept()
        

def create_input(key, OPS, ops_gui, width=160):
    qlabel = QLabel(OPS[key]["gui_name"])
    qlabel.setToolTip(OPS[key]["description"])
    qlabel.setFixedWidth(width)
    qlabel.setAlignment(QtCore.Qt.AlignRight)
    if OPS[key]["type"] == bool:
        ops_gui[key] = QCheckBox()
        ops_gui[key].setChecked(OPS[key]["default"])
    else:
        ops_gui[key] = QLineEdit()
        ops_gui[key].setFixedWidth(100)
        if OPS[key]["default"] is not None:
            if key in COMBO_KEYS:
                ops_gui[key] = QComboBox()
                strs = OPS[key]["description"].split("[")[1].split("]")[0].split(", ")
                strs = [s[1:-1] for s in strs]
                ops_gui[key].addItems(strs)
                ops_gui[key].setFixedWidth(100)
            elif OPS[key]["type"] == list or OPS[key]["type"] == tuple:
                ops_gui[key].setText(list_to_str(OPS[key]["default"]))
            else:
                ops_gui[key].setText(str(OPS[key]["default"]))
    ops_gui[key].setToolTip(OPS[key]["description"])
    return qlabel 

def get_ops(OPS, ops_gui, ops=None):
    if ops is None:
        ops = user_ops()
    for key in ops_gui.keys():
        if "gui_name" not in OPS[key]:
            ops[key] = {}
            ops[key] = get_ops(OPS[key], ops_gui[key], ops=ops[key])
        else:
            if OPS[key]["type"] == bool:
                ops[key] = ops_gui[key].isChecked()
            else:
                if key in COMBO_KEYS:
                    ops[key] = ops_gui[key].currentText()
                elif len(ops_gui[key].text()) > 0:
                    if OPS[key]["type"] == list or OPS[key]["type"] == tuple:
                        ops[key] = [float(x) for x in ops_gui[key].text().split(",")]
                    elif OPS[key]["type"] == str:
                        ops[key] = ops_gui[key].text()
                    elif OPS[key]["type"] == dict:
                        # todo : add dict parsing
                        pass
                    else:
                        # convert to int or float
                        ops[key] = OPS[key]["type"](ops_gui[key].text())
                else:
                    ops[key] = None
    return ops

def get_db(DB, db_gui, extra_keys={}):
    db = default_db()
    for key in db_gui.keys():
        if DB[key]["type"] == bool:
            db[key] = db_gui[key].isChecked()
        else:
            if key in COMBO_KEYS:
                db[key] = db_gui[key].currentText()
            elif key in FILE_KEYS:
                if key == "data_path":
                    db[key] = []
                    for i in range(9):
                        folder = db_gui[key][i].text()
                        if len(folder) > 0:
                            db[key].append(folder)
                else:
                    folder = db_gui[key].text()
                    if key == "save_path0":
                        db[key] = folder if len(folder) > 0 else db["data_path"][0]
                    else:
                        db[key] = folder if len(folder) > 0 else None

            elif len(db_gui[key].text()) > 0:
                if DB[key]["type"] == list or DB[key]["type"] == tuple:
                    db[key] = [float(x) for x in db_gui[key].text().split(",")]
                elif DB[key]["type"] == str:
                    db[key] = db_gui[key].text()
                else:
                    # convert to int or float
                    db[key] = DB[key]["type"](db_gui[key].text())
            else:
                db[key] = None

    for key in extra_keys.keys():
        db[key] = extra_keys[key]
    return db

def set_ops(OPS, ops_gui):
    for key in ops_gui.keys():
        if "gui_name" not in OPS[key]:
            set_ops(OPS[key], ops_gui[key])
        elif OPS[key]["type"] == bool:
            ops_gui[key].setChecked(OPS[key]["default"])
        else:
            if OPS[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [ops_gui[key].itemText(i) for i in range(ops_gui[key].count())]
                    ops_gui[key].setCurrentIndex(all_items.index(OPS[key]["default"]))
                elif OPS[key]["type"] == list or OPS[key]["type"] == tuple:
                    ops_gui[key].setText(list_to_str(OPS[key]["default"]))
                else:
                    ops_gui[key].setText(str(OPS[key]["default"]))
            else:
                ops_gui[key].setText("")

def set_db(DB, db_gui):
    for key in db_gui.keys():
        if DB[key]["type"] == bool:
            db_gui[key].setChecked(DB[key]["default"])
        elif key in FILE_KEYS:
            if key == "data_path":
                for i in range(9):
                    if i < len(DB[key]["default"]):
                        db_gui[key][i].setText(DB[key]["default"][i])
                    else:
                        db_gui[key][i].setText("")
            else:
                db_gui[key].setText(DB[key]["default"])
        else:
            if DB[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [db_gui[key].itemText(i) for i in range(db_gui[key].count())]
                    db_gui[key].setCurrentIndex(all_items.index(DB[key]["default"]))
                elif DB[key]["type"] == list or DB[key]["type"] == tuple:
                    print(key, DB[key]["default"])
                    db_gui[key].setText(list_to_str(DB[key]["default"]))
                else:
                    db_gui[key].setText(str(DB[key]["default"]))
            else:
                db_gui[key].setText("")

def set_default_db(DB, db):
    for key in DB.keys():
        DB[key]["default"] = db[key]

def set_default_ops(OPS, ops):
    for key in OPS.keys():
        if "gui_name" not in OPS[key]:
            set_default_ops(OPS[key], ops[key])
        else:
            OPS[key]["default"] = ops[key]   

### custom QMainWindow which allows user to fill in ops and run suite2p!
class RunWindow(QMainWindow):

    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(65, 65, 1200, 900)
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
        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        self.ibatch = 0
        self.batch_list = []

        self.extra_keys = {}
        self.create_menu_bar()

        self.db = default_db()
        self.OPS = parameters.OPS
        parameters.add_descriptions(self.OPS)
        self.DB = parameters.DB
        parameters.add_descriptions(self.DB, dstr="db")
        set_default_ops(self.OPS, user_ops())
        self.create_db_ops_inputs()

    def create_menu_bar(self):
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        self.loadOps = QAction("&Load ops file")
        self.loadOps.triggered.connect(self.load_ops)
        self.loadOps.setEnabled(True)
        file_menu.addAction(self.loadOps)
        
        self.saveDef = QAction("&Save ops as user-default")
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
        self.onePOps.triggered.connect(lambda: 
            self.load_ops(filename=OPS_FOLDER / "ops_1P.npy"))
        file_submenu.addAction(self.onePOps)
        self.dendriteOps = QAction("dendrites / axons")
        self.dendriteOps.triggered.connect(lambda: 
            self.load_ops(filename=OPS_FOLDER / "ops_dendrite.npy"))
        file_submenu.addAction(self.dendriteOps)

        batch_menu = main_menu.addMenu("&Batch processing")
        self.addToBatch = QAction("Save db+ops to batch list")
        self.addToBatch.triggered.connect(self.add_batch)
        self.addToBatch.setEnabled(False)
        batch_menu.addAction(self.addToBatch)

        self.batchList = QAction("View/edit batch list")
        self.batchList.triggered.connect(self.add_batch)
        self.batchList.setEnabled(False)
        batch_menu.addAction(self.batchList)

    def load_ops(self, filename=None):
        if filename is None:
            name = QFileDialog.getOpenFileName(self, "Open ops file (npy or json)")
            filename = name[0]

        if filename is None:
            ops = user_ops() 
        else:
            if isinstance(filename, pathlib.Path):
                filename = str(filename)
            ext = os.path.splitext(filename)[1]
            if ext == ".json":
                with open(filename, "r") as f:
                    ops = json.load(f)
            else:
                ops = np.load(filename, allow_pickle=True).item()
        set_default_ops(self.OPS, ops)
        set_default_db(self.DB, ops) # check if any db keys are in ops
        set_ops(self.OPS, self.ops_gui)
        set_db(self.DB, self.db_gui)

        for key in ops.keys():
            if key not in self.OPS.keys() or key not in self.DB.keys():
                print(f"key {key} not in GUI, saved in backend in DB for running pipeline")
                self.extra_keys[key] = ops[key]
    
    def load_db(self, filename=None):
        if filename is None:
            name = QFileDialog.getOpenFileName(self, "Open db file (npy or json)")
            filename = name[0]

        if filename is not None:
            if isinstance(filename, pathlib.Path):
                filename = str(filename)
            ext = os.path.splitext(filename)[1]
            if ext == ".json":
                with open(filename, "r") as f:
                    db = json.load(f)
            else:
                db = np.load(filename, allow_pickle=True).item()
        set_default_db(self.DB, db) 
        set_db(self.DB, self.db_gui)

        for key in db.keys():
            if key not in self.DB.keys():
                if key not in self.OPS.keys():
                    print(f"key {key} not in GUI, saved in backend in DB for running pipeline")
                    self.extra_keys[key] = db[key]
                else:
                    print(f"key {key} is an OPS key, load with ops file")

    def create_db_ops_inputs(self):
        self.db_gui = {}
        self.ops_gui = {}

        XLfont = "QLabel {font-size: 12pt; font: Arial; font-weight: bold;}"
        bigfont = "QLabel {font-size: 10pt; font: Arial; font-weight: bold;}"
        qlabel = QLabel("File paths")
        qlabel.setStyleSheet(XLfont)
        self.layout.addWidget(qlabel, 0, 0, 1, 1)
        
        # data_path
        for b, key in enumerate(["input_format", "look_one_level_down"]):
            qlabel = create_input(key, self.DB, self.db_gui, width=80 + b*50)
            self.layout.addWidget(qlabel, 1, 2*b, 1, 1)
            self.layout.addWidget(self.db_gui[key], 1, 2*b+1, 1, 1)
            
        cw = 5
        b = 2
        self.path_btns = []
        kinfos = ["folders", "(default is 1st data_path)", "(default is save_path0)"]
        for kinfo, key in zip(kinfos, FILE_KEYS):
            self.path_btns.append(QPushButton(f"add {key} {kinfo}"))
            self.path_btns[-1].clicked.connect(lambda state, x=key: self.get_folders(x))
            self.path_btns[-1].setToolTip(self.DB["data_path"]["description"])
            self.layout.addWidget(self.path_btns[-1], b, 0, 1, cw)
            b += 1
            if key!="data_path":
                self.db_gui[key] = QLabel("") 
                self.layout.addWidget(self.db_gui[key], b, 0, 1, cw)
                b += 1
            else:
                self.db_gui[key] = [QLabel("") for _ in range(9)]
                for i in range(9):
                    self.layout.addWidget(self.db_gui[key][i], b, 0, 1, cw)
                    b += 1

        n0 = 17
        self.runButton = QPushButton("RUN SUITE2P")
        self.runButton.clicked.connect(self.run_S2P)
        self.layout.addWidget(self.runButton, n0, 0, 1, 1)
        self.runButton.setEnabled(False)
        # stop process
        self.stopButton = QPushButton("STOP")
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, n0, 1, 1, 1)
        self.stopButton.clicked.connect(self.stop)
        
        self.textEdit = QTextEdit()
        self.layout.addWidget(self.textEdit, n0 + 1, 0, 16, cw+2)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(self.finished)


        self.dbkeys = [
             "keep_movie_raw", "nplanes", "nchannels",
             "functional_chan", "ignore_flyback", "save_folder",
             "h5py_key", "nwb_series", "force_sktiff"
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
            "torch_device", "tau", "fs", "diameter", "classifier_path", "use_builtin_classifier"
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
            bl = 0
            self.ops_gui[label] = {}
            for key in self.OPS[label].keys():
                if "gui_name" in self.OPS[label][key]:
                    qlabel = create_input(key, self.OPS[label], self.ops_gui[label])
                    qboxG.addWidget(qlabel, bl, 0, 1, 1)
                    qboxG.addWidget(self.ops_gui[label][key], bl, 1, 1, 1)
                    bl+=1
                else:
                    self.ops_gui[label][key] = {}
                    qboxA = QCollapsible(key)
                    qboxAG = QGridLayout()
                    _contentA = QWidget()
                    _contentA.setLayout(qboxAG)
                    bl2 = 0
                    for key2 in self.OPS[label][key].keys():
                        qlabel = create_input(key2, self.OPS[label][key], self.ops_gui[label][key])
                        qboxAG.addWidget(qlabel, bl2, 0, 1, 1)
                        qboxAG.addWidget(self.ops_gui[label][key][key2], bl2, 1, 1, 1)
                        bl2 += 1
                    qboxA.setContent(_contentA)
                    qboxG.addWidget(qboxA, bl, 0, 1, 2)
                    bl += 1
            qbox.setContent(_content)
            if label=="run":
                qbox.expand(animate=False)
            else:
                qbox.collapse(animate=False)
            layoutScroll.addWidget(qbox, b, 0, 1, 1)
            b+=1
        layoutScroll.addWidget(QLabel(""), b, 0, 1, 1)
        layoutScroll.setColumnStretch(0, b)       
        
    def save_db_ops(self):
        db = get_db(self.DB, self.db_gui, self.extra_keys)
        ops = get_ops(self.OPS, self.ops_gui)
        save_path0 = db["save_path0"]
        self.save_path = os.path.join(save_path0, db["save_folder"])
        ops_file = os.path.join(self.save_path, "ops.npy")
        db_file = os.path.join(self.save_path, "db.npy")
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        np.save(ops_file, ops)
        np.save(db_file, db)
        
    def add_batch(self):
        self.save_db_ops()
        self.batch_list.append(self.save_path)
        L = len(self.batch_list)
        
        # clear db and extra_keys
        set_default_db(self.DB, default_db())
        set_db(self.DB, self.db_gui)
        self.extra_keys = {}
        
        # and enable the run button
        self.runButton.setEnabled(True)
        self.removeBatch.setEnabled(True)
        self.listBatch.setEnabled(True)

    def run_S2P(self):
        self.finish = True
        self.error = False
        
        if len(self.batch_list) == 0:
            self.save_db_ops()            
        else:
            self.save_path = self.batch_list[self.ibatch]

        ops_file = os.path.join(self.save_path, "ops.npy")
        db_file = os.path.join(self.save_path, "db.npy")
        

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
        self.logfile = open(os.path.join(self.save_path, "run.log"), "a")
        dstring = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.logfile.write("\n >>>>> started run at %s" % dstring)

    def finished(self):
        self.logfile.close()
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        if self.finish and not self.error:
            if len(self.batch_list) == 0:
                self.parent.fname = os.path.join(self.db["save_path0"], "suite2p",
                                                 "plane0", "stat.npy")
                if os.path.exists(self.parent.fname):
                    cursor.insertText("Opening in GUI (can close this window)\n")
                    io.load_proc(self.parent)
                else:
                    cursor.insertText("not opening plane in GUI (no ROIs)\n")
            else:
                cursor.insertText("BATCH MODE: %d more recordings remaining \n" %
                                  (len(self.batch_list) - self.ibatch - 1))
                self.ibatch += 1
                if self.ibatch < len(self.batch_list):
                    self.run_S2P()
        elif not self.error:
            cursor.insertText("Interrupted by user (not finished)\n")
        else:
            cursor.insertText("Interrupted by error (not finished)\n")
        
    def save_ops(self):
        name = QFileDialog.getSaveFileName(self, "Ops name (*.npy)")
        ops = get_ops(self.OPS, self.ops_gui)
        if name:
            np.save(name, ops)
            print("saved current settings to %s" % (name))

    def save_default_ops(self):
        name = OPS_FOLDER / "ops_user.npy"
        ops = get_ops(self.OPS, self.ops_gui)
        np.save(name, ops)
        print("saved current settings in GUI as default user ops")

    def revert_default_ops(self):
        shutil.remove(OPS_FOLDER / "ops_user.npy")
        print("removing default user ops, reverting to built-in")

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

    def get_folders(self, ftype="data_path"):
        name = QFileDialog.getExistingDirectory(self, "choose folder")
        if len(name) > 0:
            print(f"adding {name} to {ftype}")
            if ftype=="data_path":
                for i in range(9):
                    if len(self.db_gui["data_path"][i].text()) == 0:
                        self.db_gui["data_path"][i].setText(name)
                        self.db_gui["data_path"][i].setToolTip(name)
                        break 
                self.runButton.setEnabled(True)
                self.batchList.setEnabled(True)
            else:
                self.db_gui[ftype].setText(name)
                self.db_gui[ftype].setToolTip(name)