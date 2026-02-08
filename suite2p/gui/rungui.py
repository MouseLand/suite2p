"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import glob, json, os, shutil, pathlib, sys, copy
from datetime import datetime
from pathlib import Path
import numpy as np

from qtpy import QtGui, QtCore 
from qtpy.QtWidgets import (QDialog, QLineEdit, QLabel, QPushButton, QWidget, 
                            QGridLayout, QMainWindow, QComboBox, QPlainTextEdit, 
                            QFileDialog, QAction, QToolButton, QStyle,
                            QListWidget, QCheckBox, QScrollArea, QAbstractItemView)
from superqt import QCollapsible 

from . import io, utils
from .rungui_utils import Suite2pWorker, XStream
from .. import default_settings, user_settings, SETTINGS_FOLDER, default_db, parameters

import logging 
logger = logging.getLogger(__name__)

### ---- this file contains helper functions for GUI and the RUN window ---- ###

def list_to_str(l):
    return ", ".join(str(l0) for l0 in l)

DB_KEYS = [  "input_format", "look_one_level_down",
             "keep_movie_raw", "nplanes", "nchannels", "swap_order",
             "functional_chan", "ignore_flyback", "save_folder",
             "batch_size",
             "h5py_key", "nwb_series", "force_sktiff"
        ]
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
        layout.addWidget(QLabel("(select multiple using ctrl)"), 0, 0, 1, 1)
        self.list = QListWidget(parent)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.list, 1, 0, 10, 1)
        self.list.addItems(parent.batch_list)

        remove = QPushButton("remove selected")
        remove.clicked.connect(lambda: self.remove_selected(parent))
        layout.addWidget(remove, 11, 0, 1, 1)

        done = QPushButton("close")
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
        
def create_input(key, SETTINGS, settings_gui, width=160):
    qlabel = QLabel(SETTINGS[key]["gui_name"])
    qlabel.setToolTip(SETTINGS[key]["description"])
    qlabel.setFixedWidth(width)
    qlabel.setAlignment(QtCore.Qt.AlignRight)
    if SETTINGS[key]["type"] == bool:
        settings_gui[key] = QCheckBox()
        settings_gui[key].setChecked(SETTINGS[key]["default"])
    else:
        settings_gui[key] = QLineEdit()
        settings_gui[key].setFixedWidth(100)
        if SETTINGS[key]["default"] is not None:
            if key in COMBO_KEYS:
                settings_gui[key] = QComboBox()
                strs = SETTINGS[key]["description"].split("[")[1].split("]")[0].split(", ")
                strs = [s[1:-1] for s in strs]
                settings_gui[key].addItems(strs)
                settings_gui[key].setFixedWidth(100)
            elif SETTINGS[key]["type"] == list or SETTINGS[key]["type"] == tuple:
                settings_gui[key].setText(list_to_str(SETTINGS[key]["default"]))
            else:
                settings_gui[key].setText(str(SETTINGS[key]["default"]))
    settings_gui[key].setToolTip(SETTINGS[key]["description"])
    return qlabel 

def get_settings(SETTINGS, settings_gui, settings=None):
    if settings is None:
        settings = user_settings()
    for key in settings_gui.keys():
        if "gui_name" not in SETTINGS[key]:
            settings[key] = {}
            settings[key] = get_settings(SETTINGS[key], settings_gui[key], settings=settings[key])
        else:
            if SETTINGS[key]["type"] == bool:
                settings[key] = settings_gui[key].isChecked()
            else:
                if key in COMBO_KEYS:
                    settings[key] = settings_gui[key].currentText()
                elif len(settings_gui[key].text()) > 0:
                    if SETTINGS[key]["type"] == list or SETTINGS[key]["type"] == tuple:
                        settings[key] = [float(x) for x in settings_gui[key].text().split(",")]
                    elif SETTINGS[key]["type"] == str:
                        settings[key] = settings_gui[key].text()
                    elif SETTINGS[key]["type"] == dict:
                        # todo : add dict parsing
                        pass
                    else:
                        # convert to int or float
                        settings[key] = SETTINGS[key]["type"](settings_gui[key].text())
                else:
                    settings[key] = None
    return settings

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
                        db[key] = folder if len(folder) > 0 else None
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

def set_settings(SETTINGS, settings_gui):
    for key in settings_gui.keys():
        if "gui_name" not in SETTINGS[key]:
            set_settings(SETTINGS[key], settings_gui[key])
        elif SETTINGS[key]["type"] == bool:
            settings_gui[key].setChecked(SETTINGS[key]["default"])
        else:
            if SETTINGS[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [settings_gui[key].itemText(i) for i in range(settings_gui[key].count())]
                    settings_gui[key].setCurrentIndex(all_items.index(SETTINGS[key]["default"]))
                elif SETTINGS[key]["type"] == list or SETTINGS[key]["type"] == tuple:
                    settings_gui[key].setText(list_to_str(SETTINGS[key]["default"]))
                else:
                    settings_gui[key].setText(str(SETTINGS[key]["default"]))
            else:
                settings_gui[key].setText("")

def set_db(DB, db_gui, parent=None):
    for key in db_gui.keys():
        if DB[key]["type"] == bool:
            db_gui[key].setChecked(DB[key]["default"])
        elif key in FILE_KEYS:
            if key == "data_path":
                for i in range(9):
                    if i < len(DB[key]["default"]):
                        db_gui[key][i].setText(DB[key]["default"][i])
                        if parent is not None:
                            parent.data_path_checkboxes[i].setVisible(True)
                    else:
                        db_gui[key][i].setText("")
                        if parent is not None:
                            parent.data_path_checkboxes[i].setVisible(False)
                # Enable/disable remove button based on whether paths exist
                if parent is not None and len(DB[key]["default"]) > 0:
                    parent.remove_path_btn.setEnabled(True)
            else:
                db_gui[key].setText(DB[key]["default"])
        else:
            if DB[key]["default"] is not None:
                if key in COMBO_KEYS:
                    all_items = [db_gui[key].itemText(i) for i in range(db_gui[key].count())]
                    db_gui[key].setCurrentIndex(all_items.index(DB[key]["default"]))
                elif DB[key]["type"] == list or DB[key]["type"] == tuple:
                    db_gui[key].setText(list_to_str(DB[key]["default"]))
                else:
                    db_gui[key].setText(str(DB[key]["default"]))
            else:
                db_gui[key].setText("")

def set_default_db(DB, db):
    for key in DB.keys():
        DB[key]["default"] = db[key]
        del db[key]

def set_default_settings(SETTINGS, settings):
    for key in SETTINGS.keys():
        if "gui_name" not in SETTINGS[key]:
            set_default_settings(SETTINGS[key], settings[key])
        else:
            SETTINGS[key]["default"] = settings[key]   
            del settings[key]

### custom QMainWindow which allows user to fill in settings and run suite2p!
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
        # initial settings values
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
        self.SETTINGS = parameters.SETTINGS
        parameters.add_descriptions(self.SETTINGS)
        self.DB = parameters.DB
        parameters.add_descriptions(self.DB, dstr="db")
        # remove keys not in GUI
        remove_keys = [key for key in self.DB if key not in FILE_KEYS and 
                           key not in DB_KEYS]
        for key in remove_keys:
            del self.DB[key]
        set_default_settings(self.SETTINGS, user_settings())

        self.create_db_settings_inputs()

        # self.load_settings(filename=".../suite2p/db.npy")

    def create_menu_bar(self):
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        self.loadOps = QAction("&Load db or settings or ops file")
        self.loadOps.triggered.connect(lambda: self.load_settings(filename=None))
        self.loadOps.setEnabled(True)
        file_menu.addAction(self.loadOps)
        
        self.saveDef = QAction("&Save settings as user-default")
        self.saveDef.triggered.connect(self.save_default_settings)
        self.saveDef.setEnabled(True)
        file_menu.addAction(self.saveDef)
        
        self.revertDef = QAction("Revert default settings to built-in")
        self.revertDef.triggered.connect(self.revert_default_settings)
        self.revertDef.setEnabled(True)
        file_menu.addAction(self.revertDef)

        self.saveOps = QAction("Save current settings to file")
        self.saveOps.triggered.connect(self.save_settings)
        self.saveOps.setEnabled(True)
        file_menu.addAction(self.saveOps)

        file_submenu = file_menu.addMenu("Load example settings")
        self.onePOps = QAction("1P imaging")
        self.onePOps.triggered.connect(lambda: 
            self.load_settings(filename=SETTINGS_FOLDER / "settings_1P.npy"))
        file_submenu.addAction(self.onePOps)
        self.dendriteOps = QAction("dendrites / axons")
        self.dendriteOps.triggered.connect(lambda: 
            self.load_settings(filename=SETTINGS_FOLDER / "settings_dendrite.npy"))
        file_submenu.addAction(self.dendriteOps)

        batch_menu = main_menu.addMenu("&Batch processing")
        self.addToBatch = QAction("Save db+settings to batch list")
        self.addToBatch.triggered.connect(self.add_batch)
        self.addToBatch.setEnabled(False)
        batch_menu.addAction(self.addToBatch)

        self.batchList = QAction("View/edit batch list")
        self.batchList.triggered.connect(self.view_batch)
        #self.batchList.setEnabled(False)
        batch_menu.addAction(self.batchList)

    def view_batch(parent):
        # will return
        BV = BatchView(parent)
        result = BV.exec_()

    def load_settings(self, filename=None):
        if filename is None:
            name = QFileDialog.getOpenFileName(self, "Open settings/db/ops file (npy or json)")
            filename = name[0]

        if filename is None:
            settings = user_settings() 
            db, settings_in = None, None
        else:
            if isinstance(filename, pathlib.Path):
                filename = str(filename)
            ext = os.path.splitext(filename)[1]
            if ext == ".json":
                with open(filename, "r") as f:
                    settings_in = json.load(f)
            else:
                settings_in = np.load(filename, allow_pickle=True).item()
            db, settings, settings_in = parameters.convert_settings_orig(settings_in, 
                                                          db=get_db(self.DB, self.db_gui, self.extra_keys), 
                                                          settings=get_settings(self.SETTINGS, self.settings_gui))
        set_default_settings(self.SETTINGS, settings)
        set_settings(self.SETTINGS, self.settings_gui)
        if db is not None:
            set_default_db(self.DB, db) # check if any db keys are in settings
            set_db(self.DB, self.db_gui, parent=self)
            if len(self.DB["data_path"]) > 0:
                self.runButton.setEnabled(True)
                self.addToBatch.setEnabled(True)

        if settings_in is not None:
            for key in settings_in.keys():
                print(f"key {key} not in GUI, saved in backend in DB for running pipeline")
                self.extra_keys[key] = settings_in[key]
    
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
        set_db(self.DB, self.db_gui, parent=self)

        for key in db.keys():
            print(f"key {key} not in GUI, saved in backend in DB for running pipeline")
            self.extra_keys[key] = db[key]

    def create_db_settings_inputs(self):
        self.db_gui = {}
        self.settings_gui = {}

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
                self.data_path_checkboxes = [QCheckBox() for _ in range(9)]
                for i in range(9):
                    self.layout.addWidget(self.data_path_checkboxes[i], b, 0, 1, 1)
                    self.layout.addWidget(self.db_gui[key][i], b, 1, 1, cw-1)
                    self.data_path_checkboxes[i].setVisible(False)  # Initially hidden
                    b += 1
                # Add remove button for selected data paths
                self.remove_path_btn = QPushButton("remove selected data_paths")
                self.remove_path_btn.clicked.connect(self.remove_selected_data_paths)
                self.remove_path_btn.setToolTip("Remove checked data_paths from the list")
                self.remove_path_btn.setEnabled(False)  # Initially disabled
                self.layout.addWidget(self.remove_path_btn, b, 0, 1, cw)
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
        
        self.textEdit = QPlainTextEdit(self)
        self.layout.addWidget(self.textEdit, n0 + 1, 0, 16, cw+2)

        XStream.stdout().messageWritten.connect(self.update_text)
        XStream.stderr().messageWritten.connect(self.update_text)

        qlabel = QLabel("file settings")
        qlabel.setStyleSheet(bigfont)
        qlabel.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(qlabel, 0, cw, 1, 2)
        b = 1
        for key in DB_KEYS[2:]:
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
            qlabel = create_input(key, self.SETTINGS, self.settings_gui)
            self.layout.addWidget(qlabel, b, cw, 1, 1)
            self.layout.addWidget(self.settings_gui[key], b, cw+1, 1, 1)
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


        labels = ["run", "io", "registration", "detection", "classification", 
                  "extraction", "dcnv_preprocess"]
        b = 0
        for label in labels:
            qbox = QCollapsible(f"{label} settings")
            qbox._toggle_btn.setStyleSheet(bigfont)
            qboxG = QGridLayout()
            _content = QWidget()
            _content.setLayout(qboxG)
            bl = 0
            self.settings_gui[label] = {}
            for key in self.SETTINGS[label].keys():
                if "gui_name" in self.SETTINGS[label][key]:
                    qlabel = create_input(key, self.SETTINGS[label], self.settings_gui[label])
                    qboxG.addWidget(qlabel, bl, 0, 1, 1)
                    qboxG.addWidget(self.settings_gui[label][key], bl, 1, 1, 1)
                    bl+=1
                else:
                    self.settings_gui[label][key] = {}
                    qboxA = QCollapsible(key)
                    qboxAG = QGridLayout()
                    _contentA = QWidget()
                    _contentA.setLayout(qboxAG)
                    bl2 = 0
                    for key2 in self.SETTINGS[label][key].keys():
                        qlabel = create_input(key2, self.SETTINGS[label][key], self.settings_gui[label][key])
                        qboxAG.addWidget(qlabel, bl2, 0, 1, 1)
                        qboxAG.addWidget(self.settings_gui[label][key][key2], bl2, 1, 1, 1)
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
        
         
        
    def save_db_settings(self):
        db = get_db(self.DB, self.db_gui, self.extra_keys)
        settings = get_settings(self.SETTINGS, self.settings_gui)
        save_path0 = db["save_path0"] if db["save_path0"] is not None else db["data_path"][0]
        self.save_path = os.path.join(save_path0, db["save_folder"])
        settings_file = os.path.join(self.save_path, "settings.npy")
        db_file = os.path.join(self.save_path, "db.npy")
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        np.save(settings_file, settings)
        np.save(db_file, db)
        
    def add_batch(self):
        self.save_db_settings()
        self.batch_list.append(self.save_path)
        L = len(self.batch_list)
        
        # clear db and extra_keys
        set_default_db(self.DB, default_db())
        set_db(self.DB, self.db_gui, parent=self)
        self.extra_keys = {}
        
        # and enable the run button
        self.runButton.setEnabled(True)

    def run_S2P(self):
        self.finish = True
        self.error = False
        
        if len(self.batch_list) == 0:
            self.save_db_settings()            
        else:
            self.save_path = self.batch_list[self.ibatch]

        settings_file = os.path.join(self.save_path, "settings.npy")
        db_file = os.path.join(self.save_path, "db.npy")

        print("Running suite2p with command:")
        cmd = f"-m suite2p --settings {settings_file} --db {db_file} --verbose"
        print("python " + cmd)

        self.worker = Suite2pWorker(self, db_file=db_file, settings_file=settings_file)
        self.worker.finished.connect(self.finished)

        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)

        self.worker.start()
        

    # define a new Slot, that receives a string
    @QtCore.Slot(str)
    def update_text(self, log_text):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(log_text)
        #self.textEdit.moveCursor(QtGui.QTextCursor.End)
        #self.textEdit.appendPlainText(log_text)
        self.textEdit.ensureCursorVisible()

    def stop(self):
        self.finish = False
        self.worker.quit()
        self.worker.terminate()
        self.finished(msg=None)

    @QtCore.Slot(str)
    def finished(self, msg=None):
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        print(msg)
        if msg is None:
            self.textEdit.appendPlainText("Interrupted by user (not finished)\n")
        elif msg == "finished":
            if len(self.batch_list) == 0:
                self.parent.fname = os.path.join(self.save_path,
                                                 "plane0", "stat.npy")
                if os.path.exists(self.parent.fname):
                    self.textEdit.appendPlainText("Opening in GUI (can close this window)\n")
                    io.load_proc(self.parent)
                else:
                    self.textEdit.appendPlainText("not opening plane in GUI (no ROIs)\n")
            else:
                self.textEdit.appendPlainText("BATCH MODE: %d more recordings remaining \n" %
                                  (len(self.batch_list) - self.ibatch - 1))
                self.ibatch += 1
                if self.ibatch < len(self.batch_list):
                    self.run_S2P()
        elif msg=="error":
            self.textEdit.appendPlainText("Interrupted by error (not finished)\n")
        
    def save_settings(self):
        name = QFileDialog.getSaveFileName(self, "Settings name (*.npy)")
        settings = get_settings(self.SETTINGS, self.settings_gui)
        if len(name) > 0 and name[0] is not None:
            name = name[0]
            np.save(name, settings)
            print(f'saved current settings to {name}')

    def save_default_settings(self):
        name = SETTINGS_FOLDER / "settings_user.npy"
        settings = get_settings(self.SETTINGS, self.settings_gui)
        np.save(name, settings)
        print("saved current settings in GUI as default user settings")

    def revert_default_settings(self):
        if (SETTINGS_FOLDER / "settings_user.npy").exists():
            os.remove(SETTINGS_FOLDER / "settings_user.npy")
        print("removing default user settings, reverting to built-in defaults")
        
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

    def get_folders(self, ftype="data_path", name=None):
        if name is None:
            name = QFileDialog.getExistingDirectory(self, "choose folder")
        if len(name) > 0:
            print(f"adding {name} to {ftype}")
            if ftype=="data_path":
                for i in range(9):
                    if len(self.db_gui["data_path"][i].text()) == 0:
                        self.db_gui["data_path"][i].setText(name)
                        self.db_gui["data_path"][i].setToolTip(name)
                        self.data_path_checkboxes[i].setVisible(True)
                        break
                self.runButton.setEnabled(True)
                self.addToBatch.setEnabled(True)
                self.remove_path_btn.setEnabled(True)
            else:
                self.db_gui[ftype].setText(name)
                self.db_gui[ftype].setToolTip(name)

    def remove_selected_data_paths(self):
        # Collect remaining paths (unchecked ones)
        remaining_paths = []
        for i in range(9):
            if not self.data_path_checkboxes[i].isChecked():
                path_text = self.db_gui["data_path"][i].text()
                if len(path_text) > 0:
                    remaining_paths.append(path_text)
            else:
                path_text = self.db_gui["data_path"][i].text()
                if len(path_text) > 0:
                    print(f"removing {path_text} from data_path")

        # Clear all fields, checkboxes, and hide them
        for i in range(9):
            self.db_gui["data_path"][i].setText("")
            self.db_gui["data_path"][i].setToolTip("")
            self.data_path_checkboxes[i].setChecked(False)
            self.data_path_checkboxes[i].setVisible(False)

        # Repopulate with remaining paths (compacted at top, no gaps)
        for i, path in enumerate(remaining_paths):
            self.db_gui["data_path"][i].setText(path)
            self.db_gui["data_path"][i].setToolTip(path)
            self.data_path_checkboxes[i].setVisible(True)

        # Disable buttons if no data paths remain
        if len(remaining_paths) == 0:
            self.runButton.setEnabled(False)
            self.addToBatch.setEnabled(False)
            self.remove_path_btn.setEnabled(False)
