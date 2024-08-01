""" from kilosort GUI """

from pathlib import Path
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import sys, os
import logging
import traceback

import numpy as np
import torch
from qtpy import QtCore

from suite2p import run_s2p

class MyLog(QtCore.QObject):
    """
    solution from https://stackoverflow.com/questions/52479442/running-a-long-python-calculation-in-a-thread-with-logging-to-a-qt-window-cras
    """
    # create a new Signal
    # - have to be a static element
    # - class  has to inherit from QObject to be able to emit signals
    signal = QtCore.Signal(str)

    # not sure if it's necessary to implement this
    def __init__(self):
        super().__init__()

# custom logging handler that can run in separate thread, and emit all logs
# via signals/slots so they can be used to update the GUI in the main thread
class ThreadLogger(logging.Handler):
    """
    solution from https://stackoverflow.com/questions/52479442/running-a-long-python-calculation-in-a-thread-with-logging-to-a-qt-window-cras
    """
    def __init__(self):
        super().__init__()
        self.log = MyLog()

    # logging.Handler.emit() is intended to be implemented by subclasses
    def emit(self, record):
        msg = self.format(record)
        self.log.signal.emit(msg)

class Suite2pWorker(QtCore.QThread):
    finished = QtCore.Signal(str)
    
    def __init__(self, parent, db_file, settings_file):
        super(Suite2pWorker, self).__init__()
        self.db_file = db_file
        self.settings_file = settings_file
        self.parent = parent
        self.logHandler = ThreadLogger()
        
    def run(self):
        db = np.load(self.db_file, allow_pickle=True).item()
        settings = np.load(self.settings_file, allow_pickle=True).item()
        
        try:
            run_s2p(db=db, settings=settings, logging=False)
            self.finished.emit("finished")
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            self.finished.emit("error")
