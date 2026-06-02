""" from kilosort GUI """

from pathlib import Path
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
import sys, os
import logging
import traceback
import io

import numpy as np
import torch
from qtpy import QtCore

from suite2p.run_s2p import logger_setup, run_s2p, get_save_folder

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

class Suite2pWorker2(QtCore.QThread):
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
            logger_setup(db['save_path'])
            run_s2p(db=db, settings=settings)
            self.finished.emit("finished")
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            self.finished.emit("error")


class QtHandler(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)

    def emit(self, record):
        record = self.format(record)
        if record:
            XStream.stdout().write(f"{record}")


class XStream(QtCore.QObject):
    _stdout = None
    _stderr = None
    messageWritten = QtCore.Signal(str)

    def flush(self):
        pass

    def fileno(self):
        raise io.UnsupportedOperation("fileno")

    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)

    @staticmethod
    def stdout():
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


def setup_logger(name):
    logger = logging.getLogger(name)
    handler = QtHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


class Suite2pWorker(QtCore.QObject):
    """Worker that runs suite2p in a separate process to avoid QThread stack limitations on macOS."""
    finished = QtCore.Signal(str)

    def __init__(self, parent, db_file, settings_file):
        super(Suite2pWorker, self).__init__()
        self.db_file = db_file
        self.settings_file = settings_file
        self.parent = parent
        self.process = None

    def start(self):
        """Start suite2p in a separate process using QProcess."""
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_output)
        self.process.finished.connect(self._on_finished)

        # Create a Python script to run suite2p
        script = f'''
import numpy as np
from suite2p.run_s2p import logger_setup, run_s2p, get_save_folder

db = np.load("{self.db_file}", allow_pickle=True).item()
settings = np.load("{self.settings_file}", allow_pickle=True).item()

logger_setup(get_save_folder(db))
run_s2p(db=db, settings=settings)
'''
        self.process.start(sys.executable, ["-c", script])

    def _on_output(self):
        """Handle output from the subprocess."""
        if self.process:
            data = self.process.readAllStandardOutput()
            text = bytes(data).decode("utf-8", errors="replace")
            print(text, end="")

    def _on_finished(self, exit_code, exit_status):
        """Handle process completion."""
        if exit_code == 0:
            self.finished.emit("finished")
        else:
            self.finished.emit("error")

    def terminate(self):
        """Terminate the subprocess if running."""
        if self.process and self.process.state() != QtCore.QProcess.NotRunning:
            self.process.terminate()

    def quit(self):
        """Stop the subprocess (alias for terminate, for QThread compatibility)."""
        self.terminate()

    def wait(self):
        """Wait for the process to finish (for compatibility)."""
        if self.process:
            self.process.waitForFinished(-1)

    def isRunning(self):
        """Check if the process is still running (for QThread compatibility)."""
        if self.process:
            return self.process.state() == QtCore.QProcess.Running
        return False