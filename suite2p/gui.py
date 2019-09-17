from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import console
import sys, json, os, pickle, time, shutil
import numpy as np
from suite2p import fig
from suite2p import run_s2p

### ---- this file contains helper functions for GUI and the RUN window ---- ###

# type in h5py key
class TextChooser(QtGui.QDialog):
    def __init__(self,parent=None):
        super(TextChooser, self).__init__(parent)
        self.setGeometry(300,300,180,100)
        self.setWindowTitle('h5 key')
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        self.win.setLayout(layout)
        self.qedit = QtGui.QLineEdit('data')
        layout.addWidget(QtGui.QLabel('h5 key for data field'),0,0,1,3)
        layout.addWidget(self.qedit,1,0,1,2)
        done = QtGui.QPushButton('OK')
        done.clicked.connect(self.exit_list)
        layout.addWidget(done,2,1,1,1)

    def exit_list(self):
        self.h5_key = self.qedit.text()
        self.accept()

### choose files for batch processing
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
        loadcell = QtGui.QPushButton('Load ops.npy / db.npy')
        loadcell.resize(200,50)
        parent.opslist = []
        loadcell.clicked.connect(lambda: self.load_ops(parent))
        layout.addWidget(loadcell,0,0,1,1)
        layout.addWidget(QtGui.QLabel('(any ops not in loaded file will be pulled from GUI)'),1,0,1,1)
        #layout.addWidget(QtGui.QLabel('select multiple files using ctrl'),2,0,1,1)
        self.list = QtGui.QListWidget(parent)
        layout.addWidget(self.list,2,0,5,4)
        #self.list.resize(450,250)
        done = QtGui.QPushButton('OK')
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done,8,3,1,1)

    def load_ops(self, parent):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open ops.npy / db.npy file',filter='*.npy')
        if name:
            try:
                ops = np.load(name[0], allow_pickle=True).item()
                badfile = True
                if 'data_path' in ops and len(ops['data_path'])>0:
                    badfile = False
                elif 'h5py' in ops and len(ops['h5py']) > 0:
                    badfile = False
                if badfile:
                    QtGui.QMessageBox.information(self, 'lacks any file paths')
                else:
                    self.list.addItem(name[0])
                    parent.opslist.append(name[0])
            except (OSError, RuntimeError, TypeError, NameError):
                QtGui.QMessageBox.information(self, 'not a dict')
        else:
            QtGui.QMessageBox.information(self, 'no file selected')

    def exit_list(self, parent):
        if len(parent.opslist) > 0:
            parent.batch = True
        self.accept()

### custom QDialog which allows user to fill in ops and run suite2p!
class RunWindow(QtGui.QDialog):
    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(0,0,1500,900)
        self.setWindowTitle('Choose run options (hold mouse over parameters to see descriptions)')
        self.win = QtGui.QWidget(self)
        self.layout = QtGui.QGridLayout()
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)
        # initial ops values
        self.opsfile = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                          'ops/ops_user.npy')
        try:
            self.ops = np.load(self.opsfile, allow_pickle=True).item()
            ops0 = run_s2p.default_ops()
            self.ops = {**ops0, **self.ops}
            print('loaded default ops')
        except Exception as e:
            print('could not load default ops, using built-in ops settings')
            self.ops = run_s2p.default_ops()
        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        self.opslist = []
        self.batch = False
        self.intkeys = ['nplanes', 'nchannels', 'functional_chan', 'align_by_chan', 'nimg_init',
                   'batch_size', 'max_iterations', 'nbinned','inner_neuropil_radius',
                   'min_neuropil_pixels', 'spatial_scale', 'do_registration']
        self.boolkeys = ['delete_bin', 'do_bidiphase', 'reg_tif', 'reg_tif_chan2',
                     'save_mat', 'combined', '1Preg', 'nonrigid', 'bruker',
                    'connected', 'roidetect', 'keep_movie_raw', 'allow_overlap', 'sparse_mode']
        tifkeys = ['nplanes','nchannels','functional_chan','tau','fs','delete_bin','do_bidiphase','bidiphase']
        outkeys = ['preclassify','save_mat','combined','reg_tif','reg_tif_chan2','aspect','bruker']
        regkeys = ['do_registration','align_by_chan','nimg_init','batch_size','smooth_sigma', 'smooth_sigma_time','maxregshift','th_badframes','keep_movie_raw','two_step_registration']
        nrkeys = [['nonrigid','block_size','snr_thresh','maxregshiftNR'], ['1Preg','spatial_hp','pre_smooth','spatial_taper']]
        cellkeys = ['roidetect','sparse_mode','diameter','spatial_scale','connected','threshold_scaling','max_overlap','max_iterations','high_pass']
        neudeconvkeys = [['allow_overlap','inner_neuropil_radius','min_neuropil_pixels'], ['win_baseline','sig_baseline','neucoeff']]
        keys = [tifkeys, outkeys, regkeys, nrkeys, cellkeys, neudeconvkeys]
        labels = ['Main settings','Output settings','Registration',['Nonrigid','1P'],'ROI detection',['Extraction/Neuropil','Deconvolution']]
        tooltips = ['each tiff has this many planes in sequence',
                    'each tiff has this many channels per plane',
                    'this channel is used to extract functional ROIs (1-based)',
                    'timescale of sensor in deconvolution (in seconds)',
                    'sampling rate (per plane)',
                    'if 1, binary file is deleted after processing is complete',
                    'whether or not to compute bidirectional phase offset of recording (from line scanning)',
                    'set a fixed number (in pixels) for the bidirectional phase offset',
                    'apply ROI classifier before signal extraction with probability threshold (set to 0 to turn off)',
                    'save output also as mat file "Fall.mat"',
                    'combine results across planes in separate folder "combined" at end of processing',
                    'if 1, registered tiffs are saved',
                    'if 1, registered tiffs of channel 2 (non-functional channel) are saved',
                    'um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)',
                    'if you have bruker single-page tiffs with Ch1 and Ch2, say 1',
                    "if 1, registration is performed if it wasn't performed already",
                    'when multi-channel, you can align by non-functional channel (1-based)',
                    '# of subsampled frames for finding reference image',
                    'number of frames per batch',
                    'gaussian smoothing after phase corr: 1.15 good for 2P recordings, recommend 2-5 for 1P recordings',
                    'gaussian smoothing in time, useful for low SNR data',
                    'max allowed registration shift, as a fraction of frame max(width and height)',
                    'this parameter determines which frames to exclude when determining cropped frame size - set it smaller to exclude more frames',
                    'if 1, unregistered binary is kept in a separate file data_raw.bin',
                    'run registration twice (useful if data is really noisy), *keep_movie_raw must be 1*',
                    'whether to use nonrigid registration (splits FOV into blocks of size block_size)',
                    'block size in number of pixels in Y and X (two numbers separated by a comma)',
                    'if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing',
                    'maximum *pixel* shift allowed for nonrigid, relative to rigid',
                    'whether to perform high-pass filtering and tapering for registration (necessary for 1P recordings)',
                    'window for spatial high-pass filtering before registration',
                    'whether to smooth before high-pass filtering before registration',
                    "how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*smooth_sigma)",
                    'whether or not to run cell (ROI) detection',
                    'whether to run sparse_mode cell extraction (scale-free) or original algorithm (default is original)',
                    'if sparse_mode=0, input average diameter of ROIs in recording (can give a list e.g. 6,9)',
                    'if sparse_mode=1, choose size of ROIs: 0 = multi-scale; 1 = 6 pixels, 2 = 12, 3 = 24, 4 = 48',
                    'whether or not to require ROIs to be fully connected (set to 0 for dendrites/boutons)',
                    'adjust the automatically determined threshold by this scalar multiplier',
                    'ROIs with greater than this overlap as a fraction of total pixels will be discarded',
                    'maximum number of iterations for ROI detection',
                    'running mean subtraction with window of size "high_pass" (use low values for 1P)',
                    'allow shared pixels to be used for fluorescence extraction from overlapping ROIs (otherwise excluded from both ROIs)',
                    'number of pixels between ROI and neuropil donut',
                    'minimum number of pixels in the neuropil',
                    'window for maximin',
                    'smoothing constant for gaussian filter',
                    'neuropil coefficient']

        bigfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        qlabel = QtGui.QLabel('File paths')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,0,0,1,1)
        #self.loadDb = QtGui.QPushButton('Load db.npy')
        #self.loadDb.clicked.connect(self.load_db)
        #self.layout.addWidget(self.loadDb,0,1,1,1)

        loadOps = QtGui.QPushButton('Load ops file')
        loadOps.clicked.connect(self.load_ops)
        saveDef = QtGui.QPushButton('Save ops as default')
        saveDef.clicked.connect(self.save_default_ops)
        saveOps = QtGui.QPushButton('Save ops to file')
        saveOps.clicked.connect(self.save_ops)
        self.layout.addWidget(loadOps,0,2,1,2)
        self.layout.addWidget(saveDef,1,2,1,2)
        self.layout.addWidget(saveOps,2,2,1,2)
        self.layout.addWidget(QtGui.QLabel(''),3,2,1,2)
        self.layout.addWidget(QtGui.QLabel('Load example ops'),4,2,1,2)
        for k in range(3):
            qw = QtGui.QPushButton('Save ops to file')
        saveOps.clicked.connect(self.save_ops)
        self.opsbtns = QtGui.QButtonGroup(self)
        opsstr = ['1P imaging', 'dendrites/axons']
        self.opsname = ['1P', 'dendrite']
        for b in range(len(opsstr)):
            btn = OpsButton(b, opsstr[b], self)
            self.opsbtns.addButton(btn, b)
            self.layout.addWidget(btn, 5+b,2,1,2)
        l=0
        self.keylist = []
        self.editlist = []
        kk=0
        wk=0
        for lkey in keys:
            k = 0
            kl=0
            if type(labels[l]) is list:
                labs = labels[l]
                keyl = lkey
            else:
                labs = [labels[l]]
                keyl = [lkey]
            for label in labs:
                qlabel = QtGui.QLabel(label)
                qlabel.setFont(bigfont)
                self.layout.addWidget(qlabel,k*2,2*(l+2),1,2)
                k+=1
                for key in keyl[kl]:
                    lops = 1
                    if self.ops[key] or (self.ops[key] == 0):
                        qedit = LineEdit(wk,key,self)
                        qlabel = QtGui.QLabel(key)
                        qlabel.setToolTip(tooltips[kk])
                        qedit.set_text(self.ops)
                        qedit.setToolTip(tooltips[kk])
                        qedit.setFixedWidth(90)
                        self.layout.addWidget(qlabel,k*2-1,2*(l+2),1,2)
                        self.layout.addWidget(qedit,k*2,2*(l+2),1,2)
                        self.keylist.append(key)
                        self.editlist.append(qedit)
                        wk+=1
                    k+=1
                    kk+=1
                kl+=1
            l+=1

        sbxstr = ['scanbox', 'min_row', 'max_row', 'min_col', 'max_col']
        sbxkeys = ['scanbox', 'min_row_sbx', 'max_row_sbx', 'min_col_sbx', 'max_col_sbx']
        sbxtooltips = ['load scanbox', 'minimum row value for FOV', 'maximum row value for FOV', \
                       'minimum col value for FOV', 'maximum col value for FOV']
        sb_pos = 8
        for sb in range(len(sbxstr)):
            qedit = LineEdit(sb,sbxkeys[sb],self)
            qlabel = QtGui.QLabel(sbxkeys[sb])
            qlabel.setToolTip(sbxtooltips[sb])
            qedit.set_text(self.ops)
            qedit.setToolTip(tooltips[sb])
            qedit.setFixedWidth(90)
            self.layout.addWidget(qlabel,sb_pos,2,1,2)
            self.layout.addWidget(qedit,sb_pos+1,2,1,2)
            self.keylist.append(key)
            self.editlist.append(qedit)
            sb_pos+=2

        # data_path
        key = 'look_one_level_down'
        qlabel = QtGui.QLabel(key)
        qlabel.setToolTip('whether to look in all subfolders when searching for tiffs')
        self.layout.addWidget(qlabel,1,0,1,1)
        qedit = LineEdit(wk,key,self)
        qedit.set_text(self.ops)
        qedit.setFixedWidth(95)
        self.layout.addWidget(qedit,2,0,1,1)
        self.keylist.append(key)
        self.editlist.append(qedit)
        self.btiff = QtGui.QPushButton('Add directory to data_path')
        self.btiff.clicked.connect(self.get_folders)
        self.layout.addWidget(self.btiff,3,0,1,2)
        qlabel = QtGui.QLabel('data_path')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,4,0,1,1)
        self.qdata = []
        for n in range(7):
            self.qdata.append(QtGui.QLabel(''))
            self.layout.addWidget(self.qdata[n],
                                  n+5,0,1,2)
        # save_path0
        self.bsbxpy = QtGui.QPushButton('OR add sbx file path')
        self.bsbxpy.clicked.connect(self.get_sbxpy)
        self.layout.addWidget(self.bsbxpy,9,0,1,2)
        self.sbxtext = QtGui.QLabel('')
        self.layout.addWidget(self.sbxtext,10,0,1,2)
        self.bh5py = QtGui.QPushButton('OR add h5 file path')
        self.bh5py.clicked.connect(self.get_h5py)
        self.layout.addWidget(self.bh5py,11,0,1,2)
        self.h5text = QtGui.QLabel('')
        self.layout.addWidget(self.h5text,12,0,1,2)
        self.bsave = QtGui.QPushButton('Add save_path (default is 1st data_path)')
        self.bsave.clicked.connect(self.save_folder)
        self.layout.addWidget(self.bsave,13,0,1,2)
        self.savelabel = QtGui.QLabel('')
        self.layout.addWidget(self.savelabel,14,0,1,2)
        # fast_disk
        self.bbin = QtGui.QPushButton('Add fast_disk (default is save_path)')
        self.bbin.clicked.connect(self.bin_folder)
        self.layout.addWidget(self.bbin,15,0,1,2)
        self.binlabel = QtGui.QLabel('')
        self.layout.addWidget(self.binlabel,16,0,1,2)
        self.runButton = QtGui.QPushButton('RUN SUITE2P')
        self.runButton.clicked.connect(lambda: self.run_S2P(parent))
        n0 = 21
        self.layout.addWidget(self.runButton,n0,0,1,1)
        self.runButton.setEnabled(False)
        self.textEdit = QtGui.QTextEdit()
        self.layout.addWidget(self.textEdit, n0+1,0,30,2*l)
        self.textEdit.setFixedHeight(300)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(lambda: self.finished(parent))
        # stop process
        self.stopButton = QtGui.QPushButton('STOP')
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, n0,1,1,1)
        self.stopButton.clicked.connect(self.stop)
        # cleanup button
        self.cleanButton = QtGui.QPushButton('Add a clean-up *.py')
        self.cleanButton.setToolTip('will run at end of processing')
        self.cleanButton.setEnabled(True)
        self.layout.addWidget(self.cleanButton, n0,2,1,2)
        self.cleanup = False
        self.cleanButton.clicked.connect(self.clean_script)
        self.cleanLabel = QtGui.QLabel('')
        self.layout.addWidget(self.cleanLabel,n0,4,1,12)
        self.listOps = QtGui.QPushButton('save settings and\n add more (batch)')
        self.listOps.clicked.connect(self.list_ops)
        self.layout.addWidget(self.listOps,n0,12,1,2)
        self.listOps.setEnabled(False)
        self.removeOps = QtGui.QPushButton('remove last added')
        self.removeOps.clicked.connect(self.remove_ops)
        self.layout.addWidget(self.removeOps,n0,14,1,2)
        self.removeOps.setEnabled(False)
        self.odata = []
        for n in range(10):
            self.odata.append(QtGui.QLabel(''))
            self.layout.addWidget(self.odata[n],
                                  n0+1+n,12,1,4)
        self.f = 0

    def remove_ops(self):
        L = len(self.opslist)
        if L == 1:
            self.batch = False
            self.opslist = []
            self.removeOps.setEnabled(False)
        else:
            del self.opslist[L-1]
        self.odata[L-1].setText('')

    def list_ops(self):
        self.batch = True
        self.compile_ops_db()
        L = len(self.opslist)
        self.odata[L].setText(self.datastr)
        np.save('ops%d.npy'%L, self.ops)
        np.save('db%d.npy'%L, self.db)
        self.opslist.append('ops%d.npy'%L)
        # clear file fields
        self.db = {}
        self.data_path = []
        if hasattr(self, 'h5_path'):
            self.h5_path = []
            self.h5_key = 'data'
        if hasattr(self, 'sbx_path'): #scanbox
            self.sbx_path = []
        self.save_path = []
        self.fast_disk = []
        # clear labels
        for n in range(7):
            self.qdata[n].setText('')
        self.savelabel.setText('')
        self.binlabel.setText('')
        self.h5text.setText('')
        # clear ops not in GUI
        self.ops = np.load(self.opsfile, allow_pickle=True).item()
        self.save_text() # grab ops in GUI
        # enable all the file loaders again
        self.bh5py.setEnabled(True)
        self.btiff.setEnabled(True)
        self.bsave.setEnabled(True)
        self.bbin.setEnabled(True)
        self.bsbxpy.setEnabled(True)
        # and enable the run button
        self.runButton.setEnabled(True)
        self.removeOps.setEnabled(True)
        self.listOps.setEnabled(False)

    def compile_ops_db(self):
        for k,key in enumerate(self.keylist):
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys)
        self.db = {}
        self.db['data_path'] = self.data_path
        self.db['subfolders'] = []
        if hasattr(self, 'h5_path') and len(self.h5_path) > 0:
            self.db['h5py'] = self.h5_path
            self.db['h5py_key'] = self.h5_key
            self.datastr = self.h5_path
        elif hasattr(self, 'sbx_path'):
            self.db['sbxpy'] = self.sbx_path
            self.datastr = self.sbx_path
        else:
            self.datastr = self.data_path[0]
        #print(self.datastr)
        if len(self.save_path)==0:
            if len(self.db['data_path'])>0:
                fpath = self.db['data_path'][0]
            elif hasattr(self, 'h5_path'):
                fpath = os.path.dirname(self.db['h5py'])
            else:
                fpath = os.path.dirname(self.db['sbxpy'])
            self.save_path = fpath
        self.db['save_path0'] = self.save_path
        if len(self.fast_disk)==0:
            self.fast_disk = self.save_path
        self.db['fast_disk'] = self.fast_disk

    def run_S2P(self, parent):
        self.finish = True
        self.error = False
        if self.batch:
            shutil.copy('ops%d.npy'%self.f, 'ops.npy')
            shutil.copy('db%d.npy'%self.f, 'db.npy')
            self.db = np.load('db.npy', allow_pickle=True).item()
        else:
            self.compile_ops_db()
            np.save('ops.npy', self.ops)
            np.save('db.npy', self.db)
        print('Running suite2p!')
        print('starting process')
        print(self.db)
        if len(self.save_path)==0:
            if len(self.db['data_path'])>0:
                fpath = self.db['data_path'][0]
            else:
                fpath = os.path.dirname(self.db['h5py'])
            self.save_path = fpath
        save_folder = os.path.join(self.save_path, 'suite2p/')
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self.logfile = open(os.path.join(save_folder, 'run.log'), 'w')
        #self.logfile.close()
        self.process.start('python -u -W ignore -m suite2p --ops ops.npy --db db.npy')

    def stop(self):
        self.finish = False
        self.process.kill()

    def started(self):
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.cleanButton.setEnabled(False)

    def finished(self, parent):
        self.logfile.close()
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        if self.finish and not self.error:
            self.cleanButton.setEnabled(True)
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Opening in GUI (can close this window)\n')
            parent.fname = os.path.join(self.db['save_path0'], 'suite2p', 'plane0','stat.npy')
            parent.load_proc()
        elif not self.error:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Interrupted by user (not finished)\n')
        else:
            cursor = self.textEdit.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText('Interrupted by error (not finished)\n')
        if self.batch:
            self.f += 1
            if self.f < len(self.opslist):
                self.run_S2P(parent)

    def save_ops(self):
        name = QtGui.QFileDialog.getSaveFileName(self,'Ops name (*.npy)')
        name = name[0]
        self.save_text()
        if name:
            np.save(name, self.ops)
            print('saved current settings to %s'%(name))

    def save_default_ops(self):
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = run_s2p.default_ops()
        self.save_text()
        np.save(name, self.ops)
        self.ops = ops
        print('saved current settings in GUI as default ops')

    def save_text(self):
        for k in range(len(self.editlist)):
            key = self.keylist[k]
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys)

    def load_ops(self):
        print('loading ops')
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open ops file (npy or json)')
        name = name[0]
        if len(name)>0:
            ext = os.path.splitext(name)[1]
            try:
                if ext == '.npy':
                    ops = np.load(name, allow_pickle=True).item()
                elif ext == '.json':
                    with open(name, 'r') as f:
                        ops = json.load(f)
                ops0 = run_s2p.default_ops()
                ops = {**ops0, **ops}
                for key in ops:
                    if key!='data_path' and key!='save_path' and key!='fast_disk' and key!='cleanup' and key!='save_path0' and key!='h5py':
                        if key in self.keylist:
                            self.editlist[self.keylist.index(key)].set_text(ops)
                        self.ops[key] = ops[key]
                if 'data_path' in ops and len(ops['data_path'])>0:
                    self.data_path = ops['data_path']
                    for n in range(7):
                        if n<len(self.data_path):
                            self.qdata[n].setText(self.data_path[n])
                        else:
                            self.qdata[n].setText('')
                    self.runButton.setEnabled(True)
                    self.bh5py.setEnabled(False)
                    self.btiff.setEnabled(True)
                    self.listOps.setEnabled(True)
                    self.bsbxpy.setEnabled(False)
                    if hasattr(self,'h5_path'):
                        self.h5text.setText('')
                        del self.h5_path
                elif 'h5py' in ops and len(ops['h5py'])>0:
                    self.h5_path = ops['h5py']
                    self.h5_key = ops['h5py_key']
                    self.h5text.setText(ops['h5py'])
                    self.data_path = []
                    for n in range(7):
                        self.qdata[n].setText('')
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(False)
                    self.bh5py.setEnabled(True)
                    self.listOps.setEnabled(True)
                    self.bsbxpy.setEnabled(False)
                elif 'scanbox' in ops>0:
                    self.sbx_path = ops['sbxpy']
                    self.sbxtext.setText(ops['sbxpy'])
                    self.data_path = []
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(False)
                    self.bh5py.setEnabled(True)
                    self.listOps.setEnabled(True)   
                    self.bsbxpy.setEnabled(True)
                if 'save_path0' in ops and len(ops['save_path0'])>0:
                    self.save_path = ops['save_path0']
                    self.savelabel.setText(self.save_path)
                if 'fast_disk' in ops and len(ops['fast_disk'])>0:
                    self.fast_disk = ops['fast_disk']
                    self.binlabel.setText(self.fast_disk)
                if 'clean_script' in ops and len(ops['clean_script'])>0:
                    self.ops['clean_script'] = ops['clean_script']
                    self.cleanLabel.setText(ops['clean_script'])

            except Exception as e:
                print('could not load ops file')
                print(e)

    def load_db(self):
        print('loading db')

    def stdout_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        output = str(self.process.readAllStandardOutput(), 'utf-8')
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        #self.logfile = open(os.path.join(self.save_path, 'suite2p/run.log'), 'a')
        self.logfile.write(output)
        #self.logfile.close()

    def stderr_write(self):
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText('>>>ERROR<<<\n')
        output = str(self.process.readAllStandardError(), 'utf-8')
        cursor.insertText(output)
        self.textEdit.ensureCursorVisible()
        self.error = True
        #self.logfile = open(os.path.join(self.save_path, 'suite2p/run.log'), 'a')
        self.logfile.write('>>>ERROR<<<\n')
        self.logfile.write(output)

    def clean_script(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open clean up file',filter='*.py')
        name = name[0]
        if name:
            self.cleanup = True
            self.cleanScript = name
            self.cleanLabel.setText(name)
            self.ops['clean_script'] = name

    def get_folders(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Add directory to data path")
        if len(name)>0:
            self.data_path.append(name)
            self.qdata[len(self.data_path)-1].setText(name)
            self.runButton.setEnabled(True)
            self.listOps.setEnabled(True)
            #self.loadDb.setEnabled(False)
            self.bh5py.setEnabled(False)
            self.bsbxpy.setEnabled(False)

    def get_h5py(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open h5 file')
        name = name[0]
        if len(name)>0:
            self.h5_path = name
            self.h5text.setText(name)
            TC = TextChooser(self)
            result = TC.exec_()
            if result:
                self.h5_key = TC.h5_key
            else:
                self.h5_key = 'data'
            self.runButton.setEnabled(True)
            self.listOps.setEnabled(True)
            #self.loadDb.setEnabled(False)
            self.btiff.setEnabled(False)
            self.bsbxpy.setEnabled(False)

    def get_sbxpy(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open sbx file')
        name = name[0]
        if len(name)>0:
            self.sbx_path = name
            self.sbxtext.setText(name)
            self.runButton.setEnabled(True)
            self.listOps.setEnabled(True)
            self.btiff.setEnabled(False)
            self.bh5py.setEnabled(False)

    def save_folder(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Save folder for data")
        if len(name)>0:
            self.save_path = name
            self.savelabel.setText(name)

    def bin_folder(self):
        name = QtGui.QFileDialog.getExistingDirectory(self, "Folder for binary file")
        self.fast_disk = name
        self.binlabel.setText(name)

class LineEdit(QtGui.QLineEdit):
    def __init__(self,k,key,parent=None):
        super(LineEdit,self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self,intkeys,boolkeys):
        key = self.key
        if key=='diameter' or key=='block_size':
            diams = self.text().replace(' ','').split(',')
            if len(diams)>1:
                okey = [int(diams[0]), int(diams[1])]
            else:
                okey = int(diams[0])
        else:
            if key in intkeys:
                okey = int(float(self.text()))
            elif key in boolkeys:
                okey = bool(int(self.text()))
            else:
                okey = float(self.text())
        return okey

    def set_text(self,ops):
        key = self.key
        if key=='diameter' or key=='block_size':
            if (type(ops[key]) is not int) and (len(ops[key])>1):
                dstr = str(int(ops[key][0])) + ', ' + str(int(ops[key][1]))
            else:
                dstr = str(int(ops[key]))
        else:
            if type(ops[key]) is not bool:
                dstr = str(ops[key])
            else:
                dstr = str(int(ops[key]))
        self.setText(dstr)

class OpsButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(OpsButton,self).__init__(parent)
        self.setText(Text)
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        try:
            opsdef = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              'ops/ops_%s.npy'%parent.opsname[bid])
            ops = np.load(opsdef, allow_pickle=True).item()
            for key in ops:
                if key in parent.keylist:
                    parent.editlist[parent.keylist.index(key)].set_text(ops)
                    parent.ops[key] = ops[key]
        except Exception as e:
            print('could not load ops file')
            print(e)


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

### custom QPushButton class that plots image when clicked
# requires buttons to put into a QButtonGroup (parent.viewbtns)
# allows only 1 button to pressed at a time
class ViewButton(QtGui.QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(ViewButton,self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in range(len(parent.views)):
            if parent.viewbtns.button(b).isEnabled():
                parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
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
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        for b in range(len(parent.colors)+2):
            if parent.colorbtns.button(b).isEnabled():
                parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot[2] = bid
        if not parent.sizebtns.button(1).isChecked():
            if bid==0:
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
        if bid==7:
            fig.corr_masks(parent)
        M = fig.draw_masks(parent)
        fig.plot_masks(parent,M)
        fig.plot_colorbar(parent,bid)
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
            if parent.ops_plot[2]!=0:
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
            if parent.ops_plot[2]==0:
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
