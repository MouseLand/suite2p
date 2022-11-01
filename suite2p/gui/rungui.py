import glob, json, os, shutil, pathlib
from datetime import datetime

import numpy as np

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QDialog, QLineEdit, QLabel, QPushButton, QWidget, QGridLayout, QButtonGroup, QComboBox, QTextEdit, QFileDialog

from cellpose.models import get_user_models, model_path, MODEL_NAMES

from . import io
from .. import default_ops


### ---- this file contains helper functions for GUI and the RUN window ---- ###

# type in h5py key
class TextChooser(QDialog):
    def __init__(self,parent=None):
        super(TextChooser, self).__init__(parent)
        self.setGeometry(300,300,180,100)
        self.setWindowTitle('h5 key')
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        self.qedit = QLineEdit('data')
        layout.addWidget(QLabel('h5 key for data field'),0,0,1,3)
        layout.addWidget(self.qedit,1,0,1,2)
        done = QPushButton('OK')
        done.clicked.connect(self.exit_list)
        layout.addWidget(done,2,1,1,1)

    def exit_list(self):
        self.h5_key = self.qedit.text()
        self.accept()

### custom QDialog which allows user to fill in ops and run suite2p!
class RunWindow(QDialog):
    def __init__(self, parent=None):
        super(RunWindow, self).__init__(parent)
        self.setGeometry(10,10,1500,900)
        self.setWindowTitle('Choose run options (hold mouse over parameters to see descriptions)')
        self.parent = parent
        self.win = QWidget(self)
        self.layout = QGridLayout()
        self.layout.setVerticalSpacing(2)
        self.layout.setHorizontalSpacing(25)
        self.win.setLayout(self.layout)
        # initial ops values
        self.opsfile = parent.opsuser
        self.ops_path = os.fspath(pathlib.Path.home().joinpath('.suite2p').joinpath('ops').absolute())
        try:
            self.reset_ops()
            print('loaded default ops')
        except Exception as e:
            print('ERROR: %s'%e)
            print('could not load default ops, using built-in ops settings')
            self.ops = default_ops()

        # remove any remaining ops files
        fs = glob.glob('ops*.npy')
        for f in fs:
            os.remove(f)
        fs = glob.glob('db*.npy')
        for f in fs:
            os.remove(f)

        self.data_path = []
        self.save_path = []
        self.fast_disk = []
        self.opslist = []
        self.batch = False
        self.f = 0
        self.create_buttons()

    def reset_ops(self):
        self.ops = np.load(self.opsfile, allow_pickle=True).item()
        ops0 = default_ops()
        self.ops = {**ops0, **self.ops}
        if hasattr(self, 'editlist'):
            for k in range(len(self.editlist)):
                self.editlist[k].set_text(self.ops)

    def create_buttons(self):
        self.intkeys = ['nplanes', 'nchannels', 'functional_chan', 'align_by_chan', 'nimg_init',
                   'batch_size', 'max_iterations', 'nbinned','inner_neuropil_radius',
                   'min_neuropil_pixels', 'spatial_scale', 'do_registration', 'anatomical_only']
        self.boolkeys = ['delete_bin', 'move_bin','do_bidiphase', 'reg_tif', 'reg_tif_chan2',
                     'save_mat', 'save_NWB' 'combined', '1Preg', 'nonrigid', 
                    'connected', 'roidetect', 'neuropil_extract', 
                    'spikedetect', 'keep_movie_raw', 'allow_overlap', 'sparse_mode']
        self.stringkeys = ['pretrained_model']
        tifkeys = ['nplanes','nchannels','functional_chan','tau','fs','do_bidiphase','bidiphase', 'multiplane_parallel', 'ignore_flyback']
        outkeys = ['preclassify','save_mat','save_NWB','combined','reg_tif','reg_tif_chan2','aspect','delete_bin','move_bin']
        regkeys = ['do_registration','align_by_chan','nimg_init','batch_size','smooth_sigma', 'smooth_sigma_time','maxregshift','th_badframes','keep_movie_raw','two_step_registration']
        nrkeys = [['nonrigid','block_size','snr_thresh','maxregshiftNR'], ['1Preg','spatial_hp_reg','pre_smooth','spatial_taper']]
        cellkeys = ['roidetect', 'denoise', 'spatial_scale', 'threshold_scaling', 'max_overlap','max_iterations','high_pass','spatial_hp_detect']
        anatkeys = ['anatomical_only', 'diameter', 'cellprob_threshold', 'flow_threshold', 'pretrained_model', 'spatial_hp_cp']
        neudeconvkeys = [['neuropil_extract', 'allow_overlap','inner_neuropil_radius','min_neuropil_pixels'], ['soma_crop','spikedetect','win_baseline','sig_baseline','neucoeff']]
        keys = [tifkeys, outkeys, regkeys, nrkeys, cellkeys, anatkeys, neudeconvkeys]
        labels = ['Main settings','Output settings','Registration',['Nonrigid','1P'],'Functional detect', 'Anat detect', ['Extraction/Neuropil','Classify/Deconv']]
        tooltips = ['each tiff has this many planes in sequence',
                    'each tiff has this many channels per plane',
                    'this channel is used to extract functional ROIs (1-based)',
                    'timescale of sensor in deconvolution (in seconds)',
                    'sampling rate (per plane)',
                    'whether or not to compute bidirectional phase offset of recording (from line scanning)',
                    'set a fixed number (in pixels) for the bidirectional phase offset',
                    'process each plane with a separate job on a computing cluster',
                    'ignore flyback planes 0-indexed separated by a comma e.g. "0,10"; "-1" means no planes ignored so all planes processed',
                    'apply ROI classifier before signal extraction with probability threshold (set to 0 to turn off)',
                    'save output also as mat file "Fall.mat"',
                    'save output also as NWB file "ophys.nwb"',
                    'combine results across planes in separate folder "combined" at end of processing',
                    'if 1, registered tiffs are saved',
                    'if 1, registered tiffs of channel 2 (non-functional channel) are saved',
                    'um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)',
                    'if 1, binary file is deleted after processing is complete',
                    'if 1, and fast_disk is different than save_disk, binary file is moved to save_disk',
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
                    'if 1, run cell (ROI) detection (either functional or anatomical if anatomical_only > 0)',
                    'if 1, run PCA denoising on binned movie to improve cell detection',
                    'choose size of ROIs: 0 = multi-scale; 1 = 6 pixels, 2 = 12, 3 = 24, 4 = 48',
                    'adjust the automatically determined threshold for finding ROIs by this scalar multiplier',
                    'ROIs with greater than this overlap as a fraction of total pixels will be discarded',
                    'maximum number of iterations for ROI detection',
                    'temporal running mean subtraction with window of size "high_pass" (use low values for 1P)',
                    'spatial high-pass filter size (used to remove spatially-correlated neuropil)',
                    'run cellpose to get masks on 1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj',
                    'input average diameter of ROIs in recording (can give a list e.g. 6,9 if aspect not equal), if set to 0 auto-determination run by Cellpose',
                    'cellprob_threshold for cellpose',
                    'flow_threshold for cellpose (throws out masks, if getting too few masks, set to 0)',
                    'model type string from Cellpose (can be a built-in model or a user model that is added to the Cellpose GUI)',
                    'high-pass image spatially by a multiple of the diameter (if field is non-uniform, a value of ~2 is recommended',
                    'whether or not to extract neuropil; if 0, Fneu is set to 0',
                    'allow shared pixels to be used for fluorescence extraction from overlapping ROIs (otherwise excluded from both ROIs)',
                    'number of pixels between ROI and neuropil donut',
                    'minimum number of pixels in the neuropil',
                    'if 1, crop dendrites for cell classification stats like compactness',
                    'if 1, run spike detection (deconvolution)',
                    'window for maximin',
                    'smoothing constant for gaussian filter',
                    'neuropil coefficient',
                    ]

        bigfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)
        qlabel = QLabel('File paths')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,0,0,1,1)
        loadOps = QPushButton('Load ops file')
        loadOps.clicked.connect(self.load_ops)
        saveDef = QPushButton('Save ops as default')
        saveDef.clicked.connect(self.save_default_ops)
        revertDef = QPushButton('Revert default ops to built-in')
        revertDef.clicked.connect(self.revert_default_ops)
        saveOps = QPushButton('Save ops to file')
        saveOps.clicked.connect(self.save_ops)
        self.layout.addWidget(loadOps,0,4,1,2)
        self.layout.addWidget(saveDef,1,4,1,2)
        self.layout.addWidget(revertDef,2,4,1,2)
        self.layout.addWidget(saveOps,3,4,1,2)
        self.layout.addWidget(QLabel(''),4,4,1,2)
        self.layout.addWidget(QLabel('Load example ops'),5,4,1,2)
        for k in range(3):
            qw = QPushButton('Save ops to file')
        #saveOps.clicked.connect(self.save_ops)
        self.opsbtns = QButtonGroup(self)
        opsstr = ['1P imaging', 'dendrites/axons']
        self.opsname = ['1P', 'dendrite']
        for b in range(len(opsstr)):
            btn = OpsButton(b, opsstr[b], self)
            self.opsbtns.addButton(btn, b)
            self.layout.addWidget(btn, 6+b,4,1,2)
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
                qlabel = QLabel(label)
                qlabel.setFont(bigfont)
                self.layout.addWidget(qlabel,k*2,2*(l+4),1,2)
                k+=1
                for key in keyl[kl]:
                    lops = 1
                    if self.ops[key] or (self.ops[key] == 0) or len(self.ops[key])==0:
                        qedit = LineEdit(wk,key,self)
                        qlabel = QLabel(key)
                        qlabel.setToolTip(tooltips[kk])
                        qedit.set_text(self.ops)
                        qedit.setToolTip(tooltips[kk])
                        qedit.setFixedWidth(90)
                        self.layout.addWidget(qlabel,k*2-1,2*(l+4),1,2)
                        self.layout.addWidget(qedit,k*2,2*(l+4),1,2)
                        self.keylist.append(key)
                        self.editlist.append(qedit)
                        wk+=1
                    k+=1
                    kk+=1
                kl+=1
            l+=1

        # data_path
        key = 'input_format'
        qlabel = QLabel(key)
        qlabel.setFont(bigfont)
        qlabel.setToolTip('File format (selects which parser to use)')
        self.layout.addWidget(qlabel,1,0,1,1)
        self.inputformat = QComboBox()
        [self.inputformat.addItem(f) for f in ['tif','bruker','sbx', 'h5','mesoscan','haus']]
        self.inputformat.currentTextChanged.connect(self.parse_inputformat)
        self.layout.addWidget(self.inputformat,2,0,1,1)

        key = 'look_one_level_down'
        qlabel = QLabel(key)
        qlabel.setToolTip('whether to look in all subfolders when searching for files')
        self.layout.addWidget(qlabel,3,0,1,1)
        qedit = LineEdit(wk,key,self)
        qedit.set_text(self.ops)
        qedit.setFixedWidth(95)
        self.layout.addWidget(qedit,4,0,1,1)
        self.keylist.append(key)
        self.editlist.append(qedit)

        cw=4
        self.btiff = QPushButton('Add directory to data_path')
        self.btiff.clicked.connect(self.get_folders)
        self.layout.addWidget(self.btiff,5,0,1,cw)
        qlabel = QLabel('data_path')
        qlabel.setFont(bigfont)
        self.layout.addWidget(qlabel,6,0,1,1)
        self.qdata = []
        for n in range(9):
            self.qdata.append(QLabel(''))
            self.layout.addWidget(self.qdata[n],
                                  n+7,0,1,cw)

        self.bsave = QPushButton('Add save_path (default is 1st data_path)')
        self.bsave.clicked.connect(self.save_folder)
        self.layout.addWidget(self.bsave,16,0,1,cw)
        self.savelabel = QLabel('')
        self.layout.addWidget(self.savelabel,17,0,1,cw)
        # fast_disk
        self.bbin = QPushButton('Add fast_disk (default is save_path)')
        self.bbin.clicked.connect(self.bin_folder)
        self.layout.addWidget(self.bbin,18,0,1,cw)
        self.binlabel = QLabel('')
        self.layout.addWidget(self.binlabel,19,0,1,cw)
        self.runButton = QPushButton('RUN SUITE2P')
        self.runButton.clicked.connect(self.run_S2P)
        n0 = 22
        self.layout.addWidget(self.runButton,n0,0,1,1)
        self.runButton.setEnabled(False)
        self.textEdit = QTextEdit()
        self.layout.addWidget(self.textEdit, n0+1,0,30,2*l)
        self.textEdit.setFixedHeight(300)
        self.process = QtCore.QProcess(self)
        self.process.readyReadStandardOutput.connect(self.stdout_write)
        self.process.readyReadStandardError.connect(self.stderr_write)
        # disable the button when running the s2p process
        self.process.started.connect(self.started)
        self.process.finished.connect(self.finished)
        # stop process
        self.stopButton = QPushButton('STOP')
        self.stopButton.setEnabled(False)
        self.layout.addWidget(self.stopButton, n0,1,1,1)
        self.stopButton.clicked.connect(self.stop)
        # cleanup button
        self.cleanButton = QPushButton('Add a clean-up *.py')
        self.cleanButton.setToolTip('will run at end of processing')
        self.cleanButton.setEnabled(True)
        self.layout.addWidget(self.cleanButton, n0,2,1,2)
        self.cleanup = False
        self.cleanButton.clicked.connect(self.clean_script)
        self.cleanLabel = QLabel('')
        self.layout.addWidget(self.cleanLabel,n0,4,1,12)
        #n0+=1
        self.listOps = QPushButton('save settings and\n add more (batch)')
        self.listOps.clicked.connect(self.add_batch)
        self.layout.addWidget(self.listOps,n0,12,1,2)
        self.listOps.setEnabled(False)
        self.removeOps = QPushButton('remove last added')
        self.removeOps.clicked.connect(self.remove_ops)
        self.layout.addWidget(self.removeOps,n0,14,1,2)
        self.removeOps.setEnabled(False)
        self.odata = []
        self.n_batch = 15
        for n in range(self.n_batch):
            self.odata.append(QLabel(''))
            self.layout.addWidget(self.odata[n],
                                  n0+1+n,12,1,4)

    def remove_ops(self):
        L = len(self.opslist)
        if L == 1:
            self.batch = False
            self.opslist = []
            self.removeOps.setEnabled(False)
        else:
            del self.opslist[L-1]
        self.odata[L-1].setText('')
        self.odata[L-1].setToolTip('')
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
            self.qdata[n].setText('')
        self.savelabel.setText('')
        self.binlabel.setText('')

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
        np.save(os.path.join(self.ops_path, 'ops%d.npy'%L), self.ops)
        np.save(os.path.join(self.ops_path, 'db%d.npy'%L), self.db)
        self.opslist.append('ops%d.npy'%L)
        if hasattr(self, 'h5_key') and len(self.h5_key) > 0:
            self.db['h5py_key'] = self.h5_key

    def compile_ops_db(self):
        for k,key in enumerate(self.keylist):
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys, self.stringkeys)
        self.db = {}
        self.db['data_path'] = self.data_path
        self.db['subfolders'] = []
        self.datastr = self.data_path[0]

        # add data type specific keys
        if hasattr(self, 'h5_key') and len(self.h5_key) > 0:
            self.db['h5py_key'] = self.h5_key
        elif self.inputformat.currentText() == 'sbx':
            self.db['sbx_ndeadcols'] = -1

        # add save_path0 and fast_disk
        if len(self.save_path)==0:
            self.save_path = self.db['data_path'][0]
        self.db['save_path0'] = self.save_path
        if len(self.fast_disk)==0:
            self.fast_disk = self.save_path
        self.db['fast_disk'] = self.fast_disk
        self.db['input_format'] = self.inputformat.currentText()

    def run_S2P(self):
        if len(self.opslist)==0:
            self.add_ops()
        # pre-download model
        pretrained_model_string = self.ops.get('pretrained_model', 'cyto')
        pretrained_model_string = pretrained_model_string if pretrained_model_string is not None else 'cyto'
        pretrained_model_path = model_path(pretrained_model_string, 0, True)
        self.finish = True
        self.error = False
        ops_file = os.path.join(self.ops_path, 'ops.npy')
        db_file = os.path.join(self.ops_path, 'db.npy')
        shutil.copy(os.path.join(self.ops_path, 'ops%d.npy'%self.f), ops_file)
        shutil.copy(os.path.join(self.ops_path, 'db%d.npy'%self.f), db_file)
        self.db = np.load(db_file, allow_pickle=True).item()
        print('Running suite2p!')
        print('starting process')
        print(self.db)
        self.process.start('python -u -W ignore -m suite2p --ops "%s" --db "%s"'%(ops_file, db_file))

    def stop(self):
        self.finish = False
        self.logfile.close()
        self.process.kill()

    def started(self):
        self.runButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        self.cleanButton.setEnabled(False)
        save_folder = os.path.join(self.db['save_path0'], 'suite2p/')
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        self.logfile = open(os.path.join(save_folder, 'run.log'), 'a')
        dstring = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.logfile.write('\n >>>>> started run at %s'%dstring)

    def finished(self):
        self.logfile.close()
        self.runButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        cursor = self.textEdit.textCursor()
        cursor.movePosition(cursor.End)    
        if self.finish and not self.error:
            self.cleanButton.setEnabled(True)
            if len(self.opslist)==1:
                self.parent.fname = os.path.join(self.db['save_path0'], 'suite2p', 'plane0','stat.npy')
                if os.path.exists(self.parent.fname):
                    cursor.insertText('Opening in GUI (can close this window)\n')
                    io.load_proc(self.parent)
                else:
                    cursor.insertText('not opening plane in GUI (no ROIs)\n')
            else:
                cursor.insertText('BATCH MODE: %d more recordings remaining \n'%(len(self.opslist)-self.f-1))
                self.f += 1
                if self.f < len(self.opslist):
                    self.run_S2P()
        elif not self.error:
            cursor.insertText('Interrupted by user (not finished)\n')
        else:
            cursor.insertText('Interrupted by error (not finished)\n')

        # remove current ops from processing list        
        if len(self.opslist)==1:
            del self.opslist[0]
    
    def save_ops(self):
        name = QFileDialog.getSaveFileName(self,'Ops name (*.npy)')
        name = name[0]
        self.save_text()
        if name:
            np.save(name, self.ops)
            print('saved current settings to %s'%(name))

    def save_default_ops(self):
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = default_ops()
        self.save_text()
        np.save(name, self.ops)
        self.ops = ops
        print('saved current settings in GUI as default ops')

    def revert_default_ops(self):
        name = self.opsfile
        ops = self.ops.copy()
        self.ops = default_ops()
        np.save(name, self.ops)
        self.load_ops(name)
        print('reverted default ops to built-in ops')

    def save_text(self):
        for k in range(len(self.editlist)):
            key = self.keylist[k]
            self.ops[key] = self.editlist[k].get_text(self.intkeys, self.boolkeys, key)

    def load_ops(self, name=None):
        print('loading ops')
        if not (isinstance(name, str) and len(name)>0):
            name = QFileDialog.getOpenFileName(self, 'Open ops file (npy or json)')
            name = name[0]
        
        if len(name) > 0:
            ext = os.path.splitext(name)[1]
            try:
                if ext == '.npy':
                    ops = np.load(name, allow_pickle=True).item()
                elif ext == '.json':
                    with open(name, 'r') as f:
                        ops = json.load(f)
                ops0 = default_ops()
                ops = {**ops0, **ops}
                for key in ops:
                    if key!='data_path' and key!='save_path' and key!='fast_disk' and key!='cleanup' and key!='save_path0' and key!='h5py':
                        if key in self.keylist:
                            self.editlist[self.keylist.index(key)].set_text(ops)
                        self.ops[key] = ops[key]
                if not 'input_format' in self.ops.keys():
                    self.ops['input_format'] = 'tif'
                if 'data_path' in ops and len(ops['data_path'])>0:
                    self.data_path = ops['data_path']
                    for n in range(9):
                        if n<len(self.data_path):
                            self.qdata[n].setText(self.data_path[n])
                        else:
                            self.qdata[n].setText('')
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(True)
                    self.listOps.setEnabled(True)
                if 'h5py_key' in ops and len(ops['h5py_key'])>0:
                    self.h5_key = ops['h5py_key']
                self.inputformat.currentTextChanged.connect(lambda x:x)
                self.inputformat.setCurrentText(self.ops['input_format'])
                self.inputformat.currentTextChanged.connect(self.parse_inputformat)
                if self.ops['input_format'] == 'sbx':
                    self.runButton.setEnabled(True)
                    self.btiff.setEnabled(False)
                    self.listOps.setEnabled(True)

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
        name = QFileDialog.getOpenFileName(self, 'Open clean up file',filter='*.py')
        name = name[0]
        if name:
            self.cleanup = True
            self.cleanScript = name
            self.cleanLabel.setText(name)
            self.ops['clean_script'] = name

    def get_folders(self):
        name = QFileDialog.getExistingDirectory(self, "Add directory to data path")
        if len(name)>0:
            self.data_path.append(name)
            self.qdata[len(self.data_path)-1].setText(name)
            self.qdata[len(self.data_path)-1].setToolTip(name)
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
            self.h5_key = 'data'

    def parse_inputformat(self):
        inputformat = self.inputformat.currentText()
        print('Input format: ' + inputformat)
        if inputformat == 'h5':
            # replace functionality of "old" button
            self.get_h5py()
        else:
            pass


    def save_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Save folder for data")
        if len(name)>0:
            self.save_path = name
            self.savelabel.setText(name)
            self.savelabel.setToolTip(name)


    def bin_folder(self):
        name = QFileDialog.getExistingDirectory(self, "Folder for binary file")
        self.fast_disk = name
        self.binlabel.setText(name)
        self.binlabel.setToolTip(name)


class LineEdit(QLineEdit):
    def __init__(self,k,key,parent=None):
        super(LineEdit,self).__init__(parent)
        self.key = key
        #self.textEdited.connect(lambda: self.edit_changed(parent.ops, k))

    def get_text(self,intkeys,boolkeys,stringkeys):
        key = self.key
        if key=='diameter' or key=='block_size':
            diams = self.text().replace(' ','').split(',')
            if len(diams)>1:
                okey = [int(diams[0]), int(diams[1])]
            else:
                okey = int(diams[0])
        elif key=='ignore_flyback':
            okey = self.text().replace(' ','').split(',')
            for i in range(len(okey)):
                okey[i] = int(okey[i])
            if len(okey)==1 and okey[0]==-1:
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

    def set_text(self,ops):
        key = self.key
        if key=='diameter' or key=='block_size':
            if (type(ops[key]) is not int) and (len(ops[key])>1):
                dstr = str(int(ops[key][0])) + ', ' + str(int(ops[key][1]))
            else:
                dstr = str(int(ops[key]))
        elif key=='ignore_flyback':
            if not isinstance(ops[key], (list, np.ndarray)):
                ops[key] = [ops[key]]
            if len(ops[key])==0:
                dstr = '-1'
            else:
                dstr = ''
                for i in ops[key]:
                    dstr += str(int(i))
                    if i<len(ops[key])-1:
                        dstr+=', '
        else:
            if type(ops[key]) is not bool:
                dstr = str(ops[key])
            else:
                dstr = str(int(ops[key]))
        self.setText(dstr)

class OpsButton(QPushButton):
    def __init__(self, bid, Text, parent=None):
        super(OpsButton,self).__init__(parent)
        self.setText(Text)
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()
    def press(self, parent, bid):
        try:
            opsdef = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              '../ops/ops_%s.npy'%parent.opsname[bid])
            ops = np.load(opsdef, allow_pickle=True).item()
            for key in ops:
                if key in parent.keylist:
                    parent.editlist[parent.keylist.index(key)].set_text(ops)
                    parent.ops[key] = ops[key]
        except Exception as e:
            print('could not load ops file')
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
