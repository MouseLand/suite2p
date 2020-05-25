import os

import numpy as np
import scipy.io
from PyQt5 import QtGui

from . import masks, traces, reggui
from .. import io


def export_fig(parent):
    parent.win.scene().contextMenuItem = parent.p1
    parent.win.scene().showExportDialog()

def enable_views_and_classifier(parent):
    for b in range(9):
        parent.quadbtns.button(b).setEnabled(True)
        parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
    for b in range(len(parent.view_names)):
        parent.viewbtns.button(b).setEnabled(True)
        parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        # parent.viewbtns.button(b).setShortcut(QtGui.QKeySequence('R'))
        if b == 0:
            parent.viewbtns.button(b).setChecked(True)
            parent.viewbtns.button(b).setStyleSheet(parent.stylePressed)
    # check for second channel
    if "meanImg_chan2_corrected" not in parent.ops:
        parent.viewbtns.button(5).setEnabled(False)
        parent.viewbtns.button(5).setStyleSheet(parent.styleInactive)
        if "meanImg_chan2" not in parent.ops:
            parent.viewbtns.button(6).setEnabled(False)
            parent.viewbtns.button(6).setStyleSheet(parent.styleInactive)

    for b in range(len(parent.color_names)):
        if b==5:
            if parent.hasred:
                parent.colorbtns.button(b).setEnabled(True)
                parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        elif b==0:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setChecked(True)
            parent.colorbtns.button(b).setStyleSheet(parent.stylePressed)
        elif b<8:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)

    #parent.applyclass.setStyleSheet(parent.styleUnpressed)
    #parent.applyclass.setEnabled(True)
    b = 0
    for btn in parent.sizebtns.buttons():
        btn.setStyleSheet(parent.styleUnpressed)
        btn.setEnabled(True)
        if b == 0:
            btn.setChecked(True)
            btn.setStyleSheet(parent.stylePressed)
            btn.press(parent)
        b += 1
    for b in range(3):
        if b==0:
            parent.topbtns.button(b).setEnabled(True)
            parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
        else:
            parent.topbtns.button(b).setEnabled(False)
            parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
    # enable classifier menu
    parent.loadClass.setEnabled(True)
    parent.loadTrain.setEnabled(True)
    parent.loadUClass.setEnabled(True)
    parent.loadSClass.setEnabled(True)
    parent.resetDefault.setEnabled(True)
    parent.visualizations.setEnabled(True)
    parent.custommask.setEnabled(True)
    # parent.p1.scene().showExportDialog()

def load_dialog(parent):
    options=QtGui.QFileDialog.Options()
    options |= QtGui.QFileDialog.DontUseNativeDialog
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open stat.npy", filter="stat.npy",
        options=options
    )
    parent.fname = name[0]
    load_proc(parent)

def load_dialog_NWB(parent):
    options=QtGui.QFileDialog.Options()
    options |= QtGui.QFileDialog.DontUseNativeDialog
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open ophys.nwb", filter="*.nwb",
        options=options
    )
    parent.fname = name[0]
    load_NWB(parent)

def load_NWB(parent):
    name = parent.fname
    print(name)
    if 1:
        procs = io.read_nwb(name)
        if procs[1]['nchannels']==2:
            parent.hasred = True
        else:
            parent.hasred = False
        load_to_GUI(parent, os.path.split(name)[0], procs)
            
        parent.loaded = True
    #except Exception as e:
    #    print('ERROR with NWB: %s'%e)

def load_proc(parent):
    name = parent.fname
    print(name)
    try:
        stat = np.load(name, allow_pickle=True)
        ypix = stat[0]["ypix"]
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print('ERROR: this is not a stat.npy file :( '
              '(needs stat[n]["ypix"]!)')
        stat = None
    if stat is not None:
        basename, fname = os.path.split(name)
        goodfolder = True
        try:
            Fcell = np.load(basename + "/F.npy")
            Fneu = np.load(basename + "/Fneu.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print(
                "ERROR: there are no fluorescence traces in this folder "
                "(F.npy/Fneu.npy)"
            )
            goodfolder = False
        try:
            Spks = np.load(basename + "/spks.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("there are no spike deconvolved traces in this folder "
                  "(spks.npy)")
            goodfolder = False
        try:
            ops = np.load(basename + "/ops.npy", allow_pickle=True).item()
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: there is no ops file in this folder (ops.npy)")
            goodfolder = False
        try:
            iscell = np.load(basename + "/iscell.npy")
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype(np.bool)
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no manual labels found (iscell.npy)")
            if goodfolder:
                NN = Fcell.shape[0]
                iscell = np.ones((NN,), np.bool)
                probcell = np.ones((NN,), np.float32)
        try:
            redcell = np.load(basename + "/redcell.npy")
            probredcell = redcell[:,1].copy()
            redcell = redcell[:,0].astype(np.bool)
            parent.hasred = True
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no channel 2 labels found (redcell.npy)")
            parent.hasred = False
            if goodfolder:
                NN = Fcell.shape[0]
                redcell = np.zeros((NN,), np.bool)
                probredcell = np.zeros((NN,), np.float32)

        if goodfolder:
            load_to_GUI(parent, basename, [stat, ops, Fcell, Fneu, Spks, 
                                            iscell, probcell, redcell, probredcell])
            
            parent.loaded = True
        else:
            print("stat.npy found, but other files not in folder")
            Text = ("stat.npy found, but other files missing, "
                    "choose another?")
            load_again(parent, Text)
    else:
        Text = "Incorrect file, not a stat.npy, choose another?"
        load_again(parent, Text)

def load_to_GUI(parent, basename, procs):
    stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell = procs
    parent.basename = basename
    parent.stat = stat
    parent.ops = ops
    parent.Fcell = Fcell
    parent.Fneu = Fneu
    parent.Spks = Spks
    parent.iscell = iscell
    parent.probcell = probcell
    parent.redcell = redcell
    parent.probredcell = probredcell
    parent.notmerged = np.ones_like(parent.iscell).astype(np.bool)
    for n in range(len(parent.stat)):
        parent.stat[n]['chan2_prob'] = parent.probredcell[n]
        parent.stat[n]['inmerge'] = 0
    parent.stat = np.array(parent.stat)
    reggui.make_masks_and_enable_buttons(parent)
    parent.ichosen = 0
    parent.imerge = [0]
    for n in range(len(parent.stat)):
        if 'imerge' not in parent.stat[n]:
            parent.stat[n]['imerge'] = []

def load_behavior(parent):
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    bloaded = False
    try:
        beh = np.load(name)
        bresample=False
        if beh.ndim>1:
            if beh.shape[1] < 2:
                beh = beh.flatten()
                if beh.shape[0] == parent.Fcell.shape[1]:
                    parent.bloaded = True
                    beh_time = np.arange(0, parent.Fcell.shape[1])
            else:
                parent.bloaded = True
                beh_time = beh[:,1]
                beh = beh[:,0]
                bresample=True
        else:
            if beh.shape[0] == parent.Fcell.shape[1]:
                parent.bloaded = True
                beh_time = np.arange(0, parent.Fcell.shape[1])
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.bloaded:
        beh -= beh.min()
        beh /= beh.max()
        parent.beh = beh
        parent.beh_time = beh_time
        if bresample:
            parent.beh_resampled = resample_frames(parent.beh, parent.beh_time, np.arange(0,parent.Fcell.shape[1]))
        else:
            parent.beh_resampled = parent.beh
        b = len(parent.colors)
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        masks.beh_masks(parent)
        traces.plot_trace(parent)
        if hasattr(parent, 'VW'):
            parent.VW.bloaded = parent.bloaded
            parent.VW.beh = parent.beh
            parent.VW.beh_time = parent.beh_time
            parent.VW.plot_traces()
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of data")

def save_redcell(parent):
    np.save(os.path.join(parent.basename, 'redcell.npy'),
            np.concatenate((np.expand_dims(parent.redcell[parent.notmerged],axis=1),
            np.expand_dims(parent.probredcell[parent.notmerged],axis=1)), axis=1))

def save_iscell(parent):
    np.save(
        parent.basename + "/iscell.npy",
        np.concatenate(
            (
                np.expand_dims(parent.iscell[parent.notmerged], axis=1),
                np.expand_dims(parent.probcell[parent.notmerged], axis=1),
            ),
            axis=1,
        ),
    )
    parent.lcell0.setText("%d" % (parent.iscell.sum()))
    parent.lcell1.setText("%d" % (parent.iscell.size - parent.iscell.sum()))

def save_mat(parent):
    print('saving to mat')
    matpath = os.path.join(parent.basename,'Fall.mat')
    if 'date_proc' in parent.ops:
        parent.ops['date_proc'] = []
    scipy.io.savemat(matpath, {'stat': parent.stat,
                         'ops': parent.ops,
                         'F': parent.Fcell,
                         'Fneu': parent.Fneu,
                         'spks': parent.Spks,
                         'iscell': np.concatenate((parent.iscell[:,np.newaxis],
                                                   parent.probcell[:,np.newaxis]), axis=1),
                         'redcell': np.concatenate((np.expand_dims(parent.redcell,axis=1),
                         np.expand_dims(parent.probredcell,axis=1)), axis=1)
                         })

def save_merge(parent):
    print('saving to NPY')
    np.save(os.path.join(parent.basename, 'ops.npy'), parent.ops)
    np.save(os.path.join(parent.basename, 'stat.npy'), parent.stat)
    np.save(os.path.join(parent.basename, 'F.npy'), parent.Fcell)
    np.save(os.path.join(parent.basename, 'Fneu.npy'), parent.Fneu)
    np.save(os.path.join(parent.basename, 'spks.npy'), parent.Spks)
    iscell =  np.concatenate((parent.iscell[:,np.newaxis],
                              parent.probcell[:,np.newaxis]), axis=1)
    np.save(os.path.join(parent.basename, 'iscell.npy'), iscell)
    np.save(os.path.join(parent.basename, 'redcell.npy'),
            np.concatenate((np.expand_dims(parent.redcell,axis=1),
            np.expand_dims(parent.probredcell,axis=1)), axis=1))
    parent.notmerged = np.ones(parent.iscell.size, np.bool)

def load_custom_mask(parent):
    name = QtGui.QFileDialog.getOpenFileName(
        parent, "Open *.npy", filter="*.npy"
    )
    name = name[0]
    cloaded = False
    try:
        mask = np.load(name)
        mask = mask.flatten()
        if mask.size == parent.Fcell.shape[0]:
            parent.cloaded = True
    except (ValueError, KeyError, OSError,
            RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.cloaded:
        parent.custom_mask = mask
        masks.custom_masks(parent)
        M = masks.draw_masks(parent)
        b = len(parent.colors)+1
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        parent.colorbtns.button(b).setChecked(True)
        parent.colorbtns.button(b).press(parent,b)
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of # of ROIs")


def load_again(parent, Text):
    tryagain = QtGui.QMessageBox.question(
        parent, "ERROR", Text, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
    )
    if tryagain == QtGui.QMessageBox.Yes:
        parent.load_dialog()
