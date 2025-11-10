"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os, time
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from qtpy import QtGui
from qtpy.QtWidgets import QFileDialog, QMessageBox

from . import utils, masks, views, graphics, traces, classgui
from .. import io


def export_fig(parent):
    parent.win.scene().contextMenuItem = parent.p1
    parent.win.scene().showExportDialog()


def make_masks_and_enable_buttons(parent):
    parent.checkBox.setChecked(True)
    parent.ops_plot["color"] = 0
    parent.ops_plot["view"] = 0
    parent.colors["cols"] = 0
    parent.colors["istat"] = 0
    if parent.checkBoxN.isChecked():
        parent.roi_text(False)
    parent.roi_text_labels = []
    parent.roitext = False
    parent.checkBoxN.setChecked(False)
    parent.checkBoxN.setEnabled(True)
    parent.loadBeh.setEnabled(True)
    parent.saveMat.setEnabled(True)
    parent.saveNWB.setEnabled(True)
    parent.saveMerge.setEnabled(True)
    parent.sugMerge.setEnabled(True)
    parent.manual.setEnabled(True)
    parent.bloaded = False
    parent.ROI_remove()
    parent.isROI = False
    parent.setWindowTitle(parent.fname)
    # set bin size to be 0.5s by default
    parent.bin = int(parent.ops["tau"] * parent.ops["fs"] / 2)
    parent.binedit.setText(str(parent.bin))
    if "chan2_thres" not in parent.ops:
        parent.ops["chan2_thres"] = 0.6
    parent.chan2prob = parent.ops["chan2_thres"]
    parent.chan2edit.setText(str(parent.chan2prob))
    # add boundaries to stat for ROI overlays
    ncells = len(parent.stat)
    for n in range(0, ncells):
        ypix = parent.stat[n]["ypix"].flatten()
        xpix = parent.stat[n]["xpix"].flatten()
        yext, xext = utils.boundary(ypix, xpix)
        parent.stat[n]["yext"] = yext
        parent.stat[n]["xext"] = xext
        ycirc, xcirc = utils.circle(parent.stat[n]["med"], parent.stat[n]["radius"])
        goodi = ((ycirc >= 0) & (xcirc >= 0) & (ycirc < parent.ops["Ly"]) &
                 (xcirc < parent.ops["Lx"]))
        parent.stat[n]["ycirc"] = ycirc[goodi]
        parent.stat[n]["xcirc"] = xcirc[goodi]
        parent.stat[n]["inmerge"] = 0
    # enable buttons
    enable_views_and_classifier(parent)
    # make views
    views.init_views(parent)
    # make color arrays for various views
    masks.make_colors(parent)

    if parent.iscell.sum() > 0:
        ich = np.nonzero(parent.iscell)[0][0]
    else:
        ich = 0
    parent.ichosen = int(ich)
    parent.imerge = [int(ich)]
    parent.iflip = int(ich)
    parent.ichosen_stats()
    parent.comboBox.setCurrentIndex(2)
    # colorbar
    parent.colormat = masks.draw_colorbar()
    masks.plot_colorbar(parent)
    tic = time.time()
    masks.init_masks(parent)
    M = masks.draw_masks(parent)
    masks.plot_masks(parent, M)
    print(f"time to draw and plot masks: {time.time() - tic : .4f} sec")
    parent.lcell1.setText("%d" % (ncells - parent.iscell.sum()))
    parent.lcell0.setText("%d" % (parent.iscell.sum()))
    graphics.init_range(parent)
    traces.plot_trace(parent)
    parent.xyrat = 1.0
    if (isinstance(parent.ops["diameter"], (list, np.ndarray)) and
            len(parent.ops["diameter"]) > 1 and parent.ops.get("aspect", 1.0)):
        parent.xyrat = parent.ops["diameter"][0] / parent.ops["diameter"][1]
    else:
        parent.xyrat = parent.ops.get("aspect", 1.0)

    parent.p1.setAspectLocked(lock=True, ratio=parent.xyrat)
    parent.p2.setAspectLocked(lock=True, ratio=parent.xyrat)
    #parent.p2.setXLink(parent.p1)
    #parent.p2.setYLink(parent.p1)
    parent.loaded = True
    parent.mode_change(2)
    parent.show()
    # no classifier loaded
    classgui.activate(parent, False)


def enable_views_and_classifier(parent):
    for b in range(9):
        parent.quadbtns.button(b).setEnabled(True)
        parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
    for b in range(len(parent.view_names)):
        parent.viewbtns.button(b).setEnabled(True)
        parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        # parent.viewbtns.button(b).setShortcut(QtGui.QKeySequence("R"))
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
        if b == 5:
            if parent.hasred:
                parent.colorbtns.button(b).setEnabled(True)
                parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        elif b == 0:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setChecked(True)
            parent.colorbtns.button(b).setStyleSheet(parent.stylePressed)
        elif b < 8:
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
        if b == 0:
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
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open stat.npy",
        "filter": "stat.npy",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    parent.fname = name[0]
    load_proc(parent)

def load_dialog_NWB(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open ophys.nwb",
        "filter": "*.nwb",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    parent.fname = name[0]
    load_NWB(parent)

def load_dialog_folder(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open folder with planeX folders",
    }    
    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    parent.fname = name
    load_folder(parent)

def load_NWB(parent):
    name = parent.fname
    print(name)
    try:
        procs = list(io.read_nwb(name))
        if procs[1]["nchannels"] == 2:
            hasred = True
        else:
            hasred = False
        procs.append(hasred)
        load_to_GUI(parent, os.path.split(name)[0], procs)

        parent.loaded = True
    except Exception as e:
        print("ERROR with NWB: %s" % e)


def load_folder(parent):
    print(parent.fname)
    save_folder = parent.fname
    plane_folders = [
        f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
    ]
    stat_found = False
    if len(plane_folders) > 0:
        stat_found = all(
            [os.path.isfile(os.path.join(f, "stat.npy")) for f in plane_folders])
    if not stat_found:
        print("No processed planeX folders in folder")
        return

    # create a combined folder to hold iscell and redcell
    output = io.combined(save_folder, save=False)
    output = list(output)
    output[1] = {**output[1], **output[2]}  # combine db and settings
    del output[2]
    parent.basename = os.path.join(parent.fname, "combined")
    load_to_GUI(parent, parent.basename, output)
    parent.loaded = True
    print(parent.fname)


def load_files(name):
    """ give stat.npy path and load all needed files for suite2p """
    try:
        stat = np.load(name, allow_pickle=True)
        ypix = stat[0]["ypix"]
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a stat.npy file :( "
              "(needs stat[n]['ypix']!)")
        stat = None
    goodfolder = False
    if stat is not None:
        basename, fname = os.path.split(name)
        goodfolder = True
        try:
            Fcell = np.load(basename + "/F.npy")
            Fneu = np.load(basename + "/Fneu.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: there are no fluorescence traces in this folder "
                  "(F.npy/Fneu.npy)")
            goodfolder = False
        try:
            Spks = np.load(basename + "/spks.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("there are no spike deconvolved traces in this folder "
                  "(spks.npy)")
            goodfolder = False
        noops = True
        try:
            ops = np.load(os.path.join(basename, "ops.npy"), allow_pickle=True).item()
            noops = False
        except:
            noops = True
        if noops:
            try:
                settings = np.load(basename + "/settings.npy", allow_pickle=True).item()
                db = np.load(basename + "/db.npy", allow_pickle=True).item()
                try:
                    reg_outputs = np.load(basename + "/reg_outputs.npy", allow_pickle=True).item()
                    detect_outputs = np.load(basename + "/detect_outputs.npy", allow_pickle=True).item()
                    ops = {**db, **settings, **reg_outputs, **detect_outputs}
                except:
                    ops = {**db, **settings}
                    print("no reg_outputs.npy or detect_outputs.npy found")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                if noops:
                    print("ERROR: there is no settings or db file in this folder (settings.npy / db.npy)")
                    goodfolder = False
        try:
            iscell = np.load(basename + "/iscell.npy")
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype("bool")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no manual labels found (iscell.npy)")
            if goodfolder:
                NN = Fcell.shape[0]
                iscell = np.ones((NN,), "bool")
                probcell = np.ones((NN,), np.float32)
        try:
            redcell = np.load(basename + "/redcell.npy")
            probredcell = redcell[:, 1].copy()
            redcell = redcell[:, 0].astype("bool")
            hasred = True
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no channel 2 labels found (redcell.npy)")
            hasred = False
            if goodfolder:
                NN = Fcell.shape[0]
                redcell = np.zeros((NN,), "bool")
                probredcell = np.zeros((NN,), np.float32)
    else:
        print("incorrect file, not a stat.npy")
        return None

    if goodfolder:
        return stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred
    else:
        print("stat.npy found, but other files not in folder")
        return None


def load_proc(parent):
    name = parent.fname
    print(name)
    basename, fname = os.path.split(name)
    output = load_files(name)
    if output is not None:
        load_to_GUI(parent, basename, output)
        parent.loaded = True
    else:
        Text = "Incorrect files, choose another?"
        load_again(parent, Text)


def load_to_GUI(parent, basename, procs):
    stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred = procs
    parent.basename = basename
    parent.stat = stat
    parent.ops = ops
    parent.Fcell = Fcell
    parent.Fneu = Fneu
    parent.Spks = Spks
    # Handle both 1D and 2D iscell formats
    if iscell.ndim == 2:
        parent.iscell = iscell[:, 0].astype("bool")
        parent.probcell = iscell[:, 1]
    else:
        parent.iscell = iscell.astype("bool")
        parent.probcell = probcell
    # Handle both 1D and 2D redcell formats
    if redcell.ndim == 2:
        parent.redcell = redcell[:, 0].astype("bool")
        parent.probredcell = redcell[:, 1]
    else:
        parent.redcell = redcell.astype("bool")
        parent.probredcell = probredcell
    parent.hasred = hasred
    parent.notmerged = np.ones_like(parent.iscell).astype("bool")
    for n in range(len(parent.stat)):
        if parent.hasred:
            parent.stat[n]["chan2_prob"] = parent.probredcell[n]
        if "snr" not in parent.stat[0]:
            dF = Fcell.copy() - 0.7 * Fneu
            snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)        
            del dF
            for n in range(len(parent.stat)):
                parent.stat[n]["snr"] = snr[n]
        parent.stat[n]["inmerge"] = 0
    parent.stat = np.array(parent.stat)
    make_masks_and_enable_buttons(parent)
    parent.ichosen = 0
    parent.imerge = [0]
    for n in range(len(parent.stat)):
        if "imerge" not in parent.stat[n]:
            parent.stat[n]["imerge"] = []


def load_behavior(parent):
    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    bloaded = False
    try:
        beh = np.load(name)
        bresample = False
        if beh.ndim > 1:
            if beh.shape[1] < 2:
                beh = beh.flatten()
                if beh.shape[0] == parent.Fcell.shape[1]:
                    parent.bloaded = True
                    beh_time = np.arange(0, parent.Fcell.shape[1])
            else:
                parent.bloaded = True
                beh_time = beh[:, 1]
                beh = beh[:, 0]
                bresample = True
        else:
            if beh.shape[0] == parent.Fcell.shape[1]:
                parent.bloaded = True
                beh_time = np.arange(0, parent.Fcell.shape[1])
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.bloaded:
        beh -= beh.min()
        beh /= beh.max()
        parent.beh = beh
        parent.beh_time = beh_time
        if bresample:
            parent.beh_resampled = resample_frames(parent.beh, parent.beh_time,
                                                   np.arange(0, parent.Fcell.shape[1]))
        else:
            parent.beh_resampled = parent.beh
        b = 8
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        masks.beh_masks(parent)
        traces.plot_trace(parent)
        if hasattr(parent, "VW"):
            parent.VW.bloaded = parent.bloaded
            parent.VW.beh = parent.beh
            parent.VW.beh_time = parent.beh_time
            parent.VW.plot_traces()
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of data")


def resample_frames(y, x, xt):
    """ resample y (defined at x) at times xt """
    ts = x.size / xt.size
    y = gaussian_filter1d(y, np.ceil(ts / 2), axis=0)
    f = interp1d(x, y, fill_value="extrapolate")
    yt = f(xt)
    return yt


def save_redcell(parent):
    np.save(
        os.path.join(parent.basename, "redcell.npy"),
        np.concatenate((np.expand_dims(parent.redcell[parent.notmerged], axis=1),
                        np.expand_dims(parent.probredcell[parent.notmerged], axis=1)),
                       axis=1))


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
    print("saving to mat")
    matpath = os.path.join(parent.basename, "Fall.mat")
    if "date_proc" in parent.ops:
        parent.ops["date_proc"] = []
    scipy.io.savemat(
        matpath, {
            "stat":
                parent.stat,
            "settings":
                parent.ops,
            "F":
                parent.Fcell,
            "Fneu":
                parent.Fneu,
            "spks":
                parent.Spks,
            "iscell":
                np.concatenate(
                    (parent.iscell[:, np.newaxis], parent.probcell[:, np.newaxis]),
                    axis=1),
            "redcell":
                np.concatenate((np.expand_dims(parent.redcell, axis=1),
                                np.expand_dims(parent.probredcell, axis=1)), axis=1)
        })


def save_merge(parent):
    print("saving to NPY")
    np.save(os.path.join(parent.basename, "settings.npy"), parent.ops)
    np.save(os.path.join(parent.basename, "stat.npy"), parent.stat)
    np.save(os.path.join(parent.basename, "F.npy"), parent.Fcell)
    np.save(os.path.join(parent.basename, "Fneu.npy"), parent.Fneu)
    if parent.hasred:
        np.save(os.path.join(parent.basename, "F_chan2.npy"), parent.F_chan2)
        np.save(os.path.join(parent.basename, "Fneu_chan2.npy"), parent.Fneu_chan2)
        np.save(
            os.path.join(parent.basename, "redcell.npy"),
            np.concatenate((np.expand_dims(
                parent.redcell, axis=1), np.expand_dims(parent.probredcell, axis=1)),
                           axis=1))
    np.save(os.path.join(parent.basename, "spks.npy"), parent.Spks)
    iscell = np.concatenate(
        (parent.iscell[:, np.newaxis], parent.probcell[:, np.newaxis]), axis=1)
    np.save(os.path.join(parent.basename, "iscell.npy"), iscell)

    parent.notmerged = np.ones(parent.iscell.size, "bool")


def load_custom_mask(parent):
    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    cloaded = False
    try:
        mask = np.load(name)
        mask = mask.flatten()
        if mask.size == parent.Fcell.shape[0]:
            b = len(parent.color_names) - 1
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
            cloaded = True
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if cloaded:
        parent.custom_mask = mask
        masks.custom_masks(parent)
        M = masks.draw_masks(parent)
        b = len(parent.colors) + 1
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        parent.colorbtns.button(b).setChecked(True)
        parent.colorbtns.button(b).press(parent, b)
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of # of ROIs")


def load_again(parent, Text):
    tryagain = QMessageBox.question(parent, "ERROR", Text,
                                    QMessageBox.Yes | QMessageBox.No)

    if tryagain == QMessageBox.Yes:
        load_dialog(parent)
