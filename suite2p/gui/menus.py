"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from qtpy import QtGui
from qtpy.QtWidgets import QAction, QMenu
from pkg_resources import iter_entry_points

from . import reggui, drawroi, merge, io, rungui, visualize, classgui
from suite2p.io.nwb import save_nwb
from suite2p.io.utils import get_suite2p_path


def mainmenu(parent):
    main_menu = parent.menuBar()
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    runS2P = QAction("&Run suite2p", parent)
    runS2P.setShortcut("Ctrl+R")
    runS2P.triggered.connect(lambda: run_suite2p(parent))
    parent.addAction(runS2P)

    # load processed data
    loadProc = QAction("&Load processed data", parent)
    loadProc.setShortcut("Ctrl+L")
    loadProc.triggered.connect(lambda: io.load_dialog(parent))
    parent.addAction(loadProc)

    # load processed data
    loadNWB = QAction("Load NWB file", parent)
    loadNWB.triggered.connect(lambda: io.load_dialog_NWB(parent))
    parent.addAction(loadNWB)

    # load folder of processed data
    loadFolder = QAction("Load &Folder with planeX folders", parent)
    loadFolder.setShortcut("Ctrl+F")
    loadFolder.triggered.connect(lambda: io.load_dialog_folder(parent))
    parent.addAction(loadFolder)

    # load a behavioral trace
    parent.loadBeh = QAction("Load behavior or stim trace (1D only)", parent)
    parent.loadBeh.triggered.connect(lambda: io.load_behavior(parent))
    parent.loadBeh.setEnabled(False)
    parent.addAction(parent.loadBeh)

    # save to matlab file
    parent.saveMat = QAction("&Save to mat file (*.mat)", parent)
    parent.saveMat.setShortcut("Ctrl+S")
    parent.saveMat.triggered.connect(lambda: io.save_mat(parent))
    parent.saveMat.setEnabled(False)
    parent.addAction(parent.saveMat)

    # Save NWB file
    parent.saveNWB = QAction("Save NWB file", parent)
    parent.saveNWB.triggered.connect(
        lambda: save_nwb(get_suite2p_path(parent.basename)))
    parent.saveNWB.setEnabled(False)
    parent.addAction(parent.saveNWB)

    # export figure
    exportFig = QAction("Export as image (svg)", parent)
    exportFig.triggered.connect(lambda: io.export_fig(parent))
    exportFig.setEnabled(True)
    parent.addAction(exportFig)

    # export figure
    parent.manual = QAction("Manual labelling", parent)
    parent.manual.triggered.connect(lambda: manual_label(parent))
    parent.manual.setEnabled(False)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(runS2P)
    file_menu.addAction(loadProc)
    file_menu.addAction(loadNWB)
    file_menu.addAction(loadFolder)
    file_menu.addAction(parent.loadBeh)
    file_menu.addAction(parent.saveNWB)
    file_menu.addAction(parent.saveMat)
    file_menu.addAction(exportFig)
    file_menu.addAction(parent.manual)


def classifier(parent):
    main_menu = parent.menuBar()
    # classifier menu
    parent.trainfiles = []
    parent.statlabels = None
    parent.loadMenu = QMenu("Load", parent)
    parent.loadClass = QAction("from file", parent)
    parent.loadClass.triggered.connect(lambda: classgui.load_classifier(parent))
    parent.loadClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadClass)
    parent.loadUClass = QAction("default classifier", parent)
    parent.loadUClass.triggered.connect(
        lambda: classgui.load_default_classifier(parent))
    parent.loadUClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadUClass)
    parent.loadSClass = QAction("built-in classifier", parent)
    parent.loadSClass.triggered.connect(lambda: classgui.load_s2p_classifier(parent))
    parent.loadSClass.setEnabled(False)
    parent.loadMenu.addAction(parent.loadSClass)
    parent.loadTrain = QAction("Build", parent)
    parent.loadTrain.triggered.connect(lambda: classgui.load_list(parent))
    parent.loadTrain.setEnabled(False)
    parent.saveDefault = QAction("Save loaded as default", parent)
    parent.saveDefault.triggered.connect(lambda: classgui.class_default(parent))
    parent.saveDefault.setEnabled(False)
    parent.resetDefault = QAction("Reset default to built-in", parent)
    parent.resetDefault.triggered.connect(lambda: classgui.reset_default(parent))
    parent.resetDefault.setEnabled(True)
    class_menu = main_menu.addMenu("&Classifier")
    class_menu.addMenu(parent.loadMenu)
    class_menu.addAction(parent.loadTrain)
    class_menu.addAction(parent.resetDefault)
    class_menu.addAction(parent.saveDefault)


def visualizations(parent):
    # visualizations menuBar
    main_menu = parent.menuBar()
    vis_menu = main_menu.addMenu("&Visualizations")
    parent.visualizations = QAction("&Visualize selected cells", parent)
    parent.visualizations.triggered.connect(lambda: vis_window(parent))
    parent.visualizations.setEnabled(False)
    vis_menu.addAction(parent.visualizations)
    parent.visualizations.setShortcut("Ctrl+V")
    parent.custommask = QAction("Load custom hue for ROIs (*.npy)", parent)
    parent.custommask.triggered.connect(lambda: io.load_custom_mask(parent))
    parent.custommask.setEnabled(False)
    vis_menu.addAction(parent.custommask)


def registration(parent):
    # registration menuBar
    main_menu = parent.menuBar()
    reg_menu = main_menu.addMenu("&Registration")
    parent.reg = QAction("View registered &binary", parent)
    parent.reg.triggered.connect(lambda: reg_window(parent))
    parent.reg.setShortcut("Ctrl+B")
    parent.reg.setEnabled(True)
    parent.regPC = QAction("View registration &Metrics", parent)
    parent.regPC.triggered.connect(lambda: regPC_window(parent))
    parent.regPC.setShortcut("Ctrl+M")
    parent.regPC.setEnabled(True)
    reg_menu.addAction(parent.reg)
    reg_menu.addAction(parent.regPC)


def mergebar(parent):
    # merge menuBar
    main_menu = parent.menuBar()
    merge_menu = main_menu.addMenu("&Merge ROIs")
    parent.sugMerge = QAction("Auto-suggest merges", parent)
    parent.sugMerge.triggered.connect(lambda: suggest_merge(parent))
    parent.sugMerge.setEnabled(False)
    parent.saveMerge = QAction("&Append merges to npy files", parent)
    parent.saveMerge.triggered.connect(lambda: io.save_merge(parent))
    parent.saveMerge.setEnabled(False)
    merge_menu.addAction(parent.sugMerge)
    merge_menu.addAction(parent.saveMerge)


def plugins(parent):
    # plugin menu
    main_menu = parent.menuBar()
    parent.plugins = {}
    plugin_menu = main_menu.addMenu("&Plugins")
    for entry_pt in iter_entry_points(group="suite2p.plugin", name=None):
        plugin_obj = entry_pt.load()  # load the advertised class from entry_points
        parent.plugins[entry_pt.name] = plugin_obj(
            parent
        )  # initialize an object instance from the loaded class and keep it alive in parent; expose parent to plugin
        action = QAction(
            parent.plugins[entry_pt.name].name, parent
        )  # create plugin menu item with the name property of the loaded class
        action.triggered.connect(parent.plugins[entry_pt.name].trigger
                                )  # attach class method "trigger" to plugin menu action
        plugin_menu.addAction(action)


def run_suite2p(parent):
    RW = rungui.RunWindow(parent)
    RW.show()


def manual_label(parent):
    MW = drawroi.ROIDraw(parent)
    MW.show()


def vis_window(parent):
    parent.VW = visualize.VisWindow(parent)
    parent.VW.show()


def reg_window(parent):
    RW = reggui.BinaryPlayer(parent)
    RW.show()


def regPC_window(parent):
    RW = reggui.PCViewer(parent)
    RW.show()


def suggest_merge(parent):
    MergeWindow = merge.MergeWindow(parent)
    MergeWindow.show()
