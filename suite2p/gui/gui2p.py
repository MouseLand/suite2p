"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
Modified to incorporate Windows 11 styling, including system theme detection, rounded corners, and improved visibility.
"""

import os
import pathlib
import shutil
import sys
import warnings
import platform

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QMainWindow, QApplication, QWidget, QGridLayout, QCheckBox, QLineEdit, QLabel

# Import local modules (adjust the relative import paths as needed)
from . import buttons, graphics, menus, io, merge, views, classgui, traces, masks
from .. import run_s2p, default_ops


def detect_windows_theme():
    """
    Detects whether Windows is using a light or dark theme.
    Returns "light" or "dark" (default is light).
    """
    theme = "light"
    if platform.system() == "Windows":
        try:
            import winreg
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key = winreg.OpenKey(registry, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize")
            # Value 1 = light theme, 0 = dark theme
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            theme = "light" if value == 1 else "dark"
        except Exception as e:
            print("Could not detect Windows theme:", e)
    return theme


def set_app_style(app):
    """
    Sets the global palette and style sheet to mimic a Windows 11 look.
    """
    theme = detect_windows_theme()

    # Define colors based on the theme (customize these values to your preference)
    if theme == "dark":
        window_bg    = "#121212"
        window_border= "#333333"
        button_bg    = "#333333"
        button_border= "#555555"
        button_hover = "#444444"
        text_color   = "#ffffff"
    else:
        window_bg    = "#ffffff"
        window_border= "#cccccc"
        button_bg    = "#eeeeee"
        button_border= "#bbbbbb"
        button_hover = "#dddddd"
        text_color   = "#000000"

    # Set the application's palette so that built-in widgets follow the theme
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(window_bg))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(text_color))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(window_bg))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(window_bg))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(text_color))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(text_color))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(text_color))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(button_bg))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(text_color))
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    app.setPalette(palette)

    # Create a style sheet that adds rounded corners, borders, and section outlines
    style_sheet = f"""
    QMainWindow {{
        background-color: {window_bg};
        border: 1px solid {window_border};
        border-radius: 10px;
    }}
    QWidget {{
        background-color: {window_bg};
        color: {text_color};
    }}
    QPushButton {{
        background-color: {button_bg};
        color: {text_color};
        border: 1px solid {button_border};
        border-radius: 8px;
        padding: 5px;
    }}
    QPushButton:checked {{
        background-color: rgb(100,50,100);
        border: 1px solid #555555;
    }}
    QPushButton:hover {{
        background-color: {button_hover};
    }}
    QCheckBox {{
        /* Make sure the checkbox text is visible */
        color: {text_color};
        spacing: 5px; /* space between checkbox and text */
        border: none;
    }}
    QLabel {{
        color: {text_color};
    }}
    
    QRadioButton {{
        color: {text_color};
    }}

    /* These rules apply to all QSliders with vertical orientation */
    QSlider[orientation="vertical"]::groove:vertical {{
        background: #666666;       /* groove color */
        width: 6px;                /* thickness of the groove */
        border-radius: 3px;
    }}
    QSlider[orientation="vertical"]::handle:vertical {{
        background: #ffffff;       /* handle color */
        border: 1px solid #bbbbbb;
        height: 14px;
        margin: -7px 0;           /* so handle is centered on groove */
        border-radius: 7px;       /* make handle circular */
    }}
    
    /* 'sub-page' is the filled portion below the handle(s) in a vertical slider */
    QSlider[orientation="vertical"]::sub-page:vertical {{
        background: #aaaaaa;
        border-radius: 3px;
    }}
    
    /* 'add-page' is the portion above the handle(s) in a vertical slider */
    QSlider[orientation="vertical"]::add-page:vertical {{
        background: #333333;
        border-radius: 3px;
    }}

    QSlider::groove:horizontal {{
        background: #555555;      /* groove color */
        height: 4px;              /* groove thickness */
        border-radius: 2px;       /* rounded corners */
        margin: 2px 0;
    }}

    QSlider::handle:horizontal {{
        background: #ffffff;      /* handle color */
        border: 1px solid #666666;
        width: 14px;
        margin: -5px 0;           /* handle offset so it sits centered on groove */
        border-radius: 7px;       /* make the handle round */
    }}
    QSlider::add-page:horizontal {{
        background: #666666;      /* color for the 'add' side of the slider */
    }}
    QSlider::sub-page:horizontal {{
        background: #999999;      /* color for the 'sub' side of the slider */
    }}
    QSlider::handle:horizontal:hover {{
        background: #dddddd;      /* lighter color on hover */
    }}
    QSlider::tickmarks:horizontal {{
        background: #ffffff;      /* color for tick lines if you show them */
    }}

    QRangeSlider {{
        background-color: #333333;
        border: 1px solid #555555;
        border-radius: 4px;
    }}

    QRangeSlider::handle {{
        background: #ffffff;
        border: 1px solid #666666;
        width: 14px;
        height: 14px;
        border-radius: 7px;
    }}

    /* Make sure QLineEdit and QComboBox have a visible border and correct colors */
    QLineEdit,
    QComboBox {{
        background-color: {window_bg};
        color: {text_color};
        border: 1px solid {button_border};
        border-radius: 4px;
        padding: 3px;
    }}

    /* If you have spin boxes or text edits, add them similarly */
    QSpinBox,
    QDoubleSpinBox,
    QPlainTextEdit,
    QTextEdit {{
        background-color: {window_bg};
        color: {text_color};
        border: 1px solid {button_border};
        border-radius: 4px;
        padding: 3px;
    }}

    /* Style tooltips so text is visible and background is distinct */
    QToolTip {{
        background-color: {button_bg};
        color: {text_color};
        border: 1px solid {button_border};
        padding: 5px;
    }}

    QGroupBox {{
        border: 1px solid {window_border};
        border-radius: 8px;
        margin-top: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 3px 0 3px;
    }}
    """
    app.setStyleSheet(style_sheet)


class MainWindow(QMainWindow):

    def __init__(self, statfile=None):
        super(MainWindow, self).__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("suite2p (run pipeline or load stat.npy)")
        import suite2p
        s2p_dir = pathlib.Path(suite2p.__file__).parent
        icon_path = os.fspath(s2p_dir.joinpath("logo", "logo.png"))

        app_icon = QtGui.QIcon()
        for size in [16, 24, 32, 48, 64, 256]:
            app_icon.addFile(icon_path, QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)

        # We now use the global style set in set_app_style(), so no hardcoded style here.
        # self.setStyleSheet("QMainWindow {background: 'black';}")

        # Update button styles to include rounded corners and borders
        self.stylePressed   = ("QPushButton { text-align: left; background-color: rgb(100,50,100); "
                                "color: white; border: 1px solid #555555; border-radius: 8px; }")
        self.styleUnpressed = ("QPushButton { text-align: left; background-color: rgb(50,50,50); "
                                "color: white; border: 1px solid #555555; border-radius: 8px; }")
        self.styleInactive  = ("QPushButton { text-align: left; background-color: rgb(50,50,50); "
                                "color: gray; border: 1px solid #555555; border-radius: 8px; }")
        self.loaded = False
        self.ops_plot = []

        ### First time running: check for user files
        user_dir = pathlib.Path.home().joinpath(".suite2p")
        user_dir.mkdir(exist_ok=True)

        # Check for classifier file
        class_dir = user_dir.joinpath("classifiers")
        class_dir.mkdir(exist_ok=True)
        self.classuser = os.fspath(class_dir.joinpath("classifier_user.npy"))
        self.classorig = os.fspath(s2p_dir.joinpath("classifiers", "classifier.npy"))
        if not os.path.isfile(self.classuser):
            shutil.copy(self.classorig, self.classuser)
        self.classfile = self.classuser

        # Check for ops file (for running suite2p)
        ops_dir = user_dir.joinpath("ops")
        ops_dir.mkdir(exist_ok=True)
        self.opsuser = os.fspath(ops_dir.joinpath("ops_user.npy"))
        if not os.path.isfile(self.opsuser):
            np.save(self.opsuser, default_ops())
        self.opsfile = self.opsuser

        menus.mainmenu(self)
        menus.classifier(self)
        menus.visualizations(self)
        menus.registration(self)
        menus.mergebar(self)
        menus.plugins(self)

        self.boldfont = QtGui.QFont("Arial", 10, QtGui.QFont.Bold)

        # Default plot options
        self.ops_plot = {
            "ROIs_on": True,
            "color": 0,
            "view": 0,
            "opacity": [127, 255],
            "saturation": [0, 255],
            "colormap": "hsv"
        }
        self.rois   = {"iROI": 0, "Sroi": 0, "Lam": 0, "LamMean": 0, "LamNorm": 0}
        self.colors = {"RGB": 0, "cols": 0, "colorbar": []}

        # --------- MAIN WIDGET LAYOUT ---------------------
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        b0 = self.make_buttons()
        self.make_graphics(b0)
        # so they’re on top of plot, draw last
        buttons.make_quadrants(self)

        # Initialize merges
        self.merged = []
        self.imerge = [0]
        self.ichosen = 0
        self.rastermap = False
        model = np.load(self.classorig, allow_pickle=True).item()
        self.default_keys = model["keys"]

        # Load initial file if provided
        if statfile is not None:
            self.fname = statfile
            io.load_proc(self)
        self.setAcceptDrops(True)
        self.show()
        self.win.show()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        print(files)
        self.fname = files[0]
        if os.path.splitext(self.fname)[-1] == ".npy":
            io.load_proc(self)
        elif os.path.splitext(self.fname)[-1] == ".nwb":
            io.load_NWB(self)
        else:
            print("invalid extension %s, use .nwb or .npy" %
                  os.path.splitext(self.fname)[-1])

    def make_buttons(self):
        # ROI CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBox = QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet("color: white;")
        self.checkBox.toggle()
        self.checkBox.stateChanged.connect(self.ROIs_on)
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)

        buttons.make_selection(self)
        buttons.make_cellnotcell(self)
        b0 = views.make_buttons(self)
        b0 = masks.make_buttons(self, b0)
        masks.make_colorbar(self, b0)
        b0 += 1
        b0 = classgui.make_buttons(self, b0)
        b0 += 1

        # ------ CELL STATS / ROI SELECTION --------
        self.stats_to_show = [
            "med", "npix", "skew", "compact", "footprint", "aspect_ratio"
        ]
        lilfont = QtGui.QFont("Arial", 8)
        qlabel = QLabel(self)
        qlabel.setFont(self.boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, b0, 0, 1, 1)
        self.ROIedit = QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, b0, 1, 1, 1)
        b0 += 1
        self.ROIstats = []
        self.ROIstats.append(qlabel)
        for k in range(1, len(self.stats_to_show) + 1):
            llabel = QLabel(self.stats_to_show[k - 1])
            self.ROIstats.append(llabel)
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].setStyleSheet("color: white;")
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.addWidget(self.ROIstats[k], b0, 0, 1, 2)
            b0 += 1
        self.l0.addWidget(QLabel(""), b0, 0, 1, 2)
        self.l0.setRowStretch(b0, 1)
        b0 += 2
        b0 = traces.make_buttons(self, b0)

        # zoom to cell CHECKBOX
        self.l0.setVerticalSpacing(4)
        self.checkBoxz = QCheckBox("zoom to cell")
        self.checkBoxz.setStyleSheet("color: white;")
        self.zoomtocell = False
        self.checkBoxz.stateChanged.connect(self.zoom_cell)
        self.l0.addWidget(self.checkBoxz, b0, 15, 1, 2)

        self.checkBoxN = QCheckBox("add ROI # to plot")
        self.checkBoxN.setStyleSheet("color: white;")
        self.roitext = False
        self.checkBoxN.stateChanged.connect(self.roi_text)
        self.checkBoxN.setEnabled(False)
        self.l0.addWidget(self.checkBoxN, b0, 18, 1, 2)

        return b0

    def roi_text(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.Checked:
            for n in range(len(self.roi_text_labels)):
                if self.iscell[n] == 1:
                    self.p1.addItem(self.roi_text_labels[n])
                else:
                    self.p2.addItem(self.roi_text_labels[n])
            self.roitext = True
        else:
            for n in range(len(self.roi_text_labels)):
                if self.iscell[n] == 1:
                    try:
                        self.p1.removeItem(self.roi_text_labels[n])
                    except:
                        pass
                else:
                    try:
                        self.p2.removeItem(self.roi_text_labels[n])
                    except:
                        pass
            self.roitext = False

    def zoom_cell(self, state):
        if self.loaded:
            if QtCore.Qt.CheckState(state) == QtCore.Qt.Checked:
                self.zoomtocell = True
            else:
                self.zoomtocell = False
            self.update_plot()

    def make_graphics(self, b0):
        ##### -------- MAIN PLOTTING AREA ---------- ####################
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, b0 - 1, 30)
        layout = self.win.ci.layout
        # --- cells image
        self.p1 = graphics.ViewBox(parent=self, lockAspect=True, name="plot1",
                                   border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p1, 0, 0)
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.view1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view1.autoDownsample = False
        self.color1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color1.autoDownsample = False
        self.p1.addItem(self.view1)
        self.p1.addItem(self.color1)
        self.view1.setLevels([0, 255])
        self.color1.setLevels([0, 255])
        # --- noncells image
        self.p2 = graphics.ViewBox(parent=self, lockAspect=True, name="plot2",
                                   border=[100, 100, 100], invertY=True)
        self.win.addItem(self.p2, 0, 1)
        self.p2.setMenuEnabled(False)
        self.p2.scene().contextMenuItem = self.p2
        self.view2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view2.autoDownsample = False
        self.color2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color2.autoDownsample = False
        self.p2.addItem(self.view2)
        self.p2.addItem(self.color2)
        self.view2.setLevels([0, 255])
        self.color2.setLevels([0, 255])

        # LINK TWO VIEWS!
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")

        # --- fluorescence trace plot
        self.p3 = graphics.TraceBox(parent=self, invertY=False)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.enableAutoRange(x=True, y=True)
        self.win.addItem(self.p3, row=1, col=0, colspan=2)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        layout = self.win.ci.layout
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)

    def keyPressEvent(self, event):
        if self.loaded:
            if event.modifiers() != QtCore.Qt.ControlModifier and event.modifiers() != QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Return:
                    if event.modifiers() == QtCore.Qt.AltModifier:
                        if len(self.imerge) > 1:
                            merge.do_merge(self)
                elif event.key() == QtCore.Qt.Key_Escape:
                    self.zoom_plot(1)
                    self.zoom_plot(3)
                    self.show()
                elif event.key() == QtCore.Qt.Key_Delete:
                    self.ROI_remove()
                elif event.key() == QtCore.Qt.Key_Q:
                    self.viewbtns.button(0).setChecked(True)
                    self.viewbtns.button(0).press(self, 0)
                elif event.key() == QtCore.Qt.Key_W:
                    self.viewbtns.button(1).setChecked(True)
                    self.viewbtns.button(1).press(self, 1)
                elif event.key() == QtCore.Qt.Key_E:
                    self.viewbtns.button(2).setChecked(True)
                    self.viewbtns.button(2).press(self, 2)
                elif event.key() == QtCore.Qt.Key_R:
                    self.viewbtns.button(3).setChecked(True)
                    self.viewbtns.button(3).press(self, 3)
                elif event.key() == QtCore.Qt.Key_T:
                    self.viewbtns.button(4).setChecked(True)
                    self.viewbtns.button(4).press(self, 4)
                elif event.key() == QtCore.Qt.Key_U:
                    if "meanImg_chan2" in self.ops:
                        self.viewbtns.button(6).setChecked(True)
                        self.viewbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_Y:
                    if "meanImg_chan2_corrected" in self.ops:
                        self.viewbtns.button(5).setChecked(True)
                        self.viewbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_Space:
                    self.checkBox.toggle()
                elif event.key() == QtCore.Qt.Key_N:
                    self.checkBoxd.toggle()
                elif event.key() == QtCore.Qt.Key_B:
                    self.checkBoxn.toggle()
                elif event.key() == QtCore.Qt.Key_V:
                    self.checkBoxt.toggle()
                elif event.key() == QtCore.Qt.Key_A:
                    self.colorbtns.button(0).setChecked(True)
                    self.colorbtns.button(0).press(self, 0)
                elif event.key() == QtCore.Qt.Key_S:
                    self.colorbtns.button(1).setChecked(True)
                    self.colorbtns.button(1).press(self, 1)
                elif event.key() == QtCore.Qt.Key_D:
                    self.colorbtns.button(2).setChecked(True)
                    self.colorbtns.button(2).press(self, 2)
                elif event.key() == QtCore.Qt.Key_F:
                    self.colorbtns.button(3).setChecked(True)
                    self.colorbtns.button(3).press(self, 3)
                elif event.key() == QtCore.Qt.Key_G:
                    self.colorbtns.button(4).setChecked(True)
                    self.colorbtns.button(4).press(self, 4)
                elif event.key() == QtCore.Qt.Key_H:
                    if self.hasred:
                        self.colorbtns.button(5).setChecked(True)
                        self.colorbtns.button(5).press(self, 5)
                elif event.key() == QtCore.Qt.Key_J:
                    self.colorbtns.button(6).setChecked(True)
                    self.colorbtns.button(6).press(self, 6)
                elif event.key() == QtCore.Qt.Key_K:
                    self.colorbtns.button(7).setChecked(True)
                    self.colorbtns.button(7).press(self, 7)
                elif event.key() == QtCore.Qt.Key_L:
                    if self.bloaded:
                        self.colorbtns.button(8).setChecked(True)
                        self.colorbtns.button(8).press(self, 8)
                elif event.key() == QtCore.Qt.Key_M:
                    if self.rastermap:
                        self.colorbtns.button(9).setChecked(True)
                        self.colorbtns.button(9).press(self, 9)
                elif event.key() == QtCore.Qt.Key_Left:
                    ctype = self.iscell[self.ichosen]
                    while -1:
                        self.ichosen = (self.ichosen - 1) % len(self.stat)
                        if self.iscell[self.ichosen] is ctype:
                            break
                    self.imerge = [self.ichosen]
                    self.ROI_remove()
                    self.update_plot()
                elif event.key() == QtCore.Qt.Key_Right:
                    self.ROI_remove()
                    ctype = self.iscell[self.ichosen]
                    while 1:
                        self.ichosen = (self.ichosen + 1) % len(self.stat)
                        if self.iscell[self.ichosen] is ctype:
                            break
                    self.imerge = [self.ichosen]
                    self.update_plot()
                    self.show()
                elif event.key() == QtCore.Qt.Key_Up:
                    masks.flip_plot(self)
                    self.ROI_remove()

    def update_plot(self):
        if self.ops_plot["color"] == 7:
            masks.corr_masks(self)
        masks.plot_colorbar(self)
        self.ichosen_stats()
        views.plot_views(self)
        M = masks.draw_masks(self)
        masks.plot_masks(self, M)
        traces.plot_trace(self)
        if self.zoomtocell:
            self.zoom_to_cell()
        self.p1.show()
        self.p2.show()
        self.win.show()
        self.show()

    def mode_change(self, i):
        """
        Changes the activity mode used when multiple neurons are selected or
        in visualization windows like rastermap or for correlation computation.
        """
        self.activityMode = i
        if self.loaded:
            self.bin = max(1, int(self.binedit.text()))
            nb = int(np.floor(float(self.Fcell.shape[1]) / float(self.bin)))
            if i == 0:
                f = self.Fcell
            elif i == 1:
                f = self.Fneu
            elif i == 2:
                f = self.Fcell - 0.7 * self.Fneu
            else:
                f = self.Spks
            ncells = len(self.stat)
            self.Fbin = f[:, :nb * self.bin].reshape((ncells, nb, self.bin)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1)**0.5
            self.trange = np.arange(0, self.Fcell.shape[1])
            if self.ops_plot["color"] == 8:
                masks.beh_masks(self)
                masks.plot_colorbar(self)
            self.update_plot()

    def top_number_chosen(self):
        self.ntop = int(self.topedit.text())
        if self.loaded:
            if not self.sizebtns.button(1).isChecked():
                for b in [1, 2]:
                    if self.topbtns.button(b).isChecked():
                        self.topbtns.button(b).top_selection(self)
                        self.show()

    def ROI_selection(self):
        draw = False
        if self.sizebtns.button(0).isChecked():
            wplot = 0
            view = self.p1.viewRange()
            draw = True
        elif self.sizebtns.button(2).isChecked():
            wplot = 1
            view = self.p2.viewRange()
            draw = True
        if draw:
            self.ROI_remove()
            self.topbtns.button(0).setStyleSheet(self.stylePressed)
            self.ROIplot = wplot
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, 300)
            dy = np.minimum(dy, 300)
            imx = imx - dx / 2
            imy = imy - dy / 2
            self.ROI = pg.RectROI([imx, imy], [dx, dy], pen="w", sideScalers=True)
            if wplot == 0:
                self.p1.addItem(self.ROI)
            else:
                self.p2.addItem(self.ROI)
            self.ROI_position()
            self.ROI.sigRegionChangeFinished.connect(self.ROI_position)
            self.isROI = True

    def ROI_remove(self):
        if self.isROI:
            if self.ROIplot == 0:
                self.p1.removeItem(self.ROI)
            else:
                self.p2.removeItem(self.ROI)
            self.isROI = False
        if self.sizebtns.button(1).isChecked():
            self.topbtns.button(0).setStyleSheet(self.styleInactive)
            self.topbtns.button(0).setEnabled(False)
        else:
            self.topbtns.button(0).setStyleSheet(self.styleUnpressed)

    def ROI_position(self):
        pos0 = self.ROI.getSceneHandlePositions()
        if self.ROIplot == 0:
            pos = self.p1.mapSceneToView(pos0[0][1])
        else:
            pos = self.p2.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        xrange = xrange[xrange >= 0]
        xrange = xrange[xrange < self.ops["Lx"]]
        yrange = yrange[yrange >= 0]
        yrange = yrange[yrange < self.ops["Ly"]]
        ypix, xpix = np.meshgrid(yrange, xrange)
        self.select_cells(ypix, xpix)

    def select_cells(self, ypix, xpix):
        i = self.ROIplot
        iROI0 = self.rois["iROI"][i, 0, ypix, xpix]
        icells = np.unique(iROI0[iROI0 >= 0])
        self.imerge = []
        for n in icells:
            if (self.rois["iROI"][i, :, ypix, xpix] == n).sum() > 0.6 * self.stat[n]["npix"]:
                self.imerge.append(n)
        if len(self.imerge) > 0:
            self.ichosen = self.imerge[0]
            self.update_plot()
            self.show()

    def number_chosen(self):
        if self.loaded:
            self.ichosen = int(self.ROIedit.text())
            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.imerge = [self.ichosen]
            self.update_plot()
            self.show()

    def ROIs_on(self, state):
        if QtCore.Qt.CheckState(state) == QtCore.Qt.Checked:
            self.ops_plot["ROIs_on"] = True
            self.p1.addItem(self.color1)
            self.p2.addItem(self.color2)
        else:
            self.ops_plot["ROIs_on"] = False
            self.p1.removeItem(self.color1)
            self.p2.removeItem(self.color2)
        self.win.show()
        self.show()

    def plot_clicked(self, event):
        """Left-click chooses a cell, right-click flips cell to other view"""
        flip = False
        choose = False
        zoom = False
        replot = False
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        if self.loaded:
            for x in items:
                if x == self.img1:
                    pos = self.p1.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 1
                elif x == self.img2:
                    pos = self.p2.mapSceneToView(event.scenePos())
                    posy = pos.x()
                    posx = pos.y()
                    iplot = 2
                elif x == self.p3:
                    iplot = 3
                elif ((x == self.p1 or x == self.p2) and x != self.img1 and x != self.img2):
                    iplot = 4
                    if event.double():
                        zoom = True
                if iplot == 1 or iplot == 2:
                    if event.button() == 2:
                        flip = True
                    elif event.button() == 1:
                        if event.double():
                            zoom = True
                        else:
                            choose = True
                if iplot == 3 and event.double():
                    zoom = True
                posy = int(posy)
                posx = int(posx)
                if zoom:
                    self.zoom_plot(iplot)
                if (choose or flip) and (iplot == 1 or iplot == 2):
                    ichosen = int(self.iROI[iplot - 1, 0, posx, posy])
                    if ichosen < 0:
                        choose = False
                        flip = False
                if choose:
                    merged = False
                    if event.modifiers() == QtCore.Qt.ShiftModifier or event.modifiers() == QtCore.Qt.ControlModifier:
                        if self.iscell[self.imerge[0]] == self.iscell[ichosen]:
                            if ichosen not in self.imerge:
                                self.imerge.append(ichosen)
                                self.ichosen = ichosen
                                merged = True
                            elif ichosen in self.imerge and len(self.imerge) > 1:
                                self.imerge.remove(ichosen)
                                self.ichosen = self.imerge[0]
                                merged = True
                    if not merged:
                        self.imerge = [ichosen]
                        self.ichosen = ichosen
                if flip:
                    if ichosen not in self.imerge:
                        self.imerge = [ichosen]
                        self.ichosen = ichosen
                    self.flip_plot(iplot)
                if choose or flip or replot:
                    if self.isROI:
                        self.ROI_remove()
                    if not self.sizebtns.button(1).isChecked():
                        for btn in self.topbtns.buttons():
                            if btn.isChecked():
                                btn.setStyleSheet(self.styleUnpressed)
                    self.update_plot()
                elif event.button() == 2:
                    if iplot == 1:
                        event.acceptedItem = self.p1
                        self.p1.raiseContextMenu(event)
                    elif iplot == 2:
                        event.acceptedItem = self.p2
                        self.p2.raiseContextMenu(event)

    def ichosen_stats(self):
        n = self.ichosen
        self.ROIedit.setText(str(self.ichosen))
        for k in range(1, len(self.stats_to_show) + 1):
            key = self.stats_to_show[k - 1]
            ival = self.stat[n][key]
            if k == 1:
                self.ROIstats[k].setText(key + ": [%d, %d]" % (ival[0], ival[1]))
            elif k == 2:
                self.ROIstats[k].setText(key + ": %d" % (ival))
            else:
                self.ROIstats[k].setText(key + ": %2.2f" % (ival))

    def zoom_to_cell(self):
        irange = 0.1 * np.array([self.Ly, self.Lx]).max()
        if len(self.imerge) > 1:
            apix = np.zeros((0, 2))
            for i, k in enumerate(self.imerge):
                apix = np.append(
                    apix,
                    np.concatenate((self.stat[k]["ypix"].flatten()[:, np.newaxis],
                                    self.stat[k]["xpix"].flatten()[:, np.newaxis]), axis=1),
                    axis=0)
            imin = apix.min(axis=0)
            imax = apix.max(axis=0)
            icent = apix.mean(axis=0)
            imin[0] = min(icent[0] - irange, imin[0])
            imin[1] = min(icent[1] - irange, imin[1])
            imax[0] = max(icent[0] + irange, imax[0])
            imax[1] = max(icent[1] + irange, imax[1])
        else:
            icent = np.array(self.stat[self.ichosen]["med"])
            imin = icent - irange
            imax = icent + irange
        self.p1.setYRange(imin[0], imax[0])
        self.p1.setXRange(imin[1], imax[1])
        self.p2.setYRange(imin[0], imax[0])
        self.p2.setXRange(imin[1], imax[1])
        self.win.show()
        self.show()


# def run(statfile=None):
#     """
#     Entry point for the GUI. This initializes QApplication,
#     applies the Windows 11–inspired style, and starts the main event loop.
#     """
#     warnings.filterwarnings("ignore")
#     app = QApplication(sys.argv)
    
#     # Set our Windows 11–inspired style and palette.
#     set_app_style(app)
#     # Optionally, use the Fusion style for a more modern look.
#     #app.setStyle("Fusion")
    
#     import suite2p
#     s2ppath = os.path.dirname(os.path.realpath(suite2p.__file__))
#     icon_path = os.path.join(s2ppath, "logo", "logo.png")
#     app_icon = QtGui.QIcon()
#     for size in [16, 24, 32, 48, 64, 256]:
#         app_icon.addFile(icon_path, QtCore.QSize(size, size))
#     app.setWindowIcon(app_icon)
    
#     GUI = MainWindow(statfile=statfile)
#     ret = app.exec_()
#     sys.exit(ret)


def run(statfile=None):
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QApplication(sys.argv)

    # Set our Windows 11–inspired style and palette.
    #set_app_style(app)
    app.setStyle("Fusion")

    import suite2p
    s2ppath = os.path.dirname(os.path.realpath(suite2p.__file__))
    icon_path = os.path.join(s2ppath, "logo", "logo.png")
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(64, 64))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainWindow(statfile=statfile)
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# To run the GUI directly, uncomment the following lines:
# if __name__ == "__main__":
#     run()
