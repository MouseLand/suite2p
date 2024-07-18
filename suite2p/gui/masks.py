"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from pathlib import Path
import matplotlib.cm
import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import QPushButton, QButtonGroup, QLabel, QComboBox, QLineEdit
from matplotlib.colors import hsv_to_rgb

import suite2p.gui.merge
from . import io


def make_buttons(parent, b0):
    """ color buttons at row b0 """
    # color buttons
    parent.color_names = [
        "A: random", "S: skew", "D: compact", "F: snr", "G: aspect_ratio",
        "H: chan2_prob", "J: classifier, cell prob=", "K: correlations, bin=",
        "L: corr with 1D var, bin=^^^", "M: rastermap / custom"
    ]
    parent.colorbtns = QButtonGroup(parent)
    clabel = QLabel(parent)
    clabel.setText("<font color='white'>Colors</font>")
    clabel.setFont(parent.boldfont)
    parent.l0.addWidget(clabel, b0, 0, 1, 1)

    iwid = 65

    # add colormaps
    parent.CmapChooser = QComboBox()
    cmaps = [
        "hsv", "viridis", "plasma", "inferno", "magma", "cividis", "viridis_r",
        "plasma_r", "inferno_r", "magma_r", "cividis_r"
    ]
    parent.CmapChooser.addItems(cmaps)
    parent.CmapChooser.setCurrentIndex(0)
    parent.CmapChooser.activated.connect(lambda: cmap_change(parent))
    parent.CmapChooser.setFont(QtGui.QFont("Arial", 8))
    parent.CmapChooser.setFixedWidth(iwid)
    parent.l0.addWidget(parent.CmapChooser, b0, 1, 1, 1)

    nv = b0
    b = 0
    # colorbars for different statistics
    colorsAll = parent.color_names.copy()
    for names in colorsAll:
        btn = ColorButton(b, "&" + names, parent)
        parent.colorbtns.addButton(btn, b)
        if b > 4 and b < 8:
            parent.l0.addWidget(btn, nv + b + 1, 0, 1, 1)
        else:
            parent.l0.addWidget(btn, nv + b + 1, 0, 1, 2)
        btn.setEnabled(False)
        parent.color_names[b] = parent.color_names[b][3:]
        b += 1
    parent.chan2edit = QLineEdit(parent)
    parent.chan2edit.setText("0.6")
    parent.chan2edit.setFixedWidth(iwid)
    parent.chan2edit.setAlignment(QtCore.Qt.AlignRight)
    parent.chan2edit.returnPressed.connect(lambda: chan2_prob(parent))
    parent.l0.addWidget(parent.chan2edit, nv + b - 4, 1, 1, 1)

    parent.probedit = QLineEdit(parent)
    parent.probedit.setText("0.5")
    parent.probedit.setFixedWidth(iwid)
    parent.probedit.setAlignment(QtCore.Qt.AlignRight)
    parent.probedit.returnPressed.connect(lambda: suite2p.gui.merge.apply(parent))
    parent.l0.addWidget(parent.probedit, nv + b - 3, 1, 1, 1)

    parent.binedit = QLineEdit(parent)
    parent.binedit.setValidator(QtGui.QIntValidator(0, 500))
    parent.binedit.setText("1")
    parent.binedit.setFixedWidth(iwid)
    parent.binedit.setAlignment(QtCore.Qt.AlignRight)
    parent.binedit.returnPressed.connect(
        lambda: parent.mode_change(parent.activityMode))
    parent.l0.addWidget(parent.binedit, nv + b - 2, 1, 1, 1)
    b0 = nv + b + 2
    return b0


def cmap_change(parent):
    index = parent.CmapChooser.currentIndex()
    parent.ops_plot["colormap"] = parent.CmapChooser.itemText(index)
    if parent.loaded:
        print("colormap changed to %s, loading..." % parent.ops_plot["colormap"])
        istat = parent.colors["istat"]
        for c in range(1, istat.shape[0]):
            parent.colors["cols"][c] = istat_transform(istat[c],
                                                       parent.ops_plot["colormap"])
            rgb_masks(parent, parent.colors["cols"][c], c)
        parent.colormat = draw_colorbar(parent.ops_plot["colormap"])
        parent.update_plot()


def hsv2rgb(cols):
    cols = cols[:, np.newaxis]
    cols = np.concatenate((cols, np.ones_like(cols), np.ones_like(cols)), axis=-1)
    cols = (255 * hsv_to_rgb(cols)).astype(np.uint8)
    return cols


def make_colors(parent):
    parent.colors["colorbar"] = []
    ncells = len(parent.stat)
    parent.colors["cols"] = np.zeros((len(parent.color_names), ncells, 3), np.uint8)
    parent.colors["istat"] = np.zeros((len(parent.color_names), ncells), np.float32)
    np.random.seed(seed=0)
    allcols = np.random.random((ncells,))
    if "meanImg_chan2" in parent.ops:
        allcols = allcols / 1.4
        allcols = allcols + 0.1
        print(f"number of red cells: {parent.redcell.sum()}")
        parent.randcols = allcols.copy()
        allcols[parent.redcell] = 0
    else:
        parent.randcols = allcols
    parent.colors["istat"][0] = parent.randcols
    parent.colors["cols"][0] = hsv2rgb(allcols)

    b = 0
    for names in parent.color_names[:-3]:
        if b > 0:
            istat = np.zeros((ncells, 1))
            if b < len(parent.color_names) - 2:
                if names in parent.stat[0]:
                    for n in range(0, ncells):
                        istat[n] = parent.stat[n][names]
                istat1 = np.percentile(istat, 2)
                istat99 = np.percentile(istat, 98)
                parent.colors["colorbar"].append(
                    [istat1, (istat99 - istat1) / 2 + istat1, istat99])
                istat = istat - istat1
                istat = istat / (istat99 - istat1)
                istat = np.maximum(0, np.minimum(1, istat))
            else:
                istat = np.expand_dims(parent.probcell, axis=1)
                parent.parent.colors["colorbar"].append([0.0, .5, 1.0])
            col = istat_transform(istat, parent.ops_plot["colormap"])
            parent.colors["cols"][b] = col
            parent.colors["istat"][b] = istat.flatten()
        else:
            parent.colors["colorbar"].append([0, 0.5, 1])
        b += 1
    parent.colors["colorbar"].append([0, 0.5, 1])
    parent.colors["colorbar"].append([0, 0.5, 1])
    parent.colors["colorbar"].append([0, 0.5, 1])

    #parent.ops_plot[4] = corrcols
    #parent.cc = cc


def flip_plot(parent):
    parent.iflip = parent.ichosen
    for n in parent.imerge:
        iscell = int(parent.iscell[n])

        parent.iscell[n] = ~parent.iscell[n]
        parent.ichosen = n
        flip_roi(parent)
        if "imerge" in parent.stat[n]:
            for k in parent.stat[n]["imerge"]:
                parent.iscell[k] = ~parent.iscell[k]

    parent.update_plot()

    # Check if `iscell.npy` file exists
    if not Path(parent.basename).joinpath("iscell.npy").exists():
        # Try the `plane0` folder in case of NWB file loaded
        if Path(parent.basename).joinpath("plane0", "iscell.npy").exists():
            parent.basename = str(Path(parent.basename).joinpath("plane0"))
        else:
            raise FileNotFoundError("Unable to find `iscell.npy` file")

    io.save_iscell(parent)


def chan2_prob(parent):
    chan2prob = float(parent.chan2edit.text())
    if abs(parent.chan2prob - chan2prob) > 1e-3:
        parent.chan2prob = chan2prob
        parent.redcell = parent.probredcell > parent.chan2prob
        chan2_masks(parent)
        parent.update_plot()
        io.save_redcell(parent)


def make_colorbar(parent, b0):
    colorbarW = pg.GraphicsLayoutWidget(parent)
    colorbarW.setMaximumHeight(60)
    colorbarW.setMaximumWidth(150)
    colorbarW.ci.layout.setRowStretchFactor(0, 2)
    colorbarW.ci.layout.setContentsMargins(0, 0, 0, 0)
    parent.l0.addWidget(colorbarW, b0, 0, 1, 2)
    parent.colorbar = pg.ImageItem()
    cbar = colorbarW.addViewBox(row=0, col=0, colspan=3)
    cbar.setMenuEnabled(False)
    cbar.addItem(parent.colorbar)
    parent.clabel = [
        colorbarW.addLabel("0.0", color=[255, 255, 255], row=1, col=0),
        colorbarW.addLabel("0.5", color=[255, 255, 255], row=1, col=1),
        colorbarW.addLabel("1.0", color=[255, 255, 255], row=1, col=2),
    ]


def init_masks(parent):
    """
    creates RGB masks using stat and puts them in M0 or M1 depending on
    whether or not iscell is True for a given ROI
    args:
        settings: mean_image, Vcorr
        stat: xpix,ypix,xext,yext
        iscell: vector with True if ROI is cell
        settings_plot: plotROI, view, color, randcols
    outputs:
        M0: ROIs that are True in iscell
        M1: ROIs that are False in iscell

    """
    stat = parent.stat
    iscell = parent.iscell
    cols = parent.colors["cols"]
    ncells = len(stat)
    Ly = parent.Ly
    Lx = parent.Lx
    parent.rois["Sroi"] = np.zeros((2, Ly, Lx), "bool")
    LamAll = np.zeros((Ly, Lx), np.float32)
    # these have 3 layers
    parent.rois["Lam"] = np.zeros((2, 3, Ly, Lx), np.float32)
    parent.rois["iROI"] = -1 * np.ones((2, 3, Ly, Lx), np.int32)

    if parent.checkBoxN.isChecked():
        parent.checkBoxN.setChecked(False)

    # ignore merged cells
    iignore = np.zeros(ncells, "bool")
    parent.roi_text_labels = []
    for n in np.arange(ncells - 1, -1, -1, int):
        ypix = stat[n]["ypix"]
        if ypix is not None and not iignore[n]:
            if "imerge" in stat[n]:
                for k in stat[n]["imerge"]:
                    iignore[k] = True
                    print(f"ROI {k} in merged ROI")
            xpix = stat[n]["xpix"]
            lam = stat[n]["lam"]
            lam = lam / lam.sum()
            i = int(1 - iscell[n])
            # add cell on top
            parent.rois["iROI"][i, 2, ypix, xpix] = parent.rois["iROI"][i, 1, ypix,
                                                                        xpix]
            parent.rois["iROI"][i, 1, ypix, xpix] = parent.rois["iROI"][i, 0, ypix,
                                                                        xpix]
            parent.rois["iROI"][i, 0, ypix, xpix] = n

            # add weighting to all layers
            parent.rois["Lam"][i, 2, ypix, xpix] = parent.rois["Lam"][i, 1, ypix, xpix]
            parent.rois["Lam"][i, 1, ypix, xpix] = parent.rois["Lam"][i, 0, ypix, xpix]
            parent.rois["Lam"][i, 0, ypix, xpix] = lam
            parent.rois["Sroi"][i, ypix, xpix] = 1
            LamAll[ypix, xpix] = lam
            med = stat[n]["med"]
            cell_str = str(n)
        else:
            cell_str = ""
            med = (0, 0)
        txt = pg.TextItem(cell_str, color=(180, 180, 180), anchor=(0.5, 0.5))
        txt.setPos(med[1], med[0])
        txt.setFont(QtGui.QFont("Times", 8, weight=QtGui.QFont.Bold))
        parent.roi_text_labels.append(txt)
    parent.roi_text_labels = parent.roi_text_labels[::-1]

    parent.rois["LamMean"] = LamAll[LamAll > 1e-10].mean()
    parent.rois["LamNorm"] = np.maximum(
        0, np.minimum(1, 0.75 * parent.rois["Lam"][:, 0] / parent.rois["LamMean"]))
    parent.colors["RGB"] = np.zeros((2, cols.shape[0], Ly, Lx, 4), np.uint8)

    for c in range(0, cols.shape[0]):
        rgb_masks(parent, cols[c], c)


def rgb_masks(parent, col, c):
    for i in range(2):
        #S = np.expand_dims(parent.rois["Sroi"][i],axis=2)
        H = col[parent.rois["iROI"][i, 0], :]
        #H = np.expand_dims(H,axis=2)
        #hsv = np.concatenate((H,S,S),axis=2)
        #rgb = (hsv_to_rgb(hsv)*255).astype(np.uint8)
        parent.colors["RGB"][i, c, :, :, :3] = H


def draw_masks(parent):  #settings, stat, settings_plot, iscell, ichosen):
    """

    creates RGB masks using stat and puts them in M0 or M1 depending on
    whether or not iscell is True for a given ROI
    args:
        settings: mean_image, Vcorr
        stat: xpix,ypix
        iscell: vector with True if ROI is cell
        settings_plot: plotROI, view, color, randcols
    outputs:
        M0: ROIs that are True in iscell
        M1: ROIs that are False in iscell

    """
    ncells = parent.iscell.shape[0]
    plotROI = parent.ops_plot["ROIs_on"]
    view = parent.ops_plot["view"]
    color = parent.ops_plot["color"]
    opacity = parent.ops_plot["opacity"]

    wplot = int(1 - parent.iscell[parent.ichosen])
    # reset transparency
    for i in range(2):
        parent.colors["RGB"][i, color, :, :,
                             3] = (opacity[view == 0] * parent.rois["Sroi"][i] *
                                   parent.rois["LamNorm"][i]).astype(np.uint8)
    M = [
        np.array(parent.colors["RGB"][0, color]),
        np.array(parent.colors["RGB"][1, color])
    ]

    if view == 0:
        for n in parent.imerge:
            ypix = parent.stat[n]["ypix"].flatten()
            xpix = parent.stat[n]["xpix"].flatten()
            v = (parent.rois["iROI"][wplot][:, ypix, xpix] > -1).sum(axis=0) - 1
            v = 1 - v / 3
            M[wplot] = make_chosen_ROI(M[wplot], ypix, xpix, v)
    else:
        for n in parent.imerge:
            ycirc = parent.stat[n]["ycirc"]
            xcirc = parent.stat[n]["xcirc"]
            ypix = parent.stat[n]["ypix"].flatten()
            xpix = parent.stat[n]["xpix"].flatten()
            M[wplot][ypix, xpix, 3] = 0
            col = parent.colors["cols"][color, n]
            sat = 1
            M[wplot] = make_chosen_circle(M[wplot], ycirc, xcirc, col, sat)

    return M[0], M[1]


def make_chosen_ROI(M0, ypix, xpix, v):
    M0[ypix, xpix, :] = np.tile((255 * v[:, np.newaxis]).astype(np.uint8), (1, 4))
    return M0


def make_chosen_circle(M0, ycirc, xcirc, col, sat):
    ncirc = ycirc.size
    M0[ycirc, xcirc, :3] = col  #[np.newaxis,:]
    M0[ycirc, xcirc, 3] = 255
    return M0


def chan2_masks(parent):
    c = 0
    col = parent.randcols.copy()
    col[parent.redcell] = 0
    col = col.flatten()
    parent.colors["cols"][c] = hsv2rgb(col)
    rgb_masks(parent, parent.colors["cols"][c], c)


def custom_masks(parent):
    c = 9
    n = np.array(parent.imerge)
    istat = parent.custom_mask
    istat1 = np.percentile(istat, 1)
    istat99 = np.percentile(istat, 99)
    cl = [istat1, (istat99 - istat1) / 2 + istat1, istat99]
    istat -= istat1
    istat /= istat99 - istat1
    istat = np.maximum(0, np.minimum(1, istat))

    parent.colors["colorbar"][c] = cl
    istat = istat / istat.max()
    col = istat_transform(istat, parent.ops_plot["colormap"])

    parent.colors["cols"][c] = col
    parent.colors["istat"][c] = istat.flatten()

    rgb_masks(parent, col, c)


def rastermap_masks(parent):
    c = 9
    n = np.array(parent.imerge)
    istat = parent.isort
    # no 1D variable loaded -- leave blank
    parent.colors["colorbar"][c] = ([0, istat.max() / 2, istat.max()])

    istat = istat / istat.max()
    col = istat_transform(istat, parent.ops_plot["colormap"])
    col[parent.isort == -1] = 0
    parent.colors["cols"][c] = col
    parent.colors["istat"][c] = istat.flatten()

    rgb_masks(parent, col, c)


def beh_masks(parent):
    c = 8
    n = np.array(parent.imerge)
    nb = int(np.floor(parent.beh_resampled.size / parent.bin))
    sn = np.reshape(parent.beh_resampled[:nb * parent.bin],
                    (nb, parent.bin)).mean(axis=1)
    sn -= sn.mean()
    snstd = (sn**2).mean()**0.5
    cc = np.dot(parent.Fbin, sn.T) / parent.Fbin.shape[-1] / (parent.Fstd * snstd)
    cc[n] = cc.mean()
    istat = cc
    inactive = False
    istat_min = istat.min()
    istat_max = istat.max()
    istat = istat - istat.min()
    istat = istat / istat.max()
    col = istat_transform(istat, parent.ops_plot["colormap"])
    parent.colors["cols"][c] = col
    parent.colors["istat"][c] = istat.flatten()
    parent.colors["colorbar"][c] = [
        istat_min, (istat_max - istat_min) / 2 + istat_min, istat_max
    ]
    rgb_masks(parent, col, c)


def corr_masks(parent):
    c = 7
    n = np.array(parent.imerge)
    sn = parent.Fbin[n].mean(axis=-2).squeeze()
    snstd = (sn**2).mean()**0.5
    cc = np.dot(parent.Fbin, sn.T) / parent.Fbin.shape[-1] / (parent.Fstd * snstd)
    cc[n] = cc.mean()
    istat = cc
    parent.colors["colorbar"][c] = [
        istat.min(), (istat.max() - istat.min()) / 2 + istat.min(),
        istat.max()
    ]
    istat = istat - istat.min()
    istat = istat / istat.max()
    col = istat_transform(istat, parent.ops_plot["colormap"])
    parent.colors["cols"][c] = col
    parent.colors["istat"][c] = istat.flatten()

    rgb_masks(parent, col, c)


def flip_for_class(parent, iscell):
    ncells = iscell.size
    if (iscell == parent.iscell).sum() < 100:
        for n in range(ncells):
            if iscell[n] != parent.iscell[n]:
                parent.iscell[n] = iscell[n]
                parent.ichosen = n
                flip_roi(parent)
    else:
        parent.iscell = iscell
        init_masks(parent)


def plot_colorbar(parent):
    bid = parent.ops_plot["color"]
    if bid == 0:
        parent.colorbar.setImage(np.zeros((20, 100, 3)))
    else:
        parent.colorbar.setImage(parent.colormat)
    for k in range(3):
        parent.clabel[k].setText("%1.2f" % parent.colors["colorbar"][bid][k])


def plot_masks(parent, M):
    #M = parent.RGB[:,:,np.newaxis], parent.Alpha[]
    parent.color1.setImage(M[0], levels=(0., 255.))
    parent.color2.setImage(M[1], levels=(0., 255.))

    #    parent.p1.addItem(txt)
    parent.color1.show()
    parent.color2.show()


def remove_roi(parent, n, i0):
    """
    removes roi n from view i0
    """
    ypix = parent.stat[n]["ypix"]
    xpix = parent.stat[n]["xpix"]
    # cell indices
    ipix = np.array((parent.rois["iROI"][i0, 0, :, :] == n).nonzero()).astype(np.int32)
    ipix1 = np.array((parent.rois["iROI"][i0, 1, :, :] == n).nonzero()).astype(np.int32)
    ipix2 = np.array((parent.rois["iROI"][i0, 2, :, :] == n).nonzero()).astype(np.int32)
    # get rid of cell and push up overlaps on main views
    parent.rois["Lam"][i0, 0, ipix[0, :],
                       ipix[1, :]] = parent.rois["Lam"][i0, 1, ipix[0, :], ipix[1, :]]
    parent.rois["Lam"][i0, 1, ipix[0, :], ipix[1, :]] = 0
    parent.rois["Lam"][i0, 1, ipix1[0, :],
                       ipix1[1, :]] = parent.rois["Lam"][i0, 2, ipix1[0, :],
                                                         ipix1[1, :]]
    parent.rois["Lam"][i0, 2, ipix1[0, :], ipix1[1, :]] = 0
    parent.rois["Lam"][i0, 2, ipix2[0, :], ipix2[1, :]] = 0
    parent.rois["iROI"][i0, 0, ipix[0, :],
                        ipix[1, :]] = parent.rois["iROI"][i0, 1, ipix[0, :], ipix[1, :]]
    parent.rois["iROI"][i0, 1, ipix[0, :], ipix[1, :]] = -1
    parent.rois["iROI"][i0, 1, ipix1[0, :],
                        ipix1[1, :]] = parent.rois["iROI"][i0, 2, ipix1[0, :],
                                                           ipix1[1, :]]
    parent.rois["iROI"][i0, 2, ipix1[0, :], ipix1[1, :]] = -1
    parent.rois["iROI"][i0, 2, ipix2[0, :], ipix2[1, :]] = -1

    # remove +/- 1 ROI exists
    parent.rois["Sroi"][i0, ypix, xpix] = parent.rois["iROI"][i0, 0, ypix, xpix] > 0

    parent.rois["LamNorm"][i0, ypix, xpix] = np.maximum(
        0,
        np.minimum(
            1, 0.75 * parent.rois["Lam"][i0, 0, ypix, xpix] / parent.rois["LamMean"]))


def add_roi(parent, n, i):
    """
    add roi n to view i
    """
    ypix = parent.stat[n]["ypix"]
    xpix = parent.stat[n]["xpix"]
    lam = parent.stat[n]["lam"]
    parent.rois["iROI"][i, 2, ypix, xpix] = parent.rois["iROI"][i, 1, ypix, xpix]
    parent.rois["iROI"][i, 1, ypix, xpix] = parent.rois["iROI"][i, 0, ypix, xpix]
    parent.rois["iROI"][i, 0, ypix, xpix] = n
    parent.rois["Lam"][i, 2, ypix, xpix] = parent.rois["Lam"][i, 1, ypix, xpix]
    parent.rois["Lam"][i, 1, ypix, xpix] = parent.rois["Lam"][i, 0, ypix, xpix]
    parent.rois["Lam"][i, 0, ypix, xpix] = lam  #/ lam.sum()

    # set whether or not an ROI + weighting of pixels
    parent.rois["Sroi"][i, ypix, xpix] = 1
    parent.rois["LamNorm"][:, ypix, xpix] = np.maximum(
        0,
        np.minimum(1, 0.75 * parent.rois["Lam"][:, 0, ypix, xpix] /
                   parent.rois["LamMean"]))


def redraw_masks(parent, ypix, xpix):
    """
    redraw masks after roi added/removed
    """
    for c in range(parent.colors["cols"].shape[0]):
        for i in range(2):
            col = parent.colors["cols"][c]
            rgb = col[parent.rois["iROI"][i, 0, ypix, xpix], :]
            parent.colors["RGB"][i, c, ypix, xpix, :3] = rgb


def flip_roi(parent):
    """
    flips roi to other plot
    there are 3 levels of overlap so this may be buggy if more than 3 cells are on
    top of each other
    """
    cols = parent.ops_plot["color"]
    n = parent.ichosen
    i = int(1 - parent.iscell[n])
    i0 = 1 - i
    if parent.checkBoxN.isChecked():
        if i0 == 0:
            parent.p1.removeItem(parent.roi_text_labels[n])
            parent.p2.addItem(parent.roi_text_labels[n])
        else:
            parent.p2.removeItem(parent.roi_text_labels[n])
            parent.p1.addItem(parent.roi_text_labels[n])

    # remove ROI
    remove_roi(parent, n, i0)
    # add cell to other side (on top) and push down overlaps
    add_roi(parent, n, i)
    # redraw colors
    ypix = parent.stat[n]["ypix"]
    xpix = parent.stat[n]["xpix"]
    redraw_masks(parent, ypix, xpix)


def draw_colorbar(colormap="hsv"):
    H = np.linspace(0, 1, 101).astype(np.float32)
    rgb = istat_transform(H, colormap)
    colormat = np.expand_dims(rgb, axis=0)
    colormat = np.tile(colormat, (20, 1, 1))
    return colormat


def istat_hsv(istat):
    istat = istat / 1.4
    istat = istat + (0.4 / 1.4)
    icols = 1 - istat
    icols = hsv2rgb(icols.flatten())
    return icols


def istat_transform(istat, colormap="hsv"):
    if colormap == "hsv":
        icols = istat_hsv(istat)
    else:
        try:
            cmap = matplotlib.cm.get_cmap(colormap)
            icols = istat
            icols = cmap(icols)[:, :3]
            icols *= 255
            icols = icols.astype(np.uint8)
        except:
            print("bad colormap, using hsv")
            icols = istat_hsv(istat)
    return icols


### Changes colors of ROIs
# button group is exclusive (at least one color is always chosen)
class ColorButton(QPushButton):

    def __init__(self, bid, Text, parent=None):
        super(ColorButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(len(parent.color_names)):
            if parent.colorbtns.button(b).isEnabled():
                parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        self.setStyleSheet(parent.stylePressed)
        parent.ops_plot["color"] = bid
        if not parent.sizebtns.button(1).isChecked():
            if bid == 0:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(False)
                    parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
            else:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(True)
                    parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
        else:
            for b in range(3):
                parent.topbtns.button(b).setEnabled(False)
                parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
        parent.update_plot()
        parent.show()
