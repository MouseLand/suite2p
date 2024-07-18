"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
from natsort import natsorted
import numpy as np
from datetime import datetime
import scipy
import pathlib
import logging 
logger = logging.getLogger(__name__)


def save_mat(settings, stat, F, Fneu, spks, iscell, redcell,
             F_chan2=None, Fneu_chan2=None):
    settings_matlab = settings.copy()
    if settings_matlab.get("date_proc"):
        try:
            settings_matlab["date_proc"] = str(
                datetime.strftime(settings_matlab["date_proc"], "%Y-%m-%d %H:%M:%S.%f"))
        except:
            pass
    for k in settings_matlab.keys():
        if isinstance(settings_matlab[k], (pathlib.WindowsPath, pathlib.PosixPath)):
            settings_matlab[k] = os.fspath(settings_matlab[k].absolute())
        elif isinstance(settings_matlab[k], list) and len(settings_matlab[k]) > 0:
            if isinstance(settings_matlab[k][0], (pathlib.WindowsPath, pathlib.PosixPath)):
                settings_matlab[k] = [os.fspath(p.absolute()) for p in settings_matlab[k]]
                logger.info(k, settings_matlab[k])

    stat = np.array(stat, dtype=object)

    if F_chan2 is None:
        scipy.io.savemat(
            file_name=os.path.join(settings["save_path"], "Fall.mat"), mdict={
                "stat": stat,
                "settings": settings_matlab,
                "F": F,
                "Fneu": Fneu,
                "spks": spks,
                "iscell": iscell,
                "redcell": redcell
            })
    else:
        scipy.io.savemat(
            file_name=os.path.join(settings["save_path"], "Fall.mat"), mdict={
                "stat": stat,
                "settings": settings_matlab,
                "F": F,
                "Fneu": Fneu,
                "spks": spks,
                "iscell": iscell,
                "redcell": redcell,
                "F_chan2": F_chan2,
                "Fneu_chan2": Fneu_chan2
            })


def compute_dydx(settings1):
    settings = settings1[0].copy()
    dx = np.zeros(len(settings1), np.int64)
    dy = np.zeros(len(settings1), np.int64)
    if ("dx" not in settings) or ("dy" not in settings):
        Lx = settings["Lx"]
        Ly = settings["Ly"]
        nX = np.ceil(np.sqrt(settings["Ly"] * settings["Lx"] * len(settings1)) / settings["Lx"])
        nX = int(nX)
        for j in range(len(settings1)):
            dx[j] = (j % nX) * Lx
            dy[j] = int(j / nX) * Ly
    else:
        dx = np.array([o["dx"] for o in settings1])
        dy = np.array([o["dy"] for o in settings1])
        unq = np.unique(np.vstack((dy, dx)), axis=1)
        nrois = unq.shape[1]
        if nrois < len(settings1):
            nplanes = len(settings1) // nrois
            Lx = np.array([o["Lx"] for o in settings1])
            Ly = np.array([o["Ly"] for o in settings1])
            ymax = (dy + Ly).max()
            xmax = (dx + Lx).max()
            nX = np.ceil(np.sqrt(ymax * xmax * nplanes) / xmax)
            nX = int(nX)
            nY = int(np.ceil(len(settings1) / nX))
            for j in range(nplanes):
                for k in range(nrois):
                    dx[j * nrois + k] += (j % nX) * xmax
                    dy[j * nrois + k] += int(j / nX) * ymax
    return dy, dx


def combined(save_folder, save=True):
    """ Combines all the folders in save_folder into a single result file.

    can turn off saving (for gui loading)

    Multi-plane recordings are arranged to best tile a square.
    Multi-roi recordings are arranged by their dx,dy physical localization.
    Multi-plane / multi-roi recordings are tiled after using dx,dy.
    """
    plane_folders = natsorted([
        f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
    ])
    settings1 = [
        np.load(os.path.join(f, "settings.npy"), allow_pickle=True).item()
        for f in plane_folders
    ]
    dy, dx = compute_dydx(settings1)
    Ly = np.array([settings["Ly"] for settings in settings1])
    Lx = np.array([settings["Lx"] for settings in settings1])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))
    meanImg = np.zeros((LY, LX))
    meanImgE = np.zeros((LY, LX))
    logger.info(settings1[0]["nchannels"], plane_folders)
    if settings1[0]["nchannels"] > 1:
        meanImg_chan2 = np.zeros((LY, LX))
    if any(["meanImg_chan2_corrected" in settings for settings in settings1]):
        meanImg_chan2_corrected = np.zeros((LY, LX))
    if any(["max_proj" in settings for settings in settings1]):
        max_proj = np.zeros((LY, LX))

    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([settings["nframes"] for settings in settings1]))
    ii = 0
    for k, settings in enumerate(settings1):
        fpath = plane_folders[k]
        if not os.path.exists(os.path.join(fpath, "stat.npy")):
            continue
        stat0 = np.load(os.path.join(fpath, "stat.npy"), allow_pickle=True)
        xrange = np.arange(dx[k], dx[k] + Lx[k])
        yrange = np.arange(dy[k], dy[k] + Ly[k])
        meanImg[np.ix_(yrange, xrange)] = settings["meanImg"]
        meanImgE[np.ix_(yrange, xrange)] = settings["meanImgE"]
        if settings["nchannels"] > 1:
            if "meanImg_chan2" in settings:
                meanImg_chan2[np.ix_(yrange, xrange)] = settings["meanImg_chan2"]
        if "meanImg_chan2_corrected" in settings:
            meanImg_chan2_corrected[np.ix_(yrange,
                                           xrange)] = settings["meanImg_chan2_corrected"]

        xrange = np.arange(dx[k] + settings["xrange"][0], dx[k] + settings["xrange"][-1])
        yrange = np.arange(dy[k] + settings["yrange"][0], dy[k] + settings["yrange"][-1])
        Vcorr[np.ix_(yrange, xrange)] = settings["Vcorr"]
        if "max_proj" in settings:
            max_proj[np.ix_(yrange, xrange)] = settings["max_proj"]
        for j in range(len(stat0)):
            stat0[j]["xpix"] += dx[k]
            stat0[j]["ypix"] += dy[k]
            stat0[j]["med"][0] += dy[k]
            stat0[j]["med"][1] += dx[k]
            stat0[j]["iplane"] = k
        F0 = np.load(os.path.join(fpath, "F.npy"))
        Fneu0 = np.load(os.path.join(fpath, "Fneu.npy"))
        spks0 = np.load(os.path.join(fpath, "spks.npy"))
        iscell0 = np.load(os.path.join(fpath, "iscell.npy"))
        if os.path.isfile(os.path.join(fpath, "redcell.npy")):
            redcell0 = np.load(os.path.join(fpath, "redcell.npy"))
            hasred = True
        else:
            redcell0 = []
            hasred = False
        nn, nt = F0.shape
        if nt < Nfr:
            fcat = np.zeros((nn, Nfr - nt), "float32")
            #logger.info(F0.shape)
            #logger.info(fcat.shape)
            F0 = np.concatenate((F0, fcat), axis=1)
            spks0 = np.concatenate((spks0, fcat), axis=1)
            Fneu0 = np.concatenate((Fneu0, fcat), axis=1)
        if ii == 0:
            F, Fneu, spks, stat, iscell, redcell = F0, Fneu0, spks0, stat0, iscell0, redcell0
        else:
            F = np.concatenate((F, F0))
            Fneu = np.concatenate((Fneu, Fneu0))
            spks = np.concatenate((spks, spks0))
            stat = np.concatenate((stat, stat0))
            iscell = np.concatenate((iscell, iscell0))
            if hasred:
                redcell = np.concatenate((redcell, redcell0))
        ii += 1
        logger.info("appended plane %d to combined view" % k)
    logger.info(meanImg_chan2.shape)
    settings["meanImg"] = meanImg
    settings["meanImgE"] = meanImgE
    if settings["nchannels"] > 1:
        settings["meanImg_chan2"] = meanImg_chan2
    if "meanImg_chan2_corrected" in settings:
        settings["meanImg_chan2_corrected"] = meanImg_chan2_corrected
    if "max_proj" in settings:
        settings["max_proj"] = max_proj
    settings["Vcorr"] = Vcorr
    settings["Ly"] = LY
    settings["Lx"] = LX
    settings["xrange"] = [0, settings["Lx"]]
    settings["yrange"] = [0, settings["Ly"]]

    fpath = os.path.join(save_folder, "combined")

    if not os.path.isdir(fpath):
        os.makedirs(fpath)

    settings["save_path"] = fpath

    # need to save iscell regardless (required for GUI function)
    np.save(os.path.join(fpath, "iscell.npy"), iscell)
    if hasred:
        np.save(os.path.join(fpath, "redcell.npy"), redcell)
    else:
        redcell = np.zeros_like(iscell)

    if save:
        np.save(os.path.join(fpath, "F.npy"), F)
        np.save(os.path.join(fpath, "Fneu.npy"), Fneu)
        np.save(os.path.join(fpath, "spks.npy"), spks)
        np.save(os.path.join(fpath, "settings.npy"), settings)
        np.save(os.path.join(fpath, "stat.npy"), stat)

        # save as matlab file
        if settings.get("save_mat"):
            matpath = os.path.join(settings["save_path"], "Fall.mat")
            save_mat(settings, stat, F, Fneu, spks, iscell, redcell)

    return (stat, settings, F, Fneu, spks, 
            iscell[:,0], iscell[:,1], 
            redcell[:,0], redcell[:,1], hasred)
