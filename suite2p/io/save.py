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


def save_mat(ops, stat, F, Fneu, spks, iscell, redcell,
             F_chan2=None, Fneu_chan2=None):
    ops_matlab = ops.copy()
    if ops_matlab.get("date_proc"):
        try:
            ops_matlab["date_proc"] = str(
                datetime.strftime(ops_matlab["date_proc"], "%Y-%m-%d %H:%M:%S.%f"))
        except:
            pass
    for k in ops_matlab.keys():
        if isinstance(ops_matlab[k], (pathlib.WindowsPath, pathlib.PosixPath)):
            ops_matlab[k] = os.fspath(ops_matlab[k].absolute())
        elif isinstance(ops_matlab[k], list) and len(ops_matlab[k]) > 0:
            if isinstance(ops_matlab[k][0], (pathlib.WindowsPath, pathlib.PosixPath)):
                ops_matlab[k] = [os.fspath(p.absolute()) for p in ops_matlab[k]]
                logger.info(k, ops_matlab[k])

    stat = np.array(stat, dtype=object)

    if F_chan2 is None:
        scipy.io.savemat(
            file_name=os.path.join(ops["save_path"], "Fall.mat"), mdict={
                "stat": stat,
                "ops": ops_matlab,
                "F": F,
                "Fneu": Fneu,
                "spks": spks,
                "iscell": iscell,
                "redcell": redcell
            })
    else:
        scipy.io.savemat(
            file_name=os.path.join(ops["save_path"], "Fall.mat"), mdict={
                "stat": stat,
                "ops": ops_matlab,
                "F": F,
                "Fneu": Fneu,
                "spks": spks,
                "iscell": iscell,
                "redcell": redcell,
                "F_chan2": F_chan2,
                "Fneu_chan2": Fneu_chan2
            })


def compute_dydx(db1):
    db = db1[0].copy()
    dx = np.zeros(len(db1), np.int64)
    dy = np.zeros(len(db1), np.int64)
    if ("dx" not in db) or ("dy" not in db):
        Lx = db["Lx"]
        Ly = db["Ly"]
        nX = np.ceil(np.sqrt(db["Ly"] * db["Lx"] * len(db1)) / db["Lx"])
        nX = int(nX)
        for j in range(len(db1)):
            dx[j] = (j % nX) * Lx
            dy[j] = int(j / nX) * Ly
    else:
        dx = np.array([o["dx"] for o in db1])
        dy = np.array([o["dy"] for o in db1])
        unq = np.unique(np.vstack((dy, dx)), axis=1)
        nrois = unq.shape[1]
        if nrois < len(db1):
            nplanes = len(db1) // nrois
            Lx = np.array([o["Lx"] for o in db1])
            Ly = np.array([o["Ly"] for o in db1])
            ymax = (dy + Ly).max()
            xmax = (dx + Lx).max()
            nX = np.ceil(np.sqrt(ymax * xmax * nplanes) / xmax)
            nX = int(nX)
            nY = int(np.ceil(len(db1) / nX))
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
    dbs = [
        np.load(os.path.join(f, "db.npy"), allow_pickle=True).item()
        for f in plane_folders
    ]
    settings = np.load(os.path.join(plane_folders[0], "settings.npy"), allow_pickle=True).item()
    if os.path.exists(os.path.join(plane_folders[0], "reg_outputs.npy")):
        dops =  [
        np.load(os.path.join(f, "reg_outputs.npy"), allow_pickle=True).item()
        for f in plane_folders
        ]
        dbs = [{**db, **dop} for db, dop in zip(dbs, dops)]
    
    if os.path.exists(os.path.join(plane_folders[0], "detect_outputs.npy")):
        dops =  [
        np.load(os.path.join(f, "detect_outputs.npy"), allow_pickle=True).item()
        for f in plane_folders
        ]
        dbs = [{**db, **dop} for db, dop in zip(dbs, dops)]
    

    dy, dx = compute_dydx(dbs)
    Ly = np.array([db["Ly"] for db in dbs])
    Lx = np.array([db["Lx"] for db in dbs])
    LY = int(np.amax(dy + Ly))
    LX = int(np.amax(dx + Lx))
    meanImg = np.zeros((LY, LX))
    meanImgE = np.zeros((LY, LX))
    logger.info(plane_folders)
    if dbs[0]["nchannels"] > 1:
        meanImg_chan2 = np.zeros((LY, LX))
    if any(["meanImg_chan2_corrected" in db for db in dbs]):
        meanImg_chan2_corrected = np.zeros((LY, LX))
    if any(["max_proj" in db for db in dbs]):
        max_proj = np.zeros((LY, LX))

    Vcorr = np.zeros((LY, LX))
    Nfr = np.amax(np.array([db["nframes"] for db in dbs]))
    ii = 0
    for k, db in enumerate(dbs):
        fpath = plane_folders[k]
        if not os.path.exists(os.path.join(fpath, "stat.npy")):
            continue
        stat0 = np.load(os.path.join(fpath, "stat.npy"), allow_pickle=True)
        xrange = np.arange(dx[k], dx[k] + Lx[k])
        yrange = np.arange(dy[k], dy[k] + Ly[k])
        meanImg[np.ix_(yrange, xrange)] = db["meanImg"]
        if "meanImgE" in db:
            meanImgE[np.ix_(yrange, xrange)] = db["meanImgE"]
        if db["nchannels"] > 1:
            if "meanImg_chan2" in db:
                meanImg_chan2[np.ix_(yrange, xrange)] = db["meanImg_chan2"]
        if "meanImg_chan2_corrected" in db:
            meanImg_chan2_corrected[np.ix_(yrange,
                                           xrange)] = db["meanImg_chan2_corrected"]

        xrange = np.arange(dx[k] + db["xrange"][0], dx[k] + db["xrange"][-1])
        yrange = np.arange(dy[k] + db["yrange"][0], dy[k] + db["yrange"][-1])
        Vcorr[np.ix_(yrange, xrange)] = db["Vcorr"]
        if "max_proj" in db:
            max_proj[np.ix_(yrange, xrange)] = db["max_proj"]
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
    
    db["meanImg"] = meanImg
    db["meanImgE"] = meanImgE
    if db["nchannels"] > 1:
        db["meanImg_chan2"] = meanImg_chan2
    if "meanImg_chan2_corrected" in db:
        db["meanImg_chan2_corrected"] = meanImg_chan2_corrected
    if "max_proj" in db:
        db["max_proj"] = max_proj
    db["Vcorr"] = Vcorr
    db["Ly"] = LY
    db["Lx"] = LX
    db["xrange"] = [0, db["Lx"]]
    db["yrange"] = [0, db["Ly"]]

    fpath = os.path.join(save_folder, "combined")

    if not os.path.isdir(fpath):
        os.makedirs(fpath)

    db["save_path"] = fpath

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
        np.save(os.path.join(fpath, "db.npy"), db)
        np.save(os.path.join(fpath, "stat.npy"), stat)
        np.save(os.path.join(fpath, "settings.npy"), settings)

        # save as matlab file
        if settings["io"]["save_mat"]:
            matpath = os.path.join(db["save_path"], "Fall.mat")
            save_mat(db, stat, F, Fneu, spks, iscell, redcell)

    return (stat, db, settings, F, Fneu, spks, 
            iscell[:,0], iscell[:,1], 
            redcell[:,0], redcell[:,1], hasred)
