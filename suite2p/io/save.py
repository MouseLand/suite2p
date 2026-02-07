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
    """
    Save Suite2p results to a MATLAB .mat file.

    Converts pathlib paths to strings, replaces None values with empty arrays,
    and writes all results to "Fall.mat" in the save path specified in `ops`.

    Parameters
    ----------
    ops : dict
        Suite2p options dictionary. Must contain "save_path". The "date_proc"
        field is converted to a string if present.
    stat : numpy.ndarray
        Array of ROI statistics dictionaries (one per detected cell).
    F : numpy.ndarray
        Fluorescence traces of shape (n_cells, n_frames).
    Fneu : numpy.ndarray
        Neuropil fluorescence traces of shape (n_cells, n_frames).
    spks : numpy.ndarray
        Deconvolved spike traces of shape (n_cells, n_frames).
    iscell : numpy.ndarray
        Cell classification array of shape (n_cells, 2), with columns for
        binary label and probability.
    redcell : numpy.ndarray
        Red channel cell classification array, or None.
    F_chan2 : numpy.ndarray, optional
        Second channel fluorescence traces of shape (n_cells, n_frames).
    Fneu_chan2 : numpy.ndarray, optional
        Second channel neuropil fluorescence traces of shape (n_cells, n_frames).
    """
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
                logger.info(f"{k}: {ops_matlab[k]}")

    stat = np.array(stat, dtype=object)


    # Check for None values in ops_matlab and replace with empty arrays
    for k, v in ops_matlab.items():
        if v is None:
            logger.warning(f"ops_matlab['{k}'] is None, replacing with empty array")
            ops_matlab[k] = np.array([])

    # Handle None variables by replacing with empty arrays
    if redcell is None:
        logger.warning("redcell is None, replacing with empty array")
        redcell = np.array([])
    if iscell is None:
        logger.warning("iscell is None, replacing with empty array")
        iscell = np.array([])

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
    """
    Compute pixel offsets (dy, dx) for tiling multiple planes/ROIs into a combined view.

    If the databases do not contain "dx" and "dy" fields, arranges planes in a
    grid that best tiles a square. If offsets are present (e.g. mesoscope ROIs),
    uses the physical offsets and further tiles across planes.

    Parameters
    ----------
    db1 : list of dict
        List of per-plane database dictionaries. Each must contain "Ly" and "Lx".
        May contain "dx" and "dy" for physical ROI offsets.

    Returns
    -------
    dy : numpy.ndarray
        Y-offsets (in pixels) for each plane, shape (len(db1),).
    dx : numpy.ndarray
        X-offsets (in pixels) for each plane, shape (len(db1),).
    """
    db = db1[0].copy()
    dx = np.zeros(len(db1), np.int64)
    dy = np.zeros(len(db1), np.int64)
    if ("dx" not in db) or ("dy" not in db) or (db["dx"] is None) or (db["dy"] is None):
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
    """
    Combine all plane folders in save_folder into a single result file.

    Loads per-plane results (stat, F, Fneu, spks, iscell, redcell), shifts ROI
    coordinates by the tiled offsets, and concatenates them into combined arrays.
    Multi-plane recordings are arranged to best tile a square. Multi-ROI
    recordings are arranged by their dx, dy physical localization.

    Parameters
    ----------
    save_folder : str
        Path to the suite2p output folder containing plane subdirectories
        (e.g. "plane0", "plane1", ...).
    save : bool, optional (default True)
        If True, save combined results (F.npy, Fneu.npy, spks.npy, stat.npy,
        db.npy, settings.npy, and optionally Fall.mat) to a "combined"
        subfolder. If False, only iscell.npy (and redcell.npy) are saved.

    Returns
    -------
    stat : numpy.ndarray
        Concatenated ROI statistics across all planes.
    db : dict
        Combined database dictionary with merged mean images and full-frame
        dimensions.
    settings : dict
        Suite2p settings dictionary.
    F : numpy.ndarray
        Combined fluorescence traces of shape (n_cells_total, n_frames).
    Fneu : numpy.ndarray
        Combined neuropil traces of shape (n_cells_total, n_frames).
    spks : numpy.ndarray
        Combined deconvolved spike traces of shape (n_cells_total, n_frames).
    iscell0 : numpy.ndarray
        Binary cell classification labels for each ROI.
    iscell1 : numpy.ndarray
        Cell classification probabilities for each ROI.
    redcell0 : numpy.ndarray
        Binary red-cell labels for each ROI.
    redcell1 : numpy.ndarray
        Red-cell probabilities for each ROI.
    hasred : bool
        Whether red-cell classification data was found.
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
