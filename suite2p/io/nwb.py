"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import datetime
import gc
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import scipy
from natsort import natsorted

logger = logging.getLogger(__name__)


class _CrossPlatformUnpickler(pickle.Unpickler):
    """Unpickler that handles PosixPath/WindowsPath across platforms."""

    def find_class(self, module, name):
        if name == "PosixPath" or name == "WindowsPath":
            return Path
        return super().find_class(module, name)

def _load_npy_cross_platform(path):
    """Load a .npy file that may contain Path objects from a different OS."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        read_header = (np.lib.format.read_array_header_1_0 if major == 1
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(f)
        if dtype.hasobject:
            return _CrossPlatformUnpickler(f).load()
    return np.load(path, allow_pickle=False)

from ..detection.stats import roi_stats
from . import utils
from .. import default_settings

try:
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import (
        Fluorescence,
        ImageSegmentation,
        OpticalChannel,
        RoiResponseSeries,
        TwoPhotonSeries,
    )

    NWB = True
except ModuleNotFoundError:
    NWB = False


def nwb_to_binary(settings):
    """convert nwb file to binary (experimental)

    converts single plane single channel nwb file to binary for suite2p processing

    Parameters
    ----------
    settings: dictionary
        requires "nwb_file" key
        optional keys "nwb_driver", "nwb_series"
        uses "nplanes", "save_path", "save_folder", "fast_disk",
        "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        settings : dictionary of first plane
            settings["reg_file"] or settings["raw_file"] is created binary
            assigns keys "Ly", "Lx", "tiffreader", "first_tiffs",
            "frames_per_folder", "nframes", "meanImg", "meanImg_chan2"

    """

    # force 1 plane 1 chan for now
    settings["nplanes"] = 1
    settings["nchannels"] = 1

    # initialize settings with reg_file and raw_file paths, etc
    settings = utils.init_settings(settings)[0]

    t0 = time.time()
    nplanes = settings["nplanes"]
    nchannels = settings["nchannels"]

    batch_size = settings["batch_size"]
    batch_size = int(nplanes * nchannels * np.ceil(batch_size / (nplanes * nchannels)))

    # open reg_file (and when available reg_file_chan2)
    if "keep_movie_raw" in settings and settings["keep_movie_raw"]:
        reg_file = open(settings["raw_file"], "wb")
        if nchannels > 1:
            reg_file_chan2 = open(settings["raw_file_chan2"], "wb")
    else:
        reg_file = open(settings["reg_file"], "wb")
        if nchannels > 1:
            reg_file_chan2 = open(settings["reg_file_chan2"], "wb")

    nwb_driver = None
    if settings.get("nwb_driver") and isinstance(nwb_driver, str):
        nwb_driver = settings["nwb_driver"]

    with NWBHDF5IO(settings["nwb_file"], "r", driver=nwb_driver) as fio:
        nwbfile = fio.read()

        # get TwoPhotonSeries
        if not settings.get("nwb_series"):
            TwoPhotonSeries_names = []
            for v in nwbfile.acquisition.values():
                if isinstance(v, TwoPhotonSeries):
                    TwoPhotonSeries_names.append(v.name)
            if len(TwoPhotonSeries_names) == 0:
                raise ValueError("no TwoPhotonSeries in NWB file")
            elif len(TwoPhotonSeries_names) > 1:
                raise Warning(
                    "more than one TwoPhotonSeries in NWB file, choosing first one")
            settings["nwb_series"] = TwoPhotonSeries_names[0]

        series = nwbfile.acquisition[settings["nwb_series"]]
        series_shape = nwbfile.acquisition[settings["nwb_series"]].data.shape
        settings["nframes"] = series_shape[0]
        settings["frames_per_file"] = np.array([settings["nframes"]])
        settings["frames_per_folder"] = np.array([settings["nframes"]])
        settings["meanImg"] = np.zeros((series_shape[1], series_shape[2]), np.float32)
        for ik in np.arange(0, settings["nframes"], batch_size):
            ikend = min(ik + batch_size, settings["nframes"])
            im = series.data[ik:ikend]

            # check if uint16
            if im.dtype.type == np.uint16:
                im = (im // 2).astype(np.int16)
            elif im.dtype.type == np.int32:
                im = (im // 2).astype(np.int16)
            elif im.dtype.type != np.int16:
                im = im.astype(np.int16)

            reg_file.write(bytearray(im))
            settings["meanImg"] += im.astype(np.float32).sum(axis=0)

            if ikend % (batch_size * 4) == 0:
                logger.info("%d frames of binary, time %0.2f sec." %
                      (ikend, time.time() - t0))
        gc.collect()

    # write settings files
    settings["Ly"], settings["Lx"] = settings["meanImg"].shape
    settings["yrange"] = np.array([0, settings["Ly"]])
    settings["xrange"] = np.array([0, settings["Lx"]])
    settings["meanImg"] /= settings["nframes"]
    if nchannels > 1:
        settings["meanImg_chan2"] /= settings["nframes"]
    # close all binary files and write settings files
    np.save(settings["settings_path"], settings)
    reg_file.close()
    if nchannels > 1:
        reg_file_chan2.close()

    return settings


def read_nwb(fpath):
    """read NWB file for use in the GUI"""
    with NWBHDF5IO(fpath, "r") as fio:
        nwbfile = fio.read()

        # ROIs
        try:
            rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"]["pixel_mask"]
            multiplane = False
        except Exception:
            rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"]["voxel_mask"]
            multiplane = True
        stat = []
        for n in range(len(rois)):
            if isinstance(rois[0], np.ndarray):
                stat.append({
                    "ypix":
                        np.array(
                            [rois[n][i][0].astype("int") for i in range(len(rois[n]))]),
                    "xpix":
                        np.array(
                            [rois[n][i][1].astype("int") for i in range(len(rois[n]))]),
                    "lam":
                        np.array([rois[n][i][-1] for i in range(len(rois[n]))]),
                })
            else:
                stat.append({
                    "ypix": rois[n]["x"].astype("int"),
                    "xpix": rois[n]["y"].astype("int"),
                    "lam": rois[n]["weight"],
                })
            if multiplane:
                stat[-1]["iplane"] = int(rois[n][0][-2])
        settings = default_settings()

        if multiplane:
            nplanes = (np.max(np.array([stat[n]["iplane"] for n in range(len(stat))])) +
                       1)
        else:
            nplanes = 1
        stat = np.array(stat)

        # settings with backgrounds
        settings1 = []
        for iplane in range(nplanes):
            settings = default_settings()
            bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
            settings["nchannels"] = 1
            for bstr in bg_strs:
                if (bstr in nwbfile.processing["ophys"]["Backgrounds_%d" %
                                                        iplane].images):
                    settings[bstr] = np.array(nwbfile.processing["ophys"]["Backgrounds_%d" %
                                                                     iplane][bstr].data)
                    if bstr == "meanImg_chan2":
                        settings["nchannels"] = 2
            settings["Ly"], settings["Lx"] = settings[bg_strs[0]].shape
            settings["yrange"] = [0, settings["Ly"]]
            settings["xrange"] = [0, settings["Lx"]]
            settings["tau"] = 1.0
            settings["fs"] = nwbfile.acquisition["TwoPhotonSeries"].rate
            settings1.append(settings.copy())

        # fluorescence
        ophys = nwbfile.processing["ophys"]

        def get_fluo(name: str) -> np.ndarray:
            """Extract Fluorescence data."""
            roi_response_series = ophys[name].roi_response_series
            if name in roi_response_series.keys():
                fluo = ophys[name][name].data[:]
            elif "plane0" in roi_response_series.keys():
                for key, value in roi_response_series.items():
                    if key == "plane0":
                        fluo = value.data[:]
                    else:
                        fluo = np.concatenate((fluo, value.data[:]), axis=1)
                fluo = np.transpose(fluo)
            else:
                raise AttributeError(f"Can't find {name} container in {fpath}")
            return fluo

        F = get_fluo("Fluorescence")
        Fneu = get_fluo("Neuropil")
        spks = get_fluo("Deconvolved")

        # cell probabilities
        iscell = [
            ophys["ImageSegmentation"]["PlaneSegmentation"]["iscell"][n]
            for n in range(len(stat))
        ]
        iscell = np.array(iscell)
        probcell = iscell[:, 1]
        iscell_bool = iscell[:, 0].astype("bool")
        # Create redcell as 2-column array for consistency with iscell format
        redcell = np.zeros_like(iscell)
        probredcell = redcell[:, 1]

        if multiplane:
            settings = settings1[0].copy()
            Lx = settings["Lx"]
            Ly = settings["Ly"]
            nX = np.ceil(np.sqrt(settings["Ly"] * settings["Lx"] * len(settings1)) / settings["Lx"])
            nX = int(nX)
            for j in range(len(settings1)):
                settings1[j]["dx"] = (j % nX) * Lx
                settings1[j]["dy"] = int(j / nX) * Ly

            LY = int(np.amax(np.array([settings["Ly"] + settings["dy"] for settings in settings1])))
            LX = int(np.amax(np.array([settings["Lx"] + settings["dx"] for settings in settings1])))
            meanImg = np.zeros((LY, LX))
            max_proj = np.zeros((LY, LX))
            if settings["nchannels"] > 1:
                meanImg_chan2 = np.zeros((LY, LX))

            Vcorr = np.zeros((LY, LX))
            for k, settings in enumerate(settings1):
                xrange = np.arange(settings["dx"], settings["dx"] + settings["Lx"])
                yrange = np.arange(settings["dy"], settings["dy"] + settings["Ly"])
                meanImg[np.ix_(yrange, xrange)] = settings["meanImg"]
                Vcorr[np.ix_(yrange, xrange)] = settings["Vcorr"]
                max_proj[np.ix_(yrange, xrange)] = settings["max_proj"]
                if settings["nchannels"] > 1:
                    if "meanImg_chan2" in settings:
                        meanImg_chan2[np.ix_(yrange, xrange)] = settings["meanImg_chan2"]
                for j in np.nonzero(
                        np.array([stat[n]["iplane"] == k for n in range(len(stat))
                                 ]))[0]:
                    stat[j]["xpix"] += settings["dx"]
                    stat[j]["ypix"] += settings["dy"]
            settings["Vcorr"] = Vcorr
            settings["max_proj"] = max_proj
            settings["meanImg"] = meanImg
            if "meanImg_chan2" in settings:
                settings["meanImg_chan2"] = meanImg_chan2
            settings["Ly"], settings["Lx"] = LY, LX
            settings["yrange"] = [0, LY]
            settings["xrange"] = [0, LX]

        # Compute roi_stats after multiplane coordinates have been adjusted
        stat = roi_stats(stat, settings["Ly"], settings["Lx"], diameter=settings["diameter"])

        # Compute skew after roi_stats (which may filter ROIs)
        dF = F - settings["extraction"]["neuropil_coefficient"] * Fneu
        for n in range(len(stat)):
            stat[n]["skew"] = scipy.stats.skew(dF[n])

    return stat, settings, F, Fneu, spks, iscell, probcell, redcell, probredcell


def save_nwb(save_folder):
    """convert folder with plane folders to NWB format"""

    plane_folders = natsorted([
        Path(f.path)
        for f in os.scandir(save_folder)
        if f.is_dir() and f.name[:5] == "plane"
    ])
    settings1 = [
        np.load(f.joinpath("settings.npy"), allow_pickle=True).item() for f in plane_folders
    ]
    dbs = [
        _load_npy_cross_platform(f.joinpath("db.npy")).item() for f in plane_folders
    ]

    # Load reg_outputs and detect_outputs for background images
    for i, f in enumerate(plane_folders):
        # Merge reg_outputs (contains meanImg, yrange, xrange, etc.)
        reg_path = f.joinpath("reg_outputs.npy")
        if reg_path.exists():
            reg_outputs = np.load(reg_path, allow_pickle=True).item()
            settings1[i] = {**settings1[i], **reg_outputs}

        # Merge detect_outputs (contains Vcorr, max_proj, etc.)
        detect_path = f.joinpath("detect_outputs.npy")
        if detect_path.exists():
            detect_outputs = np.load(detect_path, allow_pickle=True).item()
            settings1[i] = {**settings1[i], **detect_outputs}

        # Add db keys that might be needed (Ly, Lx)
        for key in ["Ly", "Lx"]:
            if key in dbs[i] and key not in settings1[i]:
                settings1[i][key] = dbs[i][key]

    # Get nchannels from the main db or from plane dbs
    nchannels = dbs[0].get("nchannels", 1)

    if NWB and (settings1[0].get("mesoscan") is None):
        if len(settings1) > 1:
            multiplane = True
        else:
            multiplane = False

        settings = settings1[0]
        if "date_proc" in settings:
            session_start_time = settings["date_proc"]
            if not session_start_time.tzinfo:
                session_start_time = session_start_time.astimezone()
        else:
            session_start_time = datetime.datetime.now().astimezone()

        # INITIALIZE NWB FILE
        nwbfile = NWBFile(
            session_description="suite2p_proc",
            identifier=str(dbs[0]["data_path"][0]),
            session_start_time=session_start_time,
        )
        logger.info(nwbfile)

        device = nwbfile.create_device(
            name="Microscope",
            description="My two-photon microscope",
            manufacturer="The best microscope manufacturer",
        )
        optical_channel = OpticalChannel(
            name="OpticalChannel",
            description="an optical channel",
            emission_lambda=500.0,
        )

        imaging_plane = nwbfile.create_imaging_plane(
            name="ImagingPlane",
            optical_channel=optical_channel,
            imaging_rate=settings["fs"],
            description="standard",
            device=device,
            excitation_lambda=600.0,
            indicator="GCaMP",
            location="V1",
            grid_spacing=([2.0, 2.0, 30.0] if multiplane else [2.0, 2.0]),
            grid_spacing_unit="microns",
        )
        # link to external data
        external_data = settings["filelist"] if "filelist" in settings else [""]
        image_series = TwoPhotonSeries(
            name="TwoPhotonSeries",
            dimension=[dbs[0]["Ly"], dbs[0]["Lx"]],
            external_file=external_data,
            imaging_plane=imaging_plane,
            starting_frame=[0 for i in range(len(external_data))],
            format="external",
            starting_time=0.0,
            rate=settings["fs"] * dbs[0]["nplanes"],
        )
        nwbfile.add_acquisition(image_series)

        # processing
        img_seg = ImageSegmentation()
        ps = img_seg.create_plane_segmentation(
            name="PlaneSegmentation",
            description="suite2p output",
            imaging_plane=imaging_plane,
            reference_images=image_series,
        )
        ophys_module = nwbfile.create_processing_module(
            name="ophys", description="optical physiology processed data")
        ophys_module.add(img_seg)

        file_strs = ["F.npy", "Fneu.npy", "spks.npy"]
        file_strs_chan2 = ["F_chan2.npy", "Fneu_chan2.npy"]
        traces, traces_chan2 = [], []
        ncells = np.zeros(len(settings1), dtype=np.int_)
        Nfr = np.array([db["nframes"] for db in dbs]).max()
        for iplane, (settings, db) in enumerate(zip(settings1, dbs)):
            if iplane == 0:
                iscell = np.load(os.path.join(plane_folders[iplane], "iscell.npy"))
                for fstr in file_strs:
                    traces.append(np.load(os.path.join(plane_folders[iplane], fstr)))
                if nchannels > 1:
                    for fstr in file_strs_chan2:
                        traces_chan2.append(
                            np.load(plane_folders[iplane].joinpath(fstr)))
                PlaneCellsIdx = iplane * np.ones(len(iscell))
            else:
                iscell = np.append(
                    iscell,
                    np.load(os.path.join(plane_folders[iplane], "iscell.npy")),
                    axis=0,
                )
                for i, fstr in enumerate(file_strs):
                    trace = np.load(os.path.join(plane_folders[iplane], fstr))
                    if trace.shape[1] < Nfr:
                        fcat = np.zeros((trace.shape[0], Nfr - trace.shape[1]),
                                        "float32")
                        trace = np.concatenate((trace, fcat), axis=1)
                    traces[i] = np.append(traces[i], trace, axis=0)
                if nchannels > 1:
                    for i, fstr in enumerate(file_strs_chan2):
                        traces_chan2[i] = np.append(
                            traces_chan2[i],
                            np.load(plane_folders[iplane].joinpath(fstr)),
                            axis=0,
                        )
                PlaneCellsIdx = np.append(
                    PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx)))

            stat = np.load(os.path.join(plane_folders[iplane], "stat.npy"),
                           allow_pickle=True)
            ncells[iplane] = len(stat)
            for n in range(ncells[iplane]):
                if multiplane:
                    pixel_mask = np.array([
                        stat[n]["ypix"],
                        stat[n]["xpix"],
                        iplane * np.ones(stat[n]["npix"]),
                        stat[n]["lam"],
                    ])
                    ps.add_roi(voxel_mask=pixel_mask.T)
                else:
                    pixel_mask = np.array(
                        [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]])
                    ps.add_roi(pixel_mask=pixel_mask.T)

        ps.add_column("iscell", "two columns - iscell & probcell", iscell)

        rt_region = []
        for iplane, settings in enumerate(settings1):
            if iplane == 0:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(np.arange(0, ncells[iplane]),),
                        description=f"ROIs for plane{int(iplane)}",
                    ))
            else:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(
                            np.arange(
                                np.sum(ncells[:iplane]),
                                ncells[iplane] + np.sum(ncells[:iplane]),
                            )),
                        description=f"ROIs for plane{int(iplane)}",
                    ))

        # FLUORESCENCE (all are required)
        name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]
        name_strs_chan2 = ["Fluorescence_chan2", "Neuropil_chan2"]

        for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
            for iplane, settings in enumerate(settings1):
                roi_resp_series = RoiResponseSeries(
                    name=f"plane{int(iplane)}",
                    data=np.transpose(traces[i][PlaneCellsIdx == iplane]),
                    rois=rt_region[iplane],
                    unit="lumens",
                    rate=settings["fs"],
                )
                if iplane == 0:
                    fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
                else:
                    fl.add_roi_response_series(roi_response_series=roi_resp_series)
            ophys_module.add(fl)

        if nchannels > 1:
            for i, (fstr, nstr) in enumerate(zip(file_strs_chan2, name_strs_chan2)):
                for iplane, settings in enumerate(settings1):
                    roi_resp_series = RoiResponseSeries(
                        name=f"plane{int(iplane)}",
                        data=np.transpose(traces_chan2[i][PlaneCellsIdx == iplane]),
                        rois=rt_region[iplane],
                        unit="lumens",
                        rate=settings["fs"],
                    )

                    if iplane == 0:
                        fl = Fluorescence(roi_response_series=roi_resp_series,
                                          name=nstr)
                    else:
                        fl.add_roi_response_series(roi_response_series=roi_resp_series)

                ophys_module.add(fl)

        # BACKGROUNDS
        # (meanImg, Vcorr and max_proj are REQUIRED)
        bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
        for iplane, settings in enumerate(settings1):
            images = Images("Backgrounds_%d" % iplane)
            for bstr in bg_strs:
                if bstr in settings:
                    if bstr == "Vcorr" or bstr == "max_proj":
                        img = np.zeros((settings["Ly"], settings["Lx"]), np.float32)
                        img[
                            settings["yrange"][0]:settings["yrange"][-1],
                            settings["xrange"][0]:settings["xrange"][-1],
                        ] = settings[bstr]
                    else:
                        img = settings[bstr]
                    images.add_image(GrayscaleImage(name=bstr, data=img))

            ophys_module.add(images)

        with NWBHDF5IO(os.path.join(save_folder, "ophys.nwb"), "w") as fio:
            fio.write(nwbfile)
    else:
        logger.info("pip install pynwb OR don't use mesoscope recording")
