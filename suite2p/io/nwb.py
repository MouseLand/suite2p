import datetime
import gc
import os
import time
from pathlib import Path

import numpy as np
import scipy
from natsort import natsorted

from .. import run_s2p
from ..detection.stats import roi_stats
from . import utils
from .. import run_s2p, default_ops

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


def nwb_to_binary(ops):
    """convert nwb file to binary (experimental)

    converts single plane single channel nwb file to binary for suite2p processing

    Parameters
    ----------
    ops: dictionary
        requires 'nwb_file' key
        optional keys 'nwb_driver', 'nwb_series'
        uses 'nplanes', 'save_path', 'save_folder', 'fast_disk',
        'nchannels', 'keep_movie_raw', 'look_one_level_down'

    Returns
    -------
        ops : dictionary of first plane
            ops['reg_file'] or ops['raw_file'] is created binary
            assigns keys 'Ly', 'Lx', 'tiffreader', 'first_tiffs',
            'frames_per_folder', 'nframes', 'meanImg', 'meanImg_chan2'

    """

    # force 1 plane 1 chan for now
    ops["nplanes"] = 1
    ops["nchannels"] = 1

    # initialize ops with reg_file and raw_file paths, etc
    ops = utils.init_ops(ops)[0]

    t0 = time.time()
    nplanes = ops["nplanes"]
    nchannels = ops["nchannels"]

    batch_size = ops["batch_size"]
    batch_size = int(nplanes * nchannels * np.ceil(batch_size / (nplanes * nchannels)))

    # open reg_file (and when available reg_file_chan2)
    if "keep_movie_raw" in ops and ops["keep_movie_raw"]:
        reg_file = open(ops["raw_file"], "wb")
        if nchannels > 1:
            reg_file_chan2 = open(ops["raw_file_chan2"], "wb")
    else:
        reg_file = open(ops["reg_file"], "wb")
        if nchannels > 1:
            reg_file_chan2 = open(ops["reg_file_chan2"], "wb")

    nwb_driver = None
    if ops.get("nwb_driver") and isinstance(nwb_driver, str):
        nwb_driver = ops["nwb_driver"]

    with NWBHDF5IO(ops["nwb_file"], "r", driver=nwb_driver) as fio:
        nwbfile = fio.read()

        # get TwoPhotonSeries
        if not ops.get("nwb_series"):
            TwoPhotonSeries_names = []
            for v in nwbfile.acquisition.values():
                if isinstance(v, TwoPhotonSeries):
                    TwoPhotonSeries_names.append(v.name)
            if len(TwoPhotonSeries_names) == 0:
                raise ValueError("no TwoPhotonSeries in NWB file")
            elif len(TwoPhotonSeries_names) > 1:
                raise Warning(
                    "more than one TwoPhotonSeries in NWB file, choosing first one"
                )
            ops["nwb_series"] = TwoPhotonSeries_names[0]

        series = nwbfile.acquisition[ops["nwb_series"]]
        series_shape = nwbfile.acquisition[ops["nwb_series"]].data.shape
        ops["nframes"] = series_shape[0]
        ops["frames_per_file"] = np.array([ops["nframes"]])
        ops["frames_per_folder"] = np.array([ops["nframes"]])
        ops["meanImg"] = np.zeros((series_shape[1], series_shape[2]), np.float32)
        for ik in np.arange(0, ops["nframes"], batch_size):
            ikend = min(ik + batch_size, ops["nframes"])
            im = series.data[ik:ikend]

            # check if uint16
            if im.dtype.type == np.uint16:
                im = (im // 2).astype(np.int16)
            elif im.dtype.type == np.int32:
                im = (im // 2).astype(np.int16)
            elif im.dtype.type != np.int16:
                im = im.astype(np.int16)

            reg_file.write(bytearray(im))
            ops["meanImg"] += im.astype(np.float32).sum(axis=0)

            if ikend % (batch_size * 4) == 0:
                print(
                    "%d frames of binary, time %0.2f sec." % (ikend, time.time() - t0)
                )
        gc.collect()

    # write ops files
    ops["Ly"], ops["Lx"] = ops["meanImg"].shape
    ops["yrange"] = np.array([0, ops["Ly"]])
    ops["xrange"] = np.array([0, ops["Lx"]])
    ops["meanImg"] /= ops["nframes"]
    if nchannels > 1:
        ops["meanImg_chan2"] /= ops["nframes"]
    # close all binary files and write ops files
    np.save(ops["ops_path"], ops)
    reg_file.close()
    if nchannels > 1:
        reg_file_chan2.close()

    return ops


def read_nwb(fpath):
    """read NWB file for use in the GUI"""
    with NWBHDF5IO(fpath, "r") as fio:
        nwbfile = fio.read()

        # ROIs
        try:
            rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"
            ]["pixel_mask"]
            multiplane = False
        except Exception:
            rois = nwbfile.processing["ophys"]["ImageSegmentation"][
                "PlaneSegmentation"
            ]["voxel_mask"]
            multiplane = True
        stat = []
        for n in range(len(rois)):
            if isinstance(rois[0], np.ndarray):
                stat.append(
                    {
                        "ypix": np.array(
                            [rois[n][i][0].astype("int") for i in range(len(rois[n]))]
                        ),
                        "xpix": np.array(
                            [rois[n][i][1].astype("int") for i in range(len(rois[n]))]
                        ),
                        "lam": np.array([rois[n][i][-1] for i in range(len(rois[n]))]),
                    }
                )
            else:
                stat.append(
                    {
                        "ypix": rois[n]["x"].astype("int"),
                        "xpix": rois[n]["y"].astype("int"),
                        "lam": rois[n]["weight"],
                    }
                )
            if multiplane:
                stat[-1]['iplane'] = int(rois[n][0][-2])
        ops = default_ops()

        if multiplane:
            nplanes = (
                np.max(np.array([stat[n]["iplane"] for n in range(len(stat))])) + 1
            )
        else:
            nplanes = 1
        stat = np.array(stat)

        # ops with backgrounds
        ops1 = []
        for iplane in range(nplanes):
            ops = default_ops()
            bg_strs = ['meanImg', 'Vcorr', 'max_proj', 'meanImg_chan2']
            ops['nchannels'] = 1
            for bstr in bg_strs:
                if (
                    bstr
                    in nwbfile.processing["ophys"]["Backgrounds_%d" % iplane].images
                ):
                    ops[bstr] = np.array(
                        nwbfile.processing["ophys"]["Backgrounds_%d" % iplane][
                            bstr
                        ].data
                    )
                    if bstr == "meanImg_chan2":
                        ops["nchannels"] = 2
            ops["Ly"], ops["Lx"] = ops[bg_strs[0]].shape
            ops["yrange"] = [0, ops["Ly"]]
            ops["xrange"] = [0, ops["Lx"]]
            ops["tau"] = 1.0
            ops["fs"] = nwbfile.acquisition["TwoPhotonSeries"].rate
            ops1.append(ops.copy())

        stat = roi_stats(stat, ops['Ly'], ops['Lx'], ops['aspect'], ops['diameter'])

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
        dF = F - ops["neucoeff"] * Fneu
        for n in range(len(stat)):
            stat[n]["skew"] = scipy.stats.skew(dF[n])

        # cell probabilities
        iscell = [
            ophys["ImageSegmentation"]["PlaneSegmentation"]["iscell"][n]
            for n in range(len(stat))
        ]
        iscell = np.array(iscell)
        probcell = iscell[:, 1]
        iscell = iscell[:, 0].astype("bool")
        redcell = np.zeros_like(iscell)
        probredcell = np.zeros_like(probcell)

        if multiplane:
            ops = ops1[0].copy()
            Lx = ops["Lx"]
            Ly = ops["Ly"]
            nX = np.ceil(np.sqrt(ops["Ly"] * ops["Lx"] * len(ops1)) / ops["Lx"])
            nX = int(nX)
            for j in range(len(ops1)):
                ops1[j]["dx"] = (j % nX) * Lx
                ops1[j]["dy"] = int(j / nX) * Ly

            LY = int(np.amax(np.array([ops["Ly"] + ops["dy"] for ops in ops1])))
            LX = int(np.amax(np.array([ops["Lx"] + ops["dx"] for ops in ops1])))
            meanImg = np.zeros((LY, LX))
            max_proj = np.zeros((LY, LX))
            if ops["nchannels"] > 1:
                meanImg_chan2 = np.zeros((LY, LX))

            Vcorr = np.zeros((LY, LX))
            for k, ops in enumerate(ops1):
                xrange = np.arange(ops["dx"], ops["dx"] + ops["Lx"])
                yrange = np.arange(ops["dy"], ops["dy"] + ops["Ly"])
                meanImg[np.ix_(yrange, xrange)] = ops["meanImg"]
                Vcorr[np.ix_(yrange, xrange)] = ops["Vcorr"]
                max_proj[np.ix_(yrange, xrange)] = ops["max_proj"]
                if ops["nchannels"] > 1:
                    if "meanImg_chan2" in ops:
                        meanImg_chan2[np.ix_(yrange, xrange)] = ops["meanImg_chan2"]
                for j in np.nonzero(
                    np.array([stat[n]["iplane"] == k for n in range(len(stat))])
                )[0]:
                    stat[j]["xpix"] += ops["dx"]
                    stat[j]["ypix"] += ops["dy"]
                    stat[j]["med"][0] += ops["dy"]
                    stat[j]["med"][1] += ops["dx"]
            ops["Vcorr"] = Vcorr
            ops["max_proj"] = max_proj
            ops["meanImg"] = meanImg
            if "meanImg_chan2" in ops:
                ops["meanImg_chan2"] = meanImg_chan2
            ops["Ly"], ops["Lx"] = LY, LX
            ops["yrange"] = [0, LY]
            ops["xrange"] = [0, LX]
    return stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell


def save_nwb(save_folder):
    """convert folder with plane folders to NWB format"""

    plane_folders = natsorted(
        [
            Path(f.path)
            for f in os.scandir(save_folder)
            if f.is_dir() and f.name[:5] == "plane"
        ]
    )
    ops1 = [
        np.load(f.joinpath("ops.npy"), allow_pickle=True).item() for f in plane_folders
    ]
    nchannels = min([ops["nchannels"] for ops in ops1])

    if NWB and not ops1[0]["mesoscan"]:
        if len(ops1) > 1:
            multiplane = True
        else:
            multiplane = False

        ops = ops1[0]
        if "date_proc" in ops:
            session_start_time = ops["date_proc"]
            if not session_start_time.tzinfo:
                session_start_time = session_start_time.astimezone()
        else:
            session_start_time = datetime.datetime.now().astimezone()

        # INITIALIZE NWB FILE
        nwbfile = NWBFile(
            session_description="suite2p_proc",
            identifier=str(ops["data_path"][0]),
            session_start_time=session_start_time,
        )
        print(nwbfile)

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
            imaging_rate=ops["fs"],
            description="standard",
            device=device,
            excitation_lambda=600.0,
            indicator="GCaMP",
            location="V1",
            grid_spacing=([2.0, 2.0, 30.0] if multiplane else [2.0, 2.0]),
            grid_spacing_unit="microns",
        )
        # link to external data
        external_data = ops["filelist"] if "filelist" in ops else [""]
        image_series = TwoPhotonSeries(
            name="TwoPhotonSeries",
            dimension=[ops["Ly"], ops["Lx"]],
            external_file=external_data,
            imaging_plane=imaging_plane,
            starting_frame=[0 for i in range(len(external_data))],
            format="external",
            starting_time=0.0,
            rate=ops["fs"] * ops["nplanes"],
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
            name="ophys", description="optical physiology processed data"
        )
        ophys_module.add(img_seg)

        file_strs = ["F.npy", "Fneu.npy", "spks.npy"]
        file_strs_chan2 = ["F_chan2.npy", "Fneu_chan2.npy"]
        traces, traces_chan2 = [], []
        ncells = np.zeros(len(ops1), dtype=np.int_)
        Nfr = np.array([ops["nframes"] for ops in ops1]).max()
        for iplane, ops in enumerate(ops1):
            if iplane == 0:
                iscell = np.load(os.path.join(ops["save_path"], "iscell.npy"))
                for fstr in file_strs:
                    traces.append(np.load(os.path.join(ops["save_path"], fstr)))
                if nchannels > 1:
                    for fstr in file_strs_chan2:
                        traces_chan2.append(
                            np.load(plane_folders[iplane].joinpath(fstr))
                        )
                PlaneCellsIdx = iplane * np.ones(len(iscell))
            else:
                iscell = np.append(
                    iscell,
                    np.load(os.path.join(ops["save_path"], "iscell.npy")),
                    axis=0,
                )
                for i, fstr in enumerate(file_strs):
                    trace = np.load(os.path.join(ops["save_path"], fstr))
                    if trace.shape[1] < Nfr:
                        fcat = np.zeros(
                            (trace.shape[0], Nfr - trace.shape[1]), "float32"
                        )
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
                    PlaneCellsIdx, iplane * np.ones(len(iscell) - len(PlaneCellsIdx))
                )

            stat = np.load(
                os.path.join(ops["save_path"], "stat.npy"), allow_pickle=True
            )
            ncells[iplane] = len(stat)
            for n in range(ncells[iplane]):
                if multiplane:
                    pixel_mask = np.array(
                        [
                            stat[n]["ypix"],
                            stat[n]["xpix"],
                            iplane * np.ones(stat[n]["npix"]),
                            stat[n]["lam"],
                        ]
                    )
                    ps.add_roi(voxel_mask=pixel_mask.T)
                else:
                    pixel_mask = np.array(
                        [stat[n]["ypix"], stat[n]["xpix"], stat[n]["lam"]]
                    )
                    ps.add_roi(pixel_mask=pixel_mask.T)

        ps.add_column("iscell", "two columns - iscell & probcell", iscell)

        rt_region = []
        for iplane, ops in enumerate(ops1):
            if iplane == 0:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(
                            np.arange(0, ncells[iplane]),
                        ),
                        description=f"ROIs for plane{int(iplane)}",
                    )
                )
            else:
                rt_region.append(
                    ps.create_roi_table_region(
                        region=list(
                            np.arange(
                                np.sum(ncells[:iplane]),
                                ncells[iplane] + np.sum(ncells[:iplane]),
                            )
                        ),
                        description=f"ROIs for plane{int(iplane)}",
                    )
                )

        # FLUORESCENCE (all are required)
        name_strs = ["Fluorescence", "Neuropil", "Deconvolved"]
        name_strs_chan2 = ["Fluorescence_chan2", "Neuropil_chan2"]

        for i, (fstr, nstr) in enumerate(zip(file_strs, name_strs)):
            for iplane, ops in enumerate(ops1):
                roi_resp_series = RoiResponseSeries(
                    name=f"plane{int(iplane)}",
                    data=np.transpose(traces[i][PlaneCellsIdx == iplane]),
                    rois=rt_region[iplane],
                    unit="lumens",
                    rate=ops["fs"],
                )
                if iplane == 0:
                    fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
                else:
                    fl.add_roi_response_series(roi_response_series=roi_resp_series)
            ophys_module.add(fl)

        if nchannels > 1:
            for i, (fstr, nstr) in enumerate(zip(file_strs_chan2, name_strs_chan2)):
                for iplane, ops in enumerate(ops1):
                    roi_resp_series = RoiResponseSeries(
                        name=f"plane{int(iplane)}",
                        data=np.transpose(traces_chan2[i][PlaneCellsIdx == iplane]),
                        rois=rt_region[iplane],
                        unit="lumens",
                        rate=ops["fs"],
                    )

                    if iplane == 0:
                        fl = Fluorescence(
                            roi_response_series=roi_resp_series, name=nstr
                        )
                    else:
                        fl.add_roi_response_series(roi_response_series=roi_resp_series)

                ophys_module.add(fl)

        # BACKGROUNDS
        # (meanImg, Vcorr and max_proj are REQUIRED)
        bg_strs = ["meanImg", "Vcorr", "max_proj", "meanImg_chan2"]
        nplanes = ops["nplanes"]
        for iplane in range(nplanes):
            images = Images("Backgrounds_%d" % iplane)
            for bstr in bg_strs:
                if bstr in ops:
                    if bstr == "Vcorr" or bstr == "max_proj":
                        img = np.zeros((ops["Ly"], ops["Lx"]), np.float32)
                        img[
                            ops["yrange"][0] : ops["yrange"][-1],
                            ops["xrange"][0] : ops["xrange"][-1],
                        ] = ops[bstr]
                    else:
                        img = ops[bstr]
                    images.add_image(GrayscaleImage(name=bstr, data=img))

            ophys_module.add(images)

        with NWBHDF5IO(os.path.join(save_folder, "ophys.nwb"), "w") as fio:
            fio.write(nwbfile)
    else:
        print('pip install pynwb OR don"t use mesoscope recording')
