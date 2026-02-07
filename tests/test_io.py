"""
Tests for the Suite2p IO module
"""
import pathlib
import platform
import re
from pathlib import Path

import numpy as np
import pytest
from natsort import natsorted
from pynwb import NWBHDF5IO
from suite2p import io
from suite2p.io.nwb import read_nwb, save_nwb
from suite2p.io.utils import get_suite2p_path
from suite2p.detection.detect import bin_movie


@pytest.fixture()
def binfile1500(test_settings):
    db, settings = test_settings
    db["file_list"] = ["input_1500.tif"]
    db["input_format"] = "tif"

    # Find files
    fs, first_files = io.get_file_list(db)
    db["file_list"] = fs
    db["first_files"] = first_files

    # Initialize dbs list (one per plane)
    dbs = io.init_dbs(db)

    # Open binary files for writing
    import contextlib
    with contextlib.ExitStack() as stack:
        raw_str = "raw" if db.get("keep_movie_raw", False) else "reg"
        fnames = [db_item[f"{raw_str}_file"] for db_item in dbs]
        files = [stack.enter_context(open(f, "wb")) for f in fnames]

        if db.get("nchannels", 1) > 1:
            fnames_chan2 = [db_item[f"{raw_str}_file_chan2"] for db_item in dbs]
            files_chan2 = [stack.enter_context(open(f, "wb")) for f in fnames_chan2]
        else:
            files_chan2 = None

        dbs = io.tiff_to_binary(dbs, settings, files, files_chan2)

    bin_filename = str(Path(dbs[0]["save_path0"]).joinpath("suite2p/plane0/data.bin"))
    with io.BinaryFile(
        Ly=dbs[0]["Ly"], Lx=dbs[0]["Lx"], filename=bin_filename
    ) as bin_file:
        yield bin_file

#TODO: Simplify this and don't do this weird save_path fixing thing. 
@pytest.fixture(scope="function")
def replace_settings_save_path_with_local_path(request):
    """
    This fixture replaces the `save_path0` variable in the `settings.npy` file
    by its local path version
    """

    # Workaround to load pickled NPY files on Windows containing
    # `PosixPath` objects
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath

    # Get the `data_folder` variable from the running test name    
    data_folder = re.search(r"\[(.*?)(-.*?)?\]", request.node.name).group(1)
    save_folder = Path("data").joinpath("test_outputs", data_folder, "suite2p")

    # Update db.npy in suite2p folder if it exists
    suite2p_db_path = save_folder.joinpath("db.npy")
    original_suite2p_db_save_path0 = None
    if suite2p_db_path.exists():
        suite2p_db = np.load(suite2p_db_path, allow_pickle=True).item()
        original_suite2p_db_save_path0 = suite2p_db.get("save_path0")
        suite2p_db["save_path0"] = str(save_folder.parent.absolute())
        np.save(suite2p_db_path, suite2p_db)

    save_path = {}
    save_db_paths = {}
    plane_folders = [
        dir
        for dir in natsorted(save_folder.iterdir())
        if dir.is_dir() and "plane" in dir.name
    ]
    for plane_idx, plane_dir in enumerate(plane_folders):

        # Temporarily change the `save_folder` variable in the settings.npy file
        settings1 = np.load(plane_dir.joinpath("settings.npy"), allow_pickle=True)
        settings_dict = settings1.item() if settings1.ndim == 0 else settings1.item(0)
        save_path[plane_dir] = settings_dict["save_path0"]
        settings_dict["save_path0"] = str(plane_dir.absolute())
        np.save(plane_dir.joinpath("settings.npy"), np.array(settings_dict))

        # Also update plane db.npy if it exists
        plane_db_path = plane_dir.joinpath("db.npy")
        if plane_db_path.exists():
            plane_db = np.load(plane_db_path, allow_pickle=True).item()
            save_db_paths[plane_dir] = (plane_db.get("save_path"), plane_db.get("save_path0"))
            plane_db["save_path"] = str(plane_dir.absolute())
            plane_db["save_path0"] = str(save_folder.parent.absolute())
            np.save(plane_db_path, plane_db)

    def concat_npy(name: str) -> np.ndarray:
        """Concatenate arrays from NUmPy files."""
        for plane_idx, plane_dir in enumerate(plane_folders):
            if plane_idx == 0:
                array_npy = np.load(
                    plane_dir.joinpath(f"{name}.npy"), allow_pickle=True
                )
            else:
                array_npy = np.append(
                    array_npy,
                    np.load(plane_dir.joinpath(f"{name}.npy"), allow_pickle=True),
                    axis=0,
                )
        return array_npy

    iscell_npy = concat_npy("iscell")
    F_npy = concat_npy("F")
    Fneu_npy = concat_npy("Fneu")
    spks_npy = concat_npy("spks")

    yield save_folder, iscell_npy, F_npy, Fneu_npy, spks_npy

    # Teardown the fixture
    # Restore suite2p db.npy
    if suite2p_db_path.exists() and original_suite2p_db_save_path0 is not None:
        suite2p_db = np.load(suite2p_db_path, allow_pickle=True).item()
        suite2p_db["save_path0"] = original_suite2p_db_save_path0
        np.save(suite2p_db_path, suite2p_db)

    for plane_dir in plane_folders:
        # Undo the changes made in the NumPy file
        settings1 = np.load(plane_dir.joinpath("settings.npy"), allow_pickle=True)
        settings_dict = settings1.item() if settings1.ndim == 0 else settings1.item(0)
        settings_dict["save_path0"] = save_path[plane_dir]
        np.save(plane_dir.joinpath("settings.npy"), np.array(settings_dict))

        # Restore plane db.npy if it exists
        plane_db_path = plane_dir.joinpath("db.npy")
        if plane_db_path.exists() and plane_dir in save_db_paths:
            plane_db = np.load(plane_db_path, allow_pickle=True).item()
            original_save_path, original_save_path0 = save_db_paths[plane_dir]
            if original_save_path is not None:
                plane_db["save_path"] = original_save_path
            if original_save_path0 is not None:
                plane_db["save_path0"] = original_save_path0
            np.save(plane_db_path, plane_db)


def test_h5_to_binary_produces_nonnegative_output_data(test_settings):
    db, settings = test_settings
    db["h5py"] = Path(db["data_path"][0]).joinpath("input.h5")
    db["nplanes"] = 3
    db["nchannels"] = 2
    db["input_format"] = "h5"

    # Find files
    fs, first_files = io.get_file_list(db)
    db["file_list"] = fs
    db["first_files"] = first_files

    # Initialize dbs list (one per plane)
    dbs = io.init_dbs(db)

    # Open binary files for writing
    import contextlib
    with contextlib.ExitStack() as stack:
        raw_str = "raw" if db.get("keep_movie_raw", False) else "reg"
        fnames = [db_item[f"{raw_str}_file"] for db_item in dbs]
        files = [stack.enter_context(open(f, "wb")) for f in fnames]

        if db["nchannels"] > 1:
            fnames_chan2 = [db_item[f"{raw_str}_file_chan2"] for db_item in dbs]
            files_chan2 = [stack.enter_context(open(f, "wb")) for f in fnames_chan2]
        else:
            files_chan2 = None

        dbs = io.h5py_to_binary(dbs, settings, files, files_chan2)

    output_data = io.BinaryFile(
        filename=Path(dbs[0]["save_path0"], "suite2p/plane0/data.bin"),
        Ly=dbs[0]["Ly"],
        Lx=dbs[0]["Lx"],
    ).data
    assert np.all(output_data >= 0)


def test_that_bin_movie_without_badframes_results_in_a_same_size_array(binfile1500):
    yrange = (0, binfile1500.Ly)
    xrange = (0, binfile1500.Lx)
    mov = bin_movie(binfile1500.data, bin_size=1, yrange=yrange, xrange=xrange)
    assert mov.shape == (1500, binfile1500.Ly, binfile1500.Lx)


def test_that_bin_movie_with_badframes_results_in_a_smaller_array(binfile1500):

    np.random.seed(42)
    # Create badframes with ~30% bad frames (so batches consistently have >50% good frames)
    badframes = np.random.random(size=binfile1500.n_frames) < 0.3
    yrange = (0, binfile1500.Ly)
    xrange = (0, binfile1500.Lx)
    mov = bin_movie(binfile1500.data, bin_size=1, yrange=yrange, xrange=xrange, badframes=badframes)

    assert len(mov) < binfile1500.n_frames, "bin_movie didn't produce a smaller array."
    assert len(mov) == len(badframes) - sum(
        badframes
    ), "bin_movie didn't produce the right size array."


def test_that_binaryfile_data_is_repeatable(binfile1500):
    data1 = binfile1500.data
    assert data1.shape == (1500, binfile1500.Ly, binfile1500.Lx)

    data2 = binfile1500.data
    assert data2.shape == (1500, binfile1500.Ly, binfile1500.Lx)

    assert np.allclose(data1, data2)

# TODO: Fix NWB round-trip tests
# @pytest.mark.parametrize(
#     "data_folder",
#     [
#         ("1plane1chan1500"),
#         ("2plane2chan1500"),
#         ("bruker"),
#     ],
# )
# def test_nwb_round_trip(replace_settings_save_path_with_local_path, data_folder):

#     # Get expected data already saved as NumPy files
#     (
#         save_folder,
#         expected_iscell,
#         expected_F,
#         expected_Fneu,
#         expected_spks,
#     ) = replace_settings_save_path_with_local_path

#     # Save as NWB file
#     save_nwb(save_folder)

#     # Check (some of) the structure of the NWB file saved
#     nwb_path = save_folder.joinpath("ophys.nwb")
#     assert nwb_path.exists()
#     with NWBHDF5IO(str(nwb_path), "r") as io:
#         read_nwbfile = io.read()
#         assert read_nwbfile.processing
#         ophys = read_nwbfile.processing["ophys"]
#         assert ophys.data_interfaces["Deconvolved"]
#         Fluorescence = ophys.data_interfaces["Fluorescence"]
#         assert Fluorescence
#         if "2plane" in data_folder:
#             assert Fluorescence["plane0"]
#             assert Fluorescence["plane1"]
#         assert ophys.data_interfaces["Neuropil"]
#         if "2chan" in data_folder:
#             Fluorescence_chan2 = ophys.data_interfaces["Fluorescence_chan2"]
#             assert Fluorescence_chan2
#             assert ophys.data_interfaces["Neuropil_chan2"]
#         iscell_nwb = (
#             ophys.data_interfaces["ImageSegmentation"]
#             .plane_segmentations["PlaneSegmentation"]
#             .columns[2]
#             .data[:]
#         )
#         np.testing.assert_array_equal(iscell_nwb, expected_iscell)

#     # Extract Suite2p info from NWB file
#     stat, settings, F, Fneu, spks, iscell, probcell, redcell, probredcell = read_nwb(
#         nwb_path
#     )

#     # Check we get the same data back as the original data
#     # after saving to NWB + reading
#     np.testing.assert_array_equal(F, expected_F)
#     np.testing.assert_array_equal(Fneu, expected_Fneu)
#     np.testing.assert_array_equal(spks, expected_spks)
#     np.testing.assert_array_equal(
#         np.transpose(np.array([iscell, probcell])), expected_iscell
#     )
#     # TODO: assert round trip for `stat` and `settings`
#     # Probably need to recreate the data files as some fields are missing in the dict
#     # expected_stat = np.load(save_folder.joinpath("plane0", "stat.npy"), allow_pickle=True)
#     # expected_settings = np.load(save_folder.joinpath("plane0", "settings.npy"), allow_pickle=True)
#     # np.testing.assert_equal(stat, expected_stat)
#     # np.testing.assert_equal(settings, expected_settings)

#     # Remove NWB file
#     nwb_path.unlink()


@pytest.mark.parametrize(
    "input_path, expected_path, success",
    [
        (
            "D:/kjkcc/jodendopn/suite2p/ncconoc/onowcno",
            "D:/kjkcc/jodendopn/suite2p",
            True,
        ),
        (
            "/home/bla/kjkcc/jodendopn/suite2p/settings.npy",
            "/home/bla/kjkcc/jodendopn/suite2p",
            True,
        ),
        ("/etc/bla/kjkcc/jodendopn/suite2p", "/etc/bla/kjkcc/jodendopn/suite2p", True),
        ("/etc/bla/kjkcc/jodendopn/", "", False),
    ],
)
def test_get_suite2p_path(input_path, expected_path, success):
    if success:
        res_path = get_suite2p_path(input_path)
        assert res_path == Path(expected_path)
    else:
        with pytest.raises(FileNotFoundError):
            get_suite2p_path(Path(input_path))
