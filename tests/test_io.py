"""
Tests for the Suite2p IO module
"""
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

@pytest.mark.parametrize(
    "data_folder",
    [
        ("1plane1chan1500"),
        ("2plane2chan1500"),
        #("bruker"),
    ],
)
def test_nwb_round_trip(data_folder):
    """Test saving Suite2p outputs to NWB and loading them back."""

    # Define the path to the suite2p folder
    save_folder = Path("data").joinpath("test_outputs", data_folder, "suite2p")
    print(save_folder)

    # Load expected data from plane folders
    plane_folders = natsorted([
        f for f in save_folder.iterdir()
        if f.is_dir() and f.name.startswith("plane")
    ])

    # Concatenate data across planes
    expected_F = np.concatenate([
        np.load(plane_dir.joinpath("F.npy"), allow_pickle=True)
        for plane_dir in plane_folders
    ], axis=0)

    expected_Fneu = np.concatenate([
        np.load(plane_dir.joinpath("Fneu.npy"), allow_pickle=True)
        for plane_dir in plane_folders
    ], axis=0)

    expected_spks = np.concatenate([
        np.load(plane_dir.joinpath("spks.npy"), allow_pickle=True)
        for plane_dir in plane_folders
    ], axis=0)

    expected_iscell = np.concatenate([
        np.load(plane_dir.joinpath("iscell.npy"), allow_pickle=True)
        for plane_dir in plane_folders
    ], axis=0)

    expected_stat = np.concatenate([
        np.load(plane_dir.joinpath("stat.npy"), allow_pickle=True)
        for plane_dir in plane_folders
    ], axis=0)

    
    # Save as NWB file
    save_nwb(save_folder)

    # Check (some of) the structure of the NWB file saved
    nwb_path = save_folder.joinpath("ophys.nwb")
    assert nwb_path.exists()
    with NWBHDF5IO(str(nwb_path), "r") as io:
        read_nwbfile = io.read()
        assert read_nwbfile.processing
        ophys = read_nwbfile.processing["ophys"]
        assert ophys.data_interfaces["Deconvolved"]
        Fluorescence = ophys.data_interfaces["Fluorescence"]
        assert Fluorescence
        if "2plane" in data_folder:
            assert Fluorescence["plane0"]
            assert Fluorescence["plane1"]
        assert ophys.data_interfaces["Neuropil"]
        if "2chan" in data_folder:
            Fluorescence_chan2 = ophys.data_interfaces["Fluorescence_chan2"]
            assert Fluorescence_chan2
            assert ophys.data_interfaces["Neuropil_chan2"]
        iscell_nwb = (
            ophys.data_interfaces["ImageSegmentation"]
            .plane_segmentations["PlaneSegmentation"]
            .columns[2]
            .data[:]
        )
        np.testing.assert_array_equal(iscell_nwb, expected_iscell)

    # Extract Suite2p info from NWB file
    stat, settings, F, Fneu, spks, iscell, probcell, redcell, probredcell = read_nwb(
        nwb_path
    )

    # Check we get the same data back as the original data
    # after saving to NWB + reading
    np.testing.assert_array_equal(F, expected_F)
    np.testing.assert_array_equal(Fneu, expected_Fneu)
    np.testing.assert_array_equal(spks, expected_spks)
    np.testing.assert_array_equal(iscell, expected_iscell)

    # Check stat round trip - compare key fields that are preserved
    assert len(stat) == len(expected_stat), "Number of ROIs mismatch"

    # For multiplane data, coordinates are adjusted to composite coordinate system
    # so we can't directly compare with the original plane-local coordinates
    if "2plane" not in data_folder:
        for i in range(len(stat)):
            np.testing.assert_array_equal(stat[i]['ypix'], expected_stat[i]['ypix'])
            np.testing.assert_array_equal(stat[i]['xpix'], expected_stat[i]['xpix'])
            np.testing.assert_allclose(stat[i]['lam'], expected_stat[i]['lam'], rtol=1e-5)

    # # Check settings - compare key fields that are preserved
    # # For multiplane data, dimensions are for the composite image
    # if "2plane" not in data_folder:
    #     assert settings['Ly'] == expected_settings['Ly']
    #     assert settings['Lx'] == expected_settings['Lx']
    #     np.testing.assert_array_equal(settings['meanImg'], expected_settings['meanImg'])
    # else:
    #     # For multiplane, just check that dimensions are reasonable
    #     assert settings['Ly'] > 0
    #     assert settings['Lx'] > 0

    # Remove NWB file
    nwb_path.unlink()


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
