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


@pytest.fixture()
def binfile1500(test_ops):
    test_ops["tiff_list"] = ["input_1500.tif"]
    op = io.tiff_to_binary(test_ops)
    bin_filename = str(Path(op["save_path0"]).joinpath("suite2p/plane0/data.bin"))
    with io.BinaryFile(
        Ly=op["Ly"], Lx=op["Lx"], filename=bin_filename
    ) as bin_file:
        yield bin_file


@pytest.fixture(scope="function")
def replace_ops_save_path_with_local_path(request):
    """
    This fixture replaces the `save_path` variable in the `ops.npy` file
    by its local path version
    """

    # Workaround to load pickled NPY files on Windows containing
    # `PosixPath` objects
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath

    # Get the `data_folder` variable from the running test name    
    data_folder = re.search(r"\[(.*?)(-.*?)?\]", request.node.name).group(1)
    save_folder = Path("data").joinpath("test_outputs", data_folder, "suite2p")

    save_path = {}
    plane_folders = [
        dir
        for dir in natsorted(save_folder.iterdir())
        if dir.is_dir() and "plane" in dir.name
    ]
    for plane_idx, plane_dir in enumerate(plane_folders):

        # Temporarily change the `save_folder` variable in the NumPy file
        ops1 = np.load(plane_dir.joinpath("ops.npy"), allow_pickle=True)
        save_path[plane_dir] = ops1.item(0)["save_path"]
        ops1.item(0)["save_path"] = str(plane_dir.absolute())
        np.save(plane_dir.joinpath("ops.npy"), ops1)

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
    for plane_dir in plane_folders:
        # Undo the changes made in the NumPy file
        ops1 = np.load(plane_dir.joinpath("ops.npy"), allow_pickle=True)
        ops1.item(0)["save_path"] = save_path[plane_dir]
        np.save(plane_dir.joinpath("ops.npy"), ops1)


def test_h5_to_binary_produces_nonnegative_output_data(test_ops):
    test_ops["h5py"] = Path(test_ops["data_path"][0]).joinpath("input.h5")
    test_ops["nplanes"] = 3
    test_ops["nchannels"] = 2
    test_ops["data_path"] = []
    op = io.h5py_to_binary(test_ops)
    output_data = io.BinaryFile(
        filename=Path(op["save_path0"], "suite2p/plane0/data.bin"),
        Ly=op["Ly"],
        Lx=op["Lx"],
    ).data
    assert np.all(output_data >= 0)


def test_that_bin_movie_without_badframes_results_in_a_same_size_array(binfile1500):
    mov = binfile1500.bin_movie(bin_size=1)
    assert mov.shape == (1500, binfile1500.Ly, binfile1500.Lx)


def test_that_bin_movie_with_badframes_results_in_a_smaller_array(binfile1500):

    np.random.seed(42)
    bad_frames = np.random.randint(2, size=binfile1500.n_frames, dtype=bool)
    mov = binfile1500.bin_movie(bin_size=1, bad_frames=bad_frames, reject_threshold=0)

    assert len(mov) < binfile1500.n_frames, "bin_movie didn't produce a smaller array."
    assert len(mov) == len(bad_frames) - sum(
        bad_frames
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
        ("bruker"),
    ],
)
def test_nwb_round_trip(replace_ops_save_path_with_local_path, data_folder):

    # Get expected data already saved as NumPy files
    (
        save_folder,
        expected_iscell,
        expected_F,
        expected_Fneu,
        expected_spks,
    ) = replace_ops_save_path_with_local_path

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
    stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell = read_nwb(
        nwb_path
    )

    # Check we get the same data back as the original data
    # after saving to NWB + reading
    np.testing.assert_array_equal(F, expected_F)
    np.testing.assert_array_equal(Fneu, expected_Fneu)
    np.testing.assert_array_equal(spks, expected_spks)
    np.testing.assert_array_equal(
        np.transpose(np.array([iscell, probcell])), expected_iscell
    )
    # TODO: assert round trip for `stat` and `ops`
    # Probably need to recreate the data files as some fields are missing in the dict
    # expected_stat = np.load(save_folder.joinpath("plane0", "stat.npy"), allow_pickle=True)
    # expected_ops = np.load(save_folder.joinpath("plane0", "ops.npy"), allow_pickle=True)
    # np.testing.assert_equal(stat, expected_stat)
    # np.testing.assert_equal(ops, expected_ops)

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
            "/home/bla/kjkcc/jodendopn/suite2p/ops.npy",
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
