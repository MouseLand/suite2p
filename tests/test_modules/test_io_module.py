import numpy as np
import filecmp
from pathlib import Path
from tifffile import imread
from suite2p import io


class TestSuite2pIoModule:
    """
    Tests for the Suite2p IO module
    """
    def test_tiff_reconstruction_from_binary_file(self, setup_and_teardown, get_test_dir_path):
        """
        Tests that reconstructed TIF file from write_tiff and tiff_to_binary has same contents
        as input TIF.
        """
        ops, tmp_dir = setup_and_teardown
        op = io.tiff_to_binary(ops)[0]
        # Read in binary file's contents as uint16 np array
        binary_file_data = np.fromfile(
            str(Path(tmp_dir).joinpath('suite2p', 'plane0', 'data.bin')),
            np.uint16
        )
        output_data = np.reshape(binary_file_data, (-1, op['Ly'], op['Lx']))
        io.write_tiff(output_data, op, 0, True)
        # Convert input TIF and output Tiff to numpy array and compare
        input_data = imread(str(Path(get_test_dir_path).joinpath('input.tif')))
        reconstructed_tiff_data = imread(
            str(Path(tmp_dir).joinpath('suite2p', 'plane0', 'reg_tif', 'file000_chan0.tif'))
        )
        assert np.array_equal(reconstructed_tiff_data, input_data)

    def test_h5_to_binary_input_tiff_same_content(self, setup_and_teardown, get_test_dir_path):
        """
        Tests if the h5
        """
        pass