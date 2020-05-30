import numpy as np
import filecmp
from pathlib import Path
from tifffile import imread
from suite2p import io


class TestSuite2pIoModule:
    """
    Tests for the Suite2p IO module
    """
    def test_tiff_and_binary_same_content(self, setup_and_teardown, get_test_dir_path):
        """
        Tests if the numpy array made from io.tiff_to_binary's binary file is identical to the numpy array
        made from the input TIF.
        """
        ops, tmp_dir = setup_and_teardown
        op = io.tiff_to_binary(ops)[0]
        # Read in binary file's contents as uint16 np array
        binary_file_data = np.fromfile(
            str(Path(tmp_dir).joinpath('suite2p', 'plane0', 'data.bin')),
            np.uint16
        )
        output_data = np.reshape(binary_file_data, (-1, op['Ly'], op['Lx']))
        # Convert input TIF to numpy array
        input_data = imread(str(Path(get_test_dir_path).joinpath('input.tif')))
        assert np.array_equal(output_data, input_data)
