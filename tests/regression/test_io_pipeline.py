

import numpy as np
from pathlib import Path
from tifffile import imread
from suite2p import io

from utils import get_binary_file_data


def test_tiff_reconstruction_from_binary_file(test_ops):
    """
    Tests to see if tif generated by tiff_to_binary and write_tiff matches test tif.
    """
    test_ops['tiff_list'] = ['input.tif']
    op = io.tiff_to_binary(test_ops)
    output_data = get_binary_file_data(op)
    # Make sure data in matrix is nonnegative
    assert np.all(output_data >= 0)
    fname = io.generate_tiff_filename(
        functional_chan=op['functional_chan'],
        align_by_chan=op['align_by_chan'],
        save_path=op['save_path'],
        k=0,
        ichan=True
    )
    io.save_tiff(output_data, fname=fname)
    reconstructed_tiff_data = imread(
        str(Path(test_ops['save_path0']).joinpath('suite2p/plane0/reg_tif/file000_chan0.tif'))
    )
    # Compare to our test data
    prior_data = imread(
        str(Path(test_ops['data_path'][0]).joinpath('1plane1chan/suite2p/test_write_tiff.tif'))
    )
    assert np.array_equal(reconstructed_tiff_data, prior_data)

