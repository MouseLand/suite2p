import numpy as np
from pathlib import Path
from tifffile import imread
from suite2p import registration


def prepare_for_registration(op):
    """
    Prepares for registration by performing functions of io module. Fills out necessary ops parameters for
    registration module.
    """
    op['reg_tif'] = True
    op['Ly'] = 256
    op['Lx'] = 256
    # Make input data non-negative
    input_data = (imread(
        str(op['data_path'][0].joinpath('registration_data', 'rigid_registration_test_data.tif'))
    ) // 2).astype(np.int16)
    bin_file_path = str(Path(op['save_path0']).joinpath('data.bin'))
    # Generate binary file
    with open(bin_file_path, 'wb') as f:
        f.write(input_data)
    op['reg_file'] = bin_file_path
    op['save_path'] = op['save_path0']
    return op


class TestSuite2pRegistrationModule:
    """
    Tests for the Suite2p Registration Module
    """
    def test_register_binary_rigid_registration_only(self, setup_and_teardown):
        """
        Tests that register_binary works for a dataset that only has rigid shifts.
        """
        op, tmp_dir = setup_and_teardown
        op = prepare_for_registration(op)
        op = registration.register_binary(op)
        registered_data = imread(str(Path(op['save_path']).joinpath('reg_tif', 'file000_chan0.tif')))
        # Make sure registered_data is identical across frames
        check_data = np.repeat(registered_data[0, :, :][np.newaxis, :, :], 500, axis=0)
        assert np.array_equal(check_data, registered_data)

    def test_register_binary_nonrigid_registration_only(selfself, setup_and_teardown):
        pass