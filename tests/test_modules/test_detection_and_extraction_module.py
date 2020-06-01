import numpy as np
from pathlib import Path
from suite2p import extraction


def prepare_for_detection(op, input_file_name, dimensions):
    """
    Prepares for detection by filling out necessary ops parameters. Removes dependence on
    other modules. Creates pre_registered binary file.
    """
    op['Lx'] = dimensions[0]
    op['Ly'] = dimensions[1]
    input_data = np.fromfile(str(input_file_name), np.int16)
    bin_file_path = str(Path(op['save_path0']).joinpath('data.bin'))
    with open(bin_file_path, 'wb') as f:
        f.write(input_data)
    op['reg_file'] = bin_file_path
    op['save_path'] = op['save_path0']
    return op


class TestSuite2pDetectionExtractionModule:
    """
    Tests for the Suite2p Detection and Extraction module.
    """
    # TODO: Separate once the detection module is created.

    def test_detection_extraction_output_1plane1chan(self, setup_and_teardown, get_test_dir_path):
        op, tmp_dir = setup_and_teardown
        op = prepare_for_detection(
            op, op['data_path'][0].joinpath('detection', 'pre_registered.npy'), (404, 360)
        )
