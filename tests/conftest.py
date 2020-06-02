import pytest
import shutil
import suite2p
import numpy as np

from pathlib import Path


class TestUtils:
    """
    Class of Test Utility functions that can be accessed in tests via the test_utils fixture below.
    """

    @staticmethod
    def create_plane_dir(op, plane):
        suite_dir = Path(op['save_path0']).joinpath('suite2p')
        suite_dir.mkdir(exist_ok=True)
        plane_dir = Path(op['save_path0']).joinpath('suite2p').joinpath('plane{}'.format(plane))
        plane_dir.mkdir(exist_ok=True)
        return plane_dir

    @staticmethod
    def write_data_to_binary(binary_path, data_path):
        input_data = np.load(data_path)
        with open(binary_path, 'wb') as f:
            input_data.tofile(f)
        return binary_path

    @staticmethod
    def check_output(output_root, outputs_to_check, test_data_dir, nplanes: int, nchannels: int):
        """
        Helper function to check if outputs given by a test are exactly the same
        as the ground truth outputs.
        """
        output_dir = Path(output_root).joinpath("suite2p")
        for i in range(nplanes):
            for output in outputs_to_check:
                test_data = np.load(
                    str(test_data_dir.joinpath('{}plane{}chan'.format(nplanes, nchannels), 'suite2p',
                                               'plane{}'.format(i), "{}.npy".format(output))), allow_pickle=True
                )
                output_data = np.load(
                    str(output_dir.joinpath('plane{}'.format(i), "{}.npy".format(output))), allow_pickle=True
                )
                print("Comparing {} for plane {}".format(output, i))
                rtol, atol = 1e-6 , 5e-2
                # Handle cases where the elements of npy arrays are dictionaries (e.g: stat.npy)
                if output == 'stat':
                    for gt_dict, output_dict in zip(test_data, output_data):
                        for k in gt_dict.keys():
                            assert np.allclose(gt_dict[k], output_dict[k], rtol=rtol, atol=atol)
                else:
                    assert np.allclose(test_data, output_data, rtol=rtol, atol=atol)


@pytest.fixture
def test_utils():
    return TestUtils


@pytest.fixture()
def get_test_dir_path():
    return Path(__file__).parent.parent.joinpath('data/test_data')


@pytest.fixture()
def setup_and_teardown(tmpdir, get_test_dir_path):
    """
    Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for
    each test. Then, removes temporary directory after test is completed.
    """
    ops = suite2p.default_ops()
    ops['data_path'] = [get_test_dir_path]
    ops['save_path0'] = str(tmpdir)
    yield ops, str(tmpdir)
    tmpdir_path = Path(str(tmpdir))
    if tmpdir_path.is_dir():
        shutil.rmtree(tmpdir)
        print('Successful removal of tmp_path {}.'.format(tmpdir))

@pytest.fixture
def helpers():
    return Helpers