import pytest
import shutil
import numpy as np
from pathlib import Path

import suite2p


# Paths to .suite2p dir and .suite2p test_data dir
suite_dir = Path(__file__).parent.parent.joinpath('data/test_data')
test_data_dir = suite_dir
print(test_data_dir)


def compare_npy_arrays(arr1, arr2, rtol, atol):
    """
    Compares contents of two numpy arrays. Checks that the elements are within an absolute tolerance of 0.008.

    Parameters
    ----------
    arr1
    arr2
    atol

    Returns
    -------

    """
    # Handle cases where the elements of npy arrays are dictionaries (e.g: stat.npy)
    if arr1.dtype == np.dtype('object') and (arr2.dtype == np.dtype('object')):
        # Iterate through dictionaries
        for i in range(len(arr1)):
            gt_curr_dict = arr1[i]
            output_curr_dict = arr2[i]
            # Iterate through keys and make sure values are same
            for k in gt_curr_dict.keys():
                assert np.allclose(gt_curr_dict[k], output_curr_dict[k], rtol=rtol, atol=atol)
    # Assume all other cases have arrays that contain numbers
    else:
        print("Max Absolute diff is : {}".format(np.max(np.abs(arr1-arr2))))
        print("Min rtol*(abs(arr2)) is : {}".format(np.min(rtol*np.abs(arr2))))
        assert np.allclose(arr1, arr2, rtol=rtol, atol=atol)


@pytest.fixture()
def setup_and_teardown(tmpdir):
    """
    Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for
    each test. Then, removes temporary directory after test is completed.
    """
    ops = suite2p.default_ops()
    ops['data_path'] = [test_data_dir]
    ops['save_path0'] = str(tmpdir)
    yield ops, str(tmpdir)
    tmpdir_path = Path(str(tmpdir))
    if tmpdir_path.is_dir():
        shutil.rmtree(tmpdir)
        print('Successful removal of tmp_path {}.'.format(tmpdir))
    
class TestCommonPipelineUseCases:
    """
    Class that tests common use cases for pipeline.
    """

    def check_output_gt(self, output_root, nplanes: int, nchannels: int):
        """
        Helper function to check if outputs given by a test are exactly the same
        as the ground truth outputs.
        """
        output_dir = Path(output_root).joinpath("suite2p")
        # Output has different structure from test_data structure
        outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
        # Check channel2 outputs if necessary
        if nchannels == 2:
            outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
        # Go through each plane folder and compare contents
        for i in range(nplanes):
            for output in outputs_to_check:
                test_data = np.load(
                    str(test_data_dir.joinpath('{0}plane{1}chan_{2}_plane{3}.npy'
                        .format(nplanes, nchannels, output, i))
                        ), allow_pickle=True
                )
                output_data = np.load(
                    str(output_dir.joinpath('plane{}'.format(i), "{}.npy".format(output))), allow_pickle=True
                )
                print("Comparing {} for plane {}".format(output, i))
                compare_npy_arrays(test_data, output_data, rtol=1e-6, atol=5e-2)

    def test_1plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 1 plane and 1 channel.
        """
        # Get ops and unique tmp_dir from fixture
        ops, tmp_dir = setup_and_teardown
        suite2p.run_s2p(ops=ops)
        # Check outputs against ground_truth files
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])

    def test_2plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 1 channel.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        suite2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])

    def test_2plane_2chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 2 channels.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        ops['nchannels'] = 2
        suite2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])
