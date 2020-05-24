import pytest
import shutil
import numpy as np
import sys
from pathlib import Path
from suite2p import run_s2p
from typing import Tuple, Iterable
from suite2p.utils import download_url_to_file


# Paths to .suite2p dir and .suite2p test_data dir
suite_dir = Path.home().joinpath('.suite2p')
test_data_dir = suite_dir.joinpath('test_data')


def get_test_data_filenames(tups_of_plane_chans: Iterable[Tuple[int, int]]):
    """
    Returns list of test_data filenames associated with given list of tuples of num_planes and num_channels.

    Parameters
    ----------
    tups_of_plane_chans: list
        each element is a tuple corresponding to one set of test data. First element is number of planes and
        second element is number of channels.

    Returns
    -------

    all_files: list
        one list that contains all requested test_data filenames.

    """
    all_files = []
    for num_planes, num_chans in tups_of_plane_chans:
        outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
        if num_chans == 2:
            outputs_to_check.extend(['F_chan2', 'Fneu_chan2','redcell'])
        for i in range(num_planes):
            for out in outputs_to_check:
                file_name = "{0}plane{1}chan_{2}_plane{3}.npy".format(num_planes, num_chans, out, i)
                all_files.append(file_name)
    return all_files


def download_test_data():
    """
    Downloads input_data and all ground_truth/test data (1 plane 1 channel, 2 planes 1 channel,
    and 2 planes 2 channel)

    Returns
    -------

    """
    # Make test_data dir if doesn't exist
    suite_dir.mkdir(exist_ok=True)
    test_data_dir.mkdir(exist_ok=True)
    td_root_url = 'http://www.suite2p.org/test_data/'
    files_to_get = ['input0.tif', 'input1.tif']
    # Get test_data for 1 plane 1 channel, 2 planes 1 channel, and 2 planes 2 channel
    files_to_get.extend(get_test_data_filenames([(1, 1), (2, 1), (2, 2)]))
    for fn in files_to_get:
        data_url = td_root_url + fn
        cached_file = test_data_dir.joinpath(fn)
        if not cached_file.exists():
            sys.stderr.write('Downloading: "{}" to {}\n'.format(data_url, cached_file))
            download_url_to_file(data_url, str(cached_file), progress=True)


def compare_npy_arrays(arr1, arr2, atol):
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
                assert np.allclose(gt_curr_dict[k], output_curr_dict[k], rtol=0, atol=atol)
    # Assume all other cases have arrays that contain numbers
    else:
        assert np.allclose(arr1, arr2, rtol=0, atol=atol)


@pytest.fixture()
def setup_and_teardown(tmpdir):
    """
    Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for
    each test. Then, removes temporary directory after test is completed.
    """
    ops = run_s2p.default_ops()
    download_test_data()
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
                compare_npy_arrays(test_data, output_data, 0.008)

    def test_1plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 1 plane and 1 channel.
        """
        # Get ops and unique tmp_dir from fixture
        ops, tmp_dir = setup_and_teardown
        run_s2p.run_s2p(ops=ops)
        # Check outputs against ground_truth files
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])

    def test_2plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 1 channel.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        run_s2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])

    def test_2plane_2chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 2 channels.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        ops['nchannels'] = 2
        run_s2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, ops['nplanes'], ops['nchannels'])