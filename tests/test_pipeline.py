import pytest
import shutil
import filecmp
import os
from suite2p import run_s2p, default_ops

# TODO: store as env variables?
root_path = '/Users/chriski/Desktop/suite2p_ws'
gt_path = root_path + '/chris_output'  # Ground Truth path
img_path = root_path + '/images_ex'  # Input Images path


@pytest.fixture()
def setup_and_teardown(tmpdir):
    """
    Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for
    each test. Then, removes temporary directory after test is completed.
    """
    ops = default_ops()
    ops['data_path'] = [img_path]
    ops['save_path0'] = str(tmpdir)
    yield ops, str(tmpdir)
    if os.path.isdir(tmpdir):
        shutil.rmtree(tmpdir)
        print('Successful removal of tmp_path {}.'.format(tmpdir))


class TestCommonPipelineUseCases:
    """
    Class that tests common use cases for pipeline.
    """

    def check_output_gt(self, output_root, gt_root, nplanes: int, nchannels: int):
        """
        Helper function to check if outputs given by a test are exactly the same
        as the ground truth outputs.
        """
        gt_dir = os.path.join(str(gt_root), "{}plane{}chan".format(nplanes, nchannels))
        output_dir = os.path.join(str(output_root), "suite2p")
        outputs_to_check = ['F.npy', 'Fneu.npy', 'iscell.npy', 'spks.npy', 'stat.npy', 'data.bin']
        # Check channel2 outputs if necessary
        if nchannels == 2:
            outputs_to_check.extend(['F_chan2.npy', 'Fneu_chan2.npy', 'data_chan2.bin'])
        # Go through each plane folder and compare contents
        for i in range(nplanes):
            for output in outputs_to_check:
                # Non-shallow comparison of both files to make sure their contents are identical
                assert filecmp.cmp(
                    os.path.join(gt_dir, 'plane{}'.format(i), output),
                    os.path.join(output_dir, 'plane{}'.format(i), output),
                    shallow=False
                )

    def test_1plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 1 plane and 1 channel.
        """
        # Get ops and unique tmp_dir from fixture
        ops, tmp_dir = setup_and_teardown
        run_s2p.run_s2p(ops=ops)
        # Check outputs against ground_truth files
        self.check_output_gt(tmp_dir, gt_path, ops['nplanes'], ops['nchannels'])

    def test_2plane_1chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 1 channel.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        run_s2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, gt_path, ops['nplanes'], ops['nchannels'])

    def test_2plane_2chan(self, setup_and_teardown):
        """
        Tests for case with 2 planes and 2 channels.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        ops['nchannels'] = 2
        run_s2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, gt_path, ops['nplanes'], ops['nchannels'])