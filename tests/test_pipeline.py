import numpy as np
from pathlib import Path
import suite2p


def get_outputs_to_check(n_channels):
    outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
    if n_channels == 2:
        outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
    return outputs_to_check


class TestCommonPipelineUseCases:
    """
    Class that tests common use cases for pipeline.
    """

    def test_1plane_1chan(self, setup_and_teardown, get_test_dir_path, test_utils):
        """
        Tests for case with 1 plane and 1 channel
        """
        # Get ops and unique tmp_dir from fixture
        ops, tmp_dir = setup_and_teardown
        suite2p.run_s2p(ops=ops)
        # Check outputs against ground_truth files
        test_utils.check_output(
            tmp_dir, get_outputs_to_check(ops['nchannels']), get_test_dir_path, ops['nplanes'], ops['nchannels']
        )

    def test_2plane_1chan(self, setup_and_teardown, get_test_dir_path, test_utils):
        """
        Tests for case with 2 planes and 1 channel.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        suite2p.run_s2p(ops=ops)
        test_utils.check_output(
            tmp_dir, get_outputs_to_check(ops['nchannels']), get_test_dir_path, ops['nplanes'], ops['nchannels']
        )

    def test_2plane_2chan(self, setup_and_teardown, get_test_dir_path, test_utils):
        """
        Tests for case with 2 planes and 2 channels.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        ops['nchannels'] = 2
        suite2p.run_s2p(ops=ops)
        test_utils.check_output(
            tmp_dir, get_outputs_to_check(ops['nchannels']), get_test_dir_path, ops['nplanes'], ops['nchannels']
        )
