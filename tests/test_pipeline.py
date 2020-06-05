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
