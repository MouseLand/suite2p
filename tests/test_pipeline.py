"""
Class that tests common use cases for pipeline.
"""

import suite2p
import utils


def get_outputs_to_check(n_channels):
    outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
    if n_channels == 2:
        outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
    return outputs_to_check


def test_2plane_2chan(default_ops):
    """
    Tests for case with 2 planes and 2 channels.
    """
    default_ops['nplanes'] = 2
    default_ops['nchannels'] = 2
    suite2p.run_s2p(ops=default_ops)
    utils.check_output(
        default_ops['save_path0'],
        get_outputs_to_check(default_ops['nchannels']),
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
    )
