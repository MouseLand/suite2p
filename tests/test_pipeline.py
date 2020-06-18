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


def test_1plane_1chan_with_batches_and_metrics(default_ops):
    """
    Tests for case with 1 plane and 1 channel with multiple batches.
    """
    default_ops['tiff_list'] = ['input1500.tif']
    default_ops['do_regmetrics'] = True
    suite2p.run_s2p(ops=default_ops)
    utils.check_output(
        default_ops['save_path0'],
        get_outputs_to_check(default_ops['nchannels']),
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
        added_tag='1500'
    )


def test_2plane_2chan(default_ops):
    """
    Tests for case with 2 planes and 2 channels.
    """
    default_ops['nplanes'] = 2
    default_ops['nchannels'] = 2
    default_ops['tiff_list'] = ['input.tif']
    suite2p.run_s2p(ops=default_ops)
    utils.check_output(
        default_ops['save_path0'],
        get_outputs_to_check(default_ops['nchannels']),
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
    )


def test_1plane_2chan_sourcery(default_ops):
    """
    Tests for case with 1 plane and 2 channel.
    """
    default_ops['nchannels'] = 2
    default_ops['sparse_mode'] = 0
    default_ops['tiff_list'] = ['input.tif']
    suite2p.run_s2p(ops=default_ops)
    utils.check_output(
        default_ops['save_path0'],
        get_outputs_to_check(default_ops['nchannels']),
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
    )