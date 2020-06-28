"""
Class that tests common use cases for pipeline.
"""

from suite2p import io
from pathlib import Path
import numpy as np
import suite2p
import utils


def get_outputs_to_check(n_channels):
    outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
    if n_channels == 2:
        outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
    return outputs_to_check


def test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format(default_ops):
    """
    Tests for case with 1 plane and 1 channel with multiple batches. Results are saved to nwb format
    then checked to see if it contains the necessary parts for use with GUI.
    """
    default_ops['tiff_list'] = ['input_1500.tif']
    default_ops['do_regmetrics'] = True
    default_ops['save_NWB'] = True
    outputs_to_check = ['F', 'Fneu', 'spks', 'iscell']
    suite2p.run_s2p(ops=default_ops)
    utils.check_output(
        default_ops['save_path0'],
        outputs_to_check,
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
        added_tag='1500'
    )
    # Read Nwb data and make sure it's identical to output data
    stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell = \
        io.read_nwb(str(Path(default_ops['save_path0']).joinpath('suite2p').joinpath('ophys.nwb')))
    utils.compare_list_of_outputs(
        0,
        outputs_to_check,
        utils.get_list_of_output_data(outputs_to_check, default_ops['save_path0'], 0),
        [F, Fneu, spks,
         np.concatenate([iscell.astype(np.float32)[:,np.newaxis], probcell.astype(np.float32)[:,np.newaxis]], axis=1)
        ]
    )


def test_2plane_2chan_with_batches(default_ops):
    """
    Tests for case with 2 planes and 2 channels with multiple batches.
    """
    default_ops['tiff_list'] = ['input_1500.tif']
    default_ops['batch_size'] = 200
    default_ops['nplanes'] = 2
    default_ops['nchannels'] = 2
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