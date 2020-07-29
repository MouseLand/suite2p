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


def test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format(test_ops):
    """
    Tests for case with 1 plane and 1 channel with multiple batches. Results are saved to nwb format
    then checked to see if it contains the necessary parts for use with GUI.
    """
    test_ops.update({
        'tiff_list': ['input_1500.tif'],
        'do_regmetrics': True,
        'save_NWB': True,
    })
    suite2p.run_s2p(ops=test_ops)

    outputs_to_check = ['F', 'Fneu', 'spks', 'iscell']
    assert all(utils.check_output(
        test_ops['save_path0'],
        outputs_to_check,
        test_ops['data_path'][0],
        test_ops['nplanes'],
        test_ops['nchannels'],
        added_tag='1500'
    ))

    # Read Nwb data and make sure it's identical to output data
    stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell = \
        io.read_nwb(str(Path(test_ops['save_path0']).joinpath('suite2p/ophys.nwb')))
    assert all(utils.compare_list_of_outputs(
        0,
        outputs_to_check,
        utils.get_list_of_output_data(outputs_to_check, test_ops['save_path0'], 0),
        [F, Fneu, spks, np.stack([iscell.astype(np.float32), probcell.astype(np.float32)]).T],
    ))

def test_2plane_2chan_with_batches(test_ops):
    """
    Tests for case with 2 planes and 2 channels with multiple batches.
    """
    test_ops.update({
        'tiff_list': ['input_1500.tif'],
        'batch_size': 200,
        'nplanes': 2,
        'nchannels': 2,
        'reg_tif': True,
        'reg_tif_chan2': True,
    })
    suite2p.run_s2p(ops=test_ops)
    assert all(utils.check_output(
        test_ops['save_path0'],
        get_outputs_to_check(test_ops['nchannels']) + ['reg_tif', 'reg_tif_chan2'],
        test_ops['data_path'][0],
        test_ops['nplanes'],
        test_ops['nchannels'],
        added_tag='1500'
    ))


def test_1plane_2chan_sourcery(test_ops):
    """
    Tests for case with 1 plane and 2 channel.
    """
    test_ops.update({
        'nchannels': 2,
        'sparse_mode': 0,
        'tiff_list': ['input.tif'],
        'keep_movie_raw': True
    })
    suite2p.run_s2p(ops=test_ops)
    assert all(utils.check_output(
        test_ops['save_path0'],
        get_outputs_to_check(test_ops['nchannels']),
        test_ops['data_path'][0],
        test_ops['nplanes'],
        test_ops['nchannels'],
    ))