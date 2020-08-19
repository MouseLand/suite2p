"""
Class that tests common use cases for pipeline.
"""

from suite2p import io
from pathlib import Path
import numpy as np
import suite2p, utils, json


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

    nplanes = test_ops['nplanes']
    assert all(utils.check_output(
        output_root=test_ops['save_path0'],
        outputs_to_check=get_outputs_to_check(test_ops['nchannels']),
        test_data_dir=test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan1500/suite2p/"),
        nplanes=nplanes,
    ))
    # Read Nwb data and make sure it's identical to output data
    stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell = \
        io.read_nwb(str(Path(test_ops['save_path0']).joinpath('suite2p/ophys.nwb')))
    assert all(utils.compare_list_of_outputs(
        0,
        get_outputs_to_check(test_ops['nchannels']),
        utils.get_list_of_output_data(get_outputs_to_check(test_ops['nchannels']), test_ops['save_path0'], 0),
        [F, Fneu, np.stack([iscell.astype(np.float32), probcell.astype(np.float32)]).T, spks, stat],
    ))


def test_2plane_2chan_with_batches(test_ops):
    """
    Tests for case with 2 planes and 2 channels with multiple batches.  Runs twice to check for consistency.
    """
    for _ in range(2):
        ops = test_ops.copy()
        ops.update({
            'tiff_list': ['input_1500.tif'],
            'batch_size': 200,
            'nplanes': 2,
            'nchannels': 2,
            'reg_tif': True,
            'reg_tif_chan2': True,
        })
        nplanes = ops['nplanes']
        suite2p.run_s2p(ops=ops)
        assert all(utils.check_output(
            output_root=ops['save_path0'],
            outputs_to_check=get_outputs_to_check(ops['nchannels']) + ['reg_tif', 'reg_tif_chan2'],
            test_data_dir=ops['data_path'][0].joinpath(f"{nplanes}plane{ops['nchannels']}chan1500/suite2p/"),
            nplanes=nplanes,
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
    nplanes = test_ops['nplanes']
    assert all(utils.check_output(
        output_root=test_ops['save_path0'],
        outputs_to_check=get_outputs_to_check(test_ops['nchannels']),
        test_data_dir=test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan/suite2p/"),
        nplanes=nplanes,
    ))


def test_mesoscan_2plane_2z(test_ops):
    """
    Tests for case with 2 planes and 2 ROIs for a mesoscan.
    """
    with open('data/test_data/mesoscan/ops.json') as f:
        meso_ops = json.load(f)
    test_ops['data_path'] = [Path(test_ops['data_path'][0]).joinpath('mesoscan')]
    for key in meso_ops.keys():
        if key not in ['data_path', 'save_path0', 'do_registration', 'roidetect']:
            test_ops[key] = meso_ops[key]
    test_ops['delete_bin'] = False
    suite2p.run_s2p(ops=test_ops)
    
    assert all(utils.check_output(
        output_root=test_ops['save_path0'],
        outputs_to_check=get_outputs_to_check(test_ops['nchannels']),
        test_data_dir=test_ops['data_path'][0].joinpath('suite2p'),
        nplanes=test_ops['nplanes']*test_ops['nrois'],
    ))