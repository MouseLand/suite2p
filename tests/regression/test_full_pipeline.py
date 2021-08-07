"""
Class that tests common use cases for pipeline.
"""

from suite2p import io
from pathlib import Path
import numpy as np
import suite2p, utils, json


def test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format(test_ops):
    """
    Tests for case with 1 plane and 1 channel with multiple batches. Results are saved to nwb format
    then checked to see if it contains the necessary parts for use with GUI.
    """
    test_ops.update({
        'tiff_list': ['input_1500.tif'],
        'do_regmetrics': True,
        'save_NWB': True,
        'save_mat': True
    })
    suite2p.run_s2p(ops=test_ops)

    nplanes = test_ops['nplanes']
    outputs_to_check = ['F', 'stat']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan1500/suite2p/plane{i}")),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))


    # Read Nwb data and make sure it's identical to output data
    stat, ops, F, Fneu, spks, iscell, probcell, redcell, probredcell = \
        io.read_nwb(str(Path(test_ops['save_path0']).joinpath('suite2p/ophys.nwb')))
    output_dir = Path(test_ops['save_path0']).joinpath(f"suite2p/plane0")
    output_name_list = ['F', 'stat']
    data_list_one = utils.get_list_of_data(output_name_list, output_dir)
    data_list_two = [F, stat]
    for output, data1, data2 in zip(output_name_list, data_list_one, data_list_two):
        if output == 'stat':  # where the elements of npy arrays are dictionaries (e.g: stat.npy)
            for gt_dict, output_dict in zip(data1, data2):
                for k in gt_dict.keys():
                    if k=='ypix' or k=='xpix':  # todo: these both are different from the original; footprint and overlap are different, std key doesn't exist in output_dict.
                        assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)
        elif output == 'iscell':  # just check the first column; are cells/noncells classified the same way?
            assert np.array_equal(data1[:, 0], data2[:, 0])
        else:
            assert np.allclose(data1, data2, rtol=1e-4, atol=5e-2)


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
            'save_mat': True
        })
        nplanes = ops['nplanes']
        suite2p.run_s2p(ops=ops)

        outputs_to_check = ['F', 'iscell', 'stat']
        #if ops['nchannels'] == 2:
        #    outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
        for i in range(nplanes):
            assert all(utils.compare_list_of_outputs(
                outputs_to_check,
                utils.get_list_of_data(outputs_to_check, ops['data_path'][0].joinpath(f"{nplanes}plane{ops['nchannels']}chan1500/suite2p/plane{i}")),
                utils.get_list_of_data(outputs_to_check, Path(ops['save_path0']).joinpath(f"suite2p/plane{i}")),
            ))


def temp_test_1plane_2chan_sourcery(test_ops):
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
    outputs_to_check = ['F', 'iscell', 'stat']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan/suite2p/plane{i}")),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
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

    nplanes = test_ops['nplanes'] * test_ops['nrois']
    outputs_to_check = ['F', 'iscell', 'stat']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, test_ops['data_path'][0].joinpath(f'suite2p/plane{i}')),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))