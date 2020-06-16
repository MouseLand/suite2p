"""
Tests for the Suite2p Registration Module
"""

import numpy as np
from pathlib import Path
from tifffile import imread

import suite2p.registration.pc
import suite2p.registration.register
from suite2p import registration


def prepare_for_registration(op, input_file_name, dimensions):
    """
    Prepares for registration by performing functions of io module. Fills out necessary ops parameters for
    registration module.
    """
    op['reg_tif'] = True
    op['Lx'] = dimensions[0]
    op['Ly'] = dimensions[1]
    nc = op['nchannels']
    # Make input data non-negative
    input_data = (imread(
        str(input_file_name)
    ) // 2).astype(np.int16)
    nframes_per_plane = input_data.shape[0] // op['nplanes']
    ops = []
    for i in range(op['nplanes']):
        # split image by number of planes
        plane_start = i * nframes_per_plane
        chan1_end = plane_start + nframes_per_plane//nc if nc == 2 else nframes_per_plane*(i+1)
        curr_op = op.copy()
        bin_file_path = str(Path(curr_op['save_path0']).joinpath('data_{}.bin'.format(i)))
        # Generate binary file for chan1
        with open(bin_file_path, 'wb') as f:
            f.write(input_data[plane_start:chan1_end, :, :])
        # Generate binary file for chan2
        if nc == 2:
            bin_file_path2 = str(Path(curr_op['save_path0']).joinpath('data_{}_2.bin'.format(i)))
            with open(bin_file_path2, 'wb') as f2:
                f2.write(input_data[chan1_end:nframes_per_plane*(i+1), :, :])
            curr_op['reg_file_chan2'] = bin_file_path2
        curr_op['reg_file'] = bin_file_path
        curr_op['save_path'] = op['save_path0']
        ops.append(curr_op)
    return ops


def check_registration_output(op, dimensions, input_path, reg_output_path_list, output_path_list):
    ops = prepare_for_registration(
        op, input_path, dimensions
    )
    reg_ops = []
    npl = op['nplanes']
    for i in range(npl):
        curr_op = registration.register_binary(ops[i])
        registered_data = imread(reg_output_path_list[i*npl])
        output_check = imread(output_path_list[i*npl])
        assert np.array_equal(registered_data, output_check)
        if op['nchannels'] == 2:
            registered_data = imread(reg_output_path_list[i * npl + 1])
            output_check = imread(output_path_list[i * npl + 1])
            assert np.array_equal(registered_data, output_check)
        reg_ops.append(curr_op)
    return reg_ops


def test_register_binary_output_with_metrics(default_ops):
    """
    Regression test that checks the output of register_binary given the `input.tif`.
    """
    default_ops['batch_size'] = 1500
    default_ops['do_regmetrics'] = True
    op = check_registration_output(
        default_ops, (256, 256),
        default_ops['data_path'][0].joinpath('registration', 'input_1500.tif'),
        [str(Path(default_ops['save_path0']).joinpath('reg_tif', 'file000_chan0.tif'))],
        [str(Path(default_ops['data_path'][0]).joinpath('registration', 'regression_output.tif'))]
    )
    registration.get_pc_metrics(op[0])


def test_register_binary_do_bidi_output(default_ops):
    """
    Regression test that checks the output of register_binary given the `input.tif` with the bidiphase,
    """
    default_ops['do_bidiphase'] = True
    check_registration_output(
        default_ops, (404, 360),
        default_ops['data_path'][0].joinpath('registration', 'bidi_shift_input.tif'),
        [str(Path(default_ops['save_path0']).joinpath('reg_tif', 'file000_chan0.tif'))],
        [str(Path(default_ops['data_path'][0]).joinpath('registration', 'regression_bidi_output.tif'))]
    )


def test_register_binary_rigid_registration_only(default_ops):
    """
    Tests that register_binary works for a dataset that only has rigid shifts.
    """
    default_ops['nonrigid'] = False
    op = prepare_for_registration(
        default_ops, default_ops['data_path'][0].joinpath('registration', 'rigid_registration_test_data.tif'), (256,256)
    )[0]
    op = registration.register_binary(op)
    registered_data = imread(str(Path(op['save_path']).joinpath('reg_tif', 'file000_chan0.tif')))
    # Make sure registered_data is identical across frames
    check_data = np.repeat(registered_data[0, :, :][np.newaxis, :, :], 500, axis=0)
    assert np.array_equal(check_data, registered_data)
    # Check and see if there are exactly 16 lines row-wise and column-wise
    num_row_lines = len(np.where(np.all(np.all(check_data == 1500, axis=0), axis=0))[0])
    num_col_lines = len(np.where(np.all(np.all(check_data == 1500, axis=0), axis=1))[0])
    assert num_col_lines == 16
    assert num_row_lines == 16


# def test_register_binary_smoothed_output(default_ops):
#     """
#     Regression test that checks the output of register_binary with smooth_sigma_time=1 and 2planes/2channels
#     given the `input.tif`.
#     """
#     default_ops['do_bidiphase'] = True
#     default_ops['smooth_sigma_time'] = 1
#     default_ops['nchannels'] = 1
#     default_ops['reg_tif'] = True
#
#     # to get tests to pass
#     check_registration_output(
#         default_ops, (404, 360),
#         default_ops['data_path'][0].joinpath('registration', 'bidi_shift_input.tif'),
#         [
#             str(Path(default_ops['save_path0']).joinpath('reg_tif', 'file000_chan0.tif')),
#             #str(Path(default_ops['save_path0']).joinpath('reg_tif_chan2', 'file000_chan1.tif'))
#         ],
#         [
#             str(Path(default_ops['data_path'][0]).joinpath('registration', 'regression_smoothed_chan0.tif')),
#             #str(Path(default_ops['data_path'][0]).joinpath('registration', 'regression_smoothed_chan1.tif'))
#         ]
#     )
