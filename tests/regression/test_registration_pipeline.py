"""
Tests for the Suite2p Registration Module
"""

import numpy as np
from pathlib import Path
from tifffile import imread
import contextlib
import os
from suite2p import registration, io #import register_binary


def prepare_for_registration(op, input_file_name, dimensions):
    """
    Prepares for registration by performing functions of io module. Fills out necessary settings parameters for
    registration module.
    """
    op.update({
        'reg_tif': True,
        'Lx': dimensions[0],
        'Ly': dimensions[1],
    })

    nc = op['nchannels']
    # Make input data non-negative
    input_data = (imread(
        str(input_file_name)
    ) // 2).astype(np.int16)
    nframes_per_plane = input_data.shape[0] // op['nplanes']
    settings = []
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
        settings.append(curr_op)
    return settings


def check_registration_output(op, dimensions, input_path, reg_output_path_list, output_path_list):
    settings = prepare_for_registration(
        op, input_path, dimensions
    )
    reg_settings = []
    npl = op['nplanes']
    for i in range(npl):
        #curr_op = register_binary(settings[i])
        raw = settings[i].get("keep_movie_raw") and "raw_file" in settings[i] and os.path.isfile(
            settings[i]["raw_file"])
        reg_file = settings[i]["reg_file"]
        raw_file = settings[i].get("raw_file", 0) if raw else reg_file
        if settings[i]["nchannels"] > 1:
            reg_file_chan2 = settings[i]["reg_file_chan2"]
            raw_file_chan2 = settings[i].get("raw_file_chan2",
                                    0) if raw else reg_file_chan2
        else:
            reg_file_chan2 = reg_file
            raw_file_chan2 = reg_file
        null = contextlib.nullcontext()
        twoc = settings[i]["nchannels"] > 1
        raw_file = settings[i]["raw_file"]
        Ly, Lx = settings[i]["Ly"], settings[i]["Lx"]
        with io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file) if raw else null as f_raw, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file) as f_reg, \
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=raw_file_chan2) if raw and twoc else null as f_raw_chan2,\
            io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file_chan2) if twoc else null as f_reg_chan2:
            
            registration_outputs = registration.register.registration_wrapper(
                f_reg, f_raw=f_raw, f_reg_chan2=f_reg_chan2,
                f_raw_chan2=f_raw_chan2, settings=settings[i])
            curr_op = registration.register.save_registration_outputs_to_settings(
                registration_outputs, settings[i])
        registered_data = imread(reg_output_path_list[i*npl])
        output_check = imread(output_path_list[i*npl])
        assert np.array_equal(registered_data, output_check)
        if op['nchannels'] == 2:
            registered_data = imread(reg_output_path_list[i * npl + 1])
            output_check = imread(output_path_list[i * npl + 1])
            assert np.array_equal(registered_data, output_check)
        reg_settings.append(curr_op)
    return reg_settings


#def test_register_binary_do_bidi_output(test_settings):
#    """
#    Regression test that checks the output of register_binary given the `input.tif` with the bidiphase,
#    """
#    test_settings['do_bidiphase'] = True
#    check_registration_output(
#        test_settings, (404, 360),
#        test_settings['data_path'][0].joinpath('registration/bidi_shift_input.tif'),
#        [str(Path(test_settings['save_path0']).joinpath('reg_tif/file000_chan0.tif'))],
#        [str(Path(test_settings['data_path'][0]).joinpath('registration/regression_bidi_output.tif'))]
#    )


#def test_register_binary_rigid_registration_only(test_settings):
#    """
#    Tests that register_binary works for a dataset that only has rigid shifts.
#    """
#    test_settings['nonrigid'] = False
#    op = prepare_for_registration(
#        test_settings,
#        test_settings['data_path'][0].joinpath('registration/rigid_registration_test_data.tif'),
#        (256, 256),
#    )[0]
#    op = register_binary(op)
#    registered_data = imread(str(Path(op['save_path']).joinpath('reg_tif/file000_chan0.tif')))
#    # Make sure registered_data is identical across frames
#    check_data = np.repeat(registered_data[0, :, :][np.newaxis, :, :], 500, axis=0)
#    assert np.array_equal(check_data, registered_data)
#    # Check and see if there are exactly 16 lines row-wise and column-wise
#    num_row_lines = len(np.where(np.all(np.all(check_data == 1500, axis=0), axis=0))[0])
#    num_col_lines = len(np.where(np.all(np.all(check_data == 1500, axis=0), axis=1))[0])
#    assert num_col_lines == 16
#    assert num_row_lines == 16

