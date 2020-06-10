"""
Tests for the Suite2p Detection module.
"""

import numpy as np
import utils
from suite2p import detection


def prepare_for_detection(op, input_file_name_list, dimensions):
    """
    Prepares for detection by filling out necessary ops parameters. Removes dependence on
    other modules. Creates pre_registered binary file.
    """
    # Set appropriate ops parameters
    op['Lx'], op['Ly'] = dimensions
    op['nframes'] = 500 // op['nplanes'] // op['nchannels']
    op['frames_per_file'] = 500 // op['nplanes'] // op['nchannels']
    op['xrange'], op['yrange'] = [[2, 402], [2, 358]]
    ops = []
    for plane in range(op['nplanes']):
        curr_op = op.copy()
        plane_dir = utils.get_plane_dir(op, plane)
        bin_path = utils.write_data_to_binary(
            str(plane_dir.joinpath('data.bin')), str(input_file_name_list[plane][0])
        )
        curr_op['reg_file'] = bin_path
        if plane == 1: # Second plane result has different crop.
            curr_op['xrange'], curr_op['yrange'] = [[1, 403], [1, 359]]
        if curr_op['nchannels'] == 2:
            bin2_path = utils.write_data_to_binary(
                str(plane_dir.joinpath('data_chan2.bin')), str(input_file_name_list[plane][1])
            )
            curr_op['reg_file_chan2'] = bin2_path
        curr_op['save_path'] = plane_dir
        curr_op['ops_path'] = plane_dir.joinpath('ops.npy')
        ops.append(curr_op)
    return ops


def test_detection_extraction_output_1plane1chan(default_ops):
    ops = prepare_for_detection(
        default_ops,
        [[default_ops['data_path'][0].joinpath('detection', 'pre_registered.npy')]],
        (404, 360)
    )
    cell_pix, cell_masks, neuropil_masks, stat = detection.main_detect(ops, None)
    utils.check_output(
        default_ops['save_path0'],
        ['F', 'Fneu', 'iscell', 'stat', 'spks'],
        default_ops['data_path'][0],
        default_ops['nplanes'],
        default_ops['nchannels'],
    )