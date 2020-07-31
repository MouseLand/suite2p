"""
Tests for the Suite2p Extraction module.
"""
import numpy as np
from suite2p import extraction
import utils


def prepare_for_extraction(op, input_file_name_list, dimensions):
    """
    Prepares for extraction by filling out necessary ops parameters. Removes dependence on
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
        curr_op['meanImg'] = np.reshape(
            np.load(str(input_file_name_list[plane][0])), (-1, op['Ly'], op['Lx'])
        ).mean(axis=0)
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


def extract_wrapper(ops):
    for plane in range(ops[0]['nplanes']):
        curr_op = ops[plane]
        plane_dir = utils.get_plane_dir(curr_op, plane)
        extract_input = np.load(
            curr_op['data_path'][0].joinpath(
                'detection',
                'detect_output_{0}p{1}c{2}.npy'.format(curr_op['nplanes'], curr_op['nchannels'], plane)),
            allow_pickle=True
        )[()]
        extraction.extract(
            curr_op,
            extract_input['cell_pix'],
            extract_input['cell_masks'],
            extract_input['neuropil_masks'],
            extract_input['stat']
        )
        F = np.load(plane_dir.joinpath('F.npy'))
        Fneu = np.load(plane_dir.joinpath('Fneu.npy'))
        dF = F - curr_op['neucoeff'] * Fneu
        dF = extraction.preprocess(dF, curr_op)
        spks = extraction.oasis(dF, curr_op['batch_size'], curr_op['tau'], curr_op['fs'])
        np.save(plane_dir.joinpath('spks.npy'), spks)


def test_extraction_output_1plane1chan(test_ops):
    ops = prepare_for_extraction(
        test_ops,
        [[test_ops['data_path'][0].joinpath('detection', 'pre_registered.npy')]],
        (404, 360)
    )
    extract_wrapper(ops)
    utils.check_output(
        test_ops['save_path0'],
        ['F', 'Fneu', 'stat', 'spks'],
        test_ops['data_path'][0],
        test_ops['nplanes'],
        test_ops['nchannels'],
    )


def test_extraction_output_2plane2chan(test_ops):
    test_ops['nchannels'] = 2
    test_ops['nplanes'] = 2
    detection_dir = test_ops['data_path'][0].joinpath('detection')
    ops = prepare_for_extraction(
        test_ops,
        [
            [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
            [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
        ]
        , (404, 360),
    )
    ops[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
    ops[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))
    extract_wrapper(ops)
    utils.check_output(
        test_ops['save_path0'],
        ['F', 'Fneu', 'F_chan2', 'Fneu_chan2', 'stat', 'spks'],
        test_ops['data_path'][0],
        test_ops['nplanes'],
        test_ops['nchannels'],
    )