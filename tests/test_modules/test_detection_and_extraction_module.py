import numpy as np
from pathlib import Path
from suite2p import extraction, registration


def prepare_for_detection(op, input_file_name_list, dimensions, test_utils):
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
        plane_dir = test_utils.get_plane_dir(op, plane)
        bin_path = test_utils.write_data_to_binary(
            str(plane_dir.joinpath('data.bin')), str(input_file_name_list[plane][0])
        )
        curr_op['reg_file'] = bin_path
        if plane == 1: # Second plane result has different crop.
            curr_op['xrange'], curr_op['yrange'] = [[1, 403], [1, 359]]
        if curr_op['nchannels'] == 2:
            bin2_path = test_utils.write_data_to_binary(
                str(plane_dir.joinpath('data_chan2.bin')), str(input_file_name_list[plane][1])
            )
            curr_op['reg_file_chan2'] = bin2_path
        curr_op['save_path'] = plane_dir
        curr_op['ops_path'] = plane_dir.joinpath('ops.npy')
        ops.append(curr_op)
    return ops


def detect_and_extract_wrapper(ops, test_utils):
    for plane in range(ops[0]['nplanes']):
        curr_op = ops[plane]
        plane_dir = test_utils.get_plane_dir(curr_op, plane)
        # Detection Part
        curr_op = extraction.detect_and_extract(curr_op)
        # Extraction part
        F = np.load(plane_dir.joinpath('F.npy'))
        Fneu = np.load(plane_dir.joinpath('Fneu.npy'))
        dF = F - curr_op['neucoeff'] * Fneu
        dF = extraction.preprocess(dF, curr_op)
        spks = extraction.oasis(dF, curr_op)
        np.save(plane_dir.joinpath('spks.npy'), spks)


class TestSuite2pDetectionExtractionModule:
    """
    Tests for the Suite2p Detection and Extraction module.
    """
    # TODO: Separate once the detection module is created.

    def test_detection_extraction_output_1plane1chan(self, setup_and_teardown, get_test_dir_path, test_utils):
        op, tmp_dir = setup_and_teardown
        ops = prepare_for_detection(
            op, [[op['data_path'][0].joinpath('detection', 'pre_registered.npy')]], (404, 360),
            test_utils
        )
        detect_and_extract_wrapper(ops, test_utils)
        test_utils.check_output(
            tmp_dir, ['F', 'Fneu', 'iscell', 'stat', 'spks'], get_test_dir_path, op['nplanes'], op['nchannels']
        )

    def test_detection_extraction_output_2plane2chan(self, setup_and_teardown, get_test_dir_path, test_utils):
        op, tmp_dir = setup_and_teardown
        op['nchannels'] = 2
        op['nplanes'] = 2
        detection_dir = op['data_path'][0].joinpath('detection')
        ops = prepare_for_detection(
            op,
            [
                [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
                [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
            ]
            , (404, 360),
            test_utils
        )
        ops[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
        ops[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))
        detect_and_extract_wrapper(ops, test_utils)
        test_utils.check_output(
            tmp_dir, ['F', 'Fneu', 'iscell', 'stat', 'spks'], get_test_dir_path, op['nplanes'], op['nchannels']
        )
