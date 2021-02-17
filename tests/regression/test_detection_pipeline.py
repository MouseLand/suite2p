"""
Tests for the Suite2p Detection module.
"""
from pathlib import Path
import numpy as np
import utils
from suite2p import detection
from suite2p.extraction import masks
from suite2p.io import BinaryFile


def prepare_for_detection(op, input_file_name_list, dimensions):
    """
    Prepares for detection by filling out necessary ops parameters. Removes dependence on
    other modules. Creates pre_registered binary file.
    """
    # Set appropriate ops parameters
    op.update({
        'Lx': dimensions[0],
        'Ly': dimensions[1],
        'nframes': 500 // op['nplanes'] // op['nchannels'],
        'frames_per_file': 500 // op['nplanes'] // op['nchannels'],
        'xrange': [2, 402],
        'yrange': [2, 358],
    })
    ops = []
    for plane in range(op['nplanes']):
        curr_op = op.copy()
        plane_dir = Path(op['save_path0']).joinpath(f'suite2p/plane{plane}')
        plane_dir.mkdir(exist_ok=True, parents=True)
        bin_path = str(plane_dir.joinpath('data.bin'))
        BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][0]), bin_path)
        curr_op['meanImg'] = np.reshape(
            np.load(str(input_file_name_list[plane][0])), (-1, op['Ly'], op['Lx'])
        ).mean(axis=0)
        curr_op['reg_file'] = bin_path
        if plane == 1: # Second plane result has different crop.
            curr_op['xrange'] = [1, 403]
            curr_op['yrange'] = [1, 359]
        if curr_op['nchannels'] == 2:
            bin2_path = str(plane_dir.joinpath('data_chan2.bin'))
            BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][1]), bin2_path)
            curr_op['reg_file_chan2'] = bin2_path
        curr_op['save_path'] = plane_dir
        curr_op['ops_path'] = plane_dir.joinpath('ops.npy')
        ops.append(curr_op)
    return ops


def detect_wrapper(ops):
    """
    Calls the main detect function and compares output dictionaries (cell_pix, cell_masks,
    neuropil_masks, stat) with prior output dicts.
    """
    for i in range(len(ops)):
        op = ops[i]
        op, stat = detection.detect(ops=op)
        output_check = np.load(
            op['data_path'][0].joinpath(f"detection/detect_output_{ op['nplanes'] }p{ op['nchannels'] }c{ i }.npy"),
            allow_pickle=True
        )[()]
        #assert np.array_equal(output_check['cell_pix'], cell_pix)
        cell_masks = masks.create_masks(op, stat)[0]
        assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(cell_masks, output_check['cell_masks']))
        #assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(neuropil_masks, output_check['neuropil_masks']))
        for s in stat:
            s['lam'] /= s['lam'].sum()
        for gt_dict, output_dict in zip(stat, output_check['stat']):
            for k in gt_dict.keys():
                if k=='ypix' or k=='xpix' or k=='lam':
                    assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)


def test_detection_output_1plane1chan(test_ops):
    ops = prepare_for_detection(
        test_ops,
        [[test_ops['data_path'][0].joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )
    detect_wrapper(ops)


def test_detection_output_2plane2chan(test_ops):
    test_ops.update({
        'nchannels': 2,
        'nplanes': 2,
    })
    detection_dir = test_ops['data_path'][0].joinpath('detection')
    ops = prepare_for_detection(
        test_ops,
        [
            [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
            [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
        ]
        , (404, 360),
    )
    ops[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
    ops[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))
    detect_wrapper(ops)
    nplanes = test_ops['nplanes']

    outputs_to_check = ['redcell']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan/suite2p/plane{i}")),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))
