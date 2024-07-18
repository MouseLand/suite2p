"""
Tests for the Suite2p Detection module.
"""
from pathlib import Path
import numpy as np
import utils
from suite2p import detection
from suite2p.extraction import masks

def detect_wrapper(settings):
    """
    Calls the main detect function and compares output dictionaries (cell_pix, cell_masks,
    neuropil_masks, stat) with prior output dicts.
    """
    for i in range(len(settings)):
        op = settings[i]
        op['neuropil_extract'] = True
        op, stat = detection.detect(settings=op)
        output_check = np.load(
            op['data_path'][0].parent.joinpath(f"test_outputs/detection/expected_detect_output_{ op['nplanes'] }p{ op['nchannels'] }c{ i }.npy"),
            allow_pickle=True
        )[()]
        #assert np.array_equal(output_check['cell_pix'], cell_pix)
        cell_masks, neuropil_masks = masks.create_masks(stat, op['Ly'], op['Lx'], settings=op)
        assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(cell_masks, output_check['cell_masks']))
        assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(neuropil_masks, output_check['neuropil_masks']))
        for gt_dict, output_dict in zip(stat, output_check['stat']):
            for k in gt_dict.keys():
                if k=='ypix' or k=='xpix' or k=='lam':
                    assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)

def test_detection_output_1plane1chan(test_settings):
    test_settings.update({
        'tiff_list': ['input.tif'],
    })
    settings = utils.DetectionTestUtils.prepare(
        test_settings,
        [[test_settings['data_path'][0].joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )
    detect_wrapper(settings)


def test_detection_output_2plane2chan(test_settings):
    test_settings.update({
        'nchannels': 2,
        'nplanes': 2,
    })
    detection_dir = test_settings['data_path'][0].joinpath('detection')
    settings = utils.DetectionTestUtils.prepare(
        test_settings,
        [
            [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
            [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
        ]
        , (404, 360),
    )
    settings[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
    settings[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))
    detect_wrapper(settings)
    nplanes = test_settings['nplanes']

    outputs_to_check = ['redcell']
    # rely on the 2plane2chan1500's redcell
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, test_settings['data_path'][0].parent.joinpath(f"test_outputs/detection/suite2p/plane{i}")),
            utils.get_list_of_data(outputs_to_check, Path(test_settings['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))
