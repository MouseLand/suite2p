"""
Tests for the Suite2p Detection module.
Structured to match generate_test_data.py pattern for true regression testing.
"""
from pathlib import Path
import numpy as np
import suite2p
import utils
from suite2p.extraction import masks


def test_detection_output_1plane1chan(test_settings):
    """
    Regression test for 1 plane, 1 channel detection.
    Runs detection exactly like generate_test_data.py and compares outputs.
    """
    db, settings = test_settings  # Unpack the tuple
    db = {**db, **settings}  # Merge for DetectionTestUtils
    db.update({'file_list': ['input.tif']})

    # Prepare settings exactly like generate_test_data.py
    settings = utils.DetectionTestUtils.prepare(
        db,
        [[Path(db['data_path'][0]).joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )

    op = settings[0]
    # Run detection exactly like generate_test_data.py
    with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
        op['neuropil_extract'] = True
        _, stat, _ = suite2p.detection.detection_wrapper(f_reg, settings=op)
        cell_masks, neuropil_masks = masks.create_masks(stat, op['Ly'], op['Lx'], neuropil_extract=True)

    # Load expected output
    expected = np.load(
        op['data_path'][0].parent.joinpath('test_outputs/detection/expected_detect_output_1p1c0.npy'),
        allow_pickle=True
    )[()]

    # Compare outputs
    assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(cell_masks, expected['cell_masks']))
    assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(neuropil_masks, expected['neuropil_masks']))
    for gt_dict, output_dict in zip(stat, expected['stat']):
        for k in gt_dict.keys():
            if k in ['ypix', 'xpix', 'lam']:
                assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)


def test_detection_output_2plane2chan(test_settings):
    """
    Regression test for 2 planes, 2 channels detection.
    Runs detection exactly like generate_test_data.py and compares outputs.
    """
    db, settings = test_settings  # Unpack the tuple
    db = {**db, **settings}  # Merge for DetectionTestUtils
    db.update({'file_list': ['input.tif'], 'nchannels': 2, 'nplanes': 2})

    detection_dir = Path(db['data_path'][0]).joinpath('detection')

    # Prepare settings exactly like generate_test_data.py
    settings = utils.DetectionTestUtils.prepare(
        db,
        [
            [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
            [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
        ],
        (404, 360)
    )

    # Load channel 2 mean images like generate_test_data.py
    settings[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
    settings[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))

    # Run detection for each plane exactly like generate_test_data.py
    for i, op in enumerate(settings):
        with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
            op['neuropil_extract'] = True
            _, stat, _ = suite2p.detection.detection_wrapper(f_reg, settings=op)
            cell_masks, neuropil_masks = masks.create_masks(stat, op['Ly'], op['Lx'], neuropil_extract=True)

        # Load expected output for this plane
        expected = np.load(
            Path(db['data_path'][0]).parent.joinpath(f'test_outputs/detection/expected_detect_output_2p2c{i}.npy'),
            allow_pickle=True
        )[()]

        # Compare outputs
        assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(cell_masks, expected['cell_masks']))
        assert all(np.allclose(a, b, rtol=1e-4, atol=5e-2) for a, b in zip(neuropil_masks, expected['neuropil_masks']))
        for gt_dict, output_dict in zip(stat, expected['stat']):
            for k in gt_dict.keys():
                if k in ['ypix', 'xpix', 'lam']:
                    assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)
