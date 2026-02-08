"""
Tests for the Suite2p Extraction module.
Structured to match generate_test_data.py pattern for true regression testing.
"""
import numpy as np
import torch
import suite2p
from suite2p import extraction
from pathlib import Path
import utils


def test_pre_process_baseline(test_settings):
    """
    Regression test for baseline preprocessing methods.
    Tests different baseline methods exactly like generate_test_data.py.
    """
    db, settings = test_settings  # Unpack the tuple
    op = {**db, **settings}  # Merge for legacy test utilities
    
    # Load F from full pipeline output
    f = np.load(op['data_path'][0].parent.joinpath('test_outputs/1plane1chan1500/suite2p/plane0/F.npy'))

    # Test all baseline methods like generate_test_data.py
    baseline_vals = ['maximin', 'constant', 'constant_prctile']
    for bv in baseline_vals:
        pre_f = extraction.preprocess(
            F=f,
            baseline=bv,
            win_baseline=op['dcnv_preprocess']['win_baseline'],
            sig_baseline=op['dcnv_preprocess']['sig_baseline'],
            fs=op['fs'],
            prctile_baseline=op['dcnv_preprocess']['prctile_baseline'],
            device=torch.device(op['torch_device'])
        )
        expected_f = np.load(op['data_path'][0].parent.joinpath(f'test_outputs/extraction/{bv}_f.npy'))
        assert np.allclose(pre_f, expected_f, rtol=1e-4, atol=5e-2)


def test_extraction_output_1plane1chan(test_settings):
    """
    Regression test for 1 plane, 1 channel extraction.
    Runs extraction exactly like generate_test_data.py and compares outputs.
    """
    db, settings = test_settings  # Unpack the tuple
    db = {**db, **settings}  # Merge for ExtractionTestUtils

    # Prepare settings exactly like generate_test_data.py
    settings = utils.ExtractionTestUtils.prepare(
        db,
        [[Path(db['data_path'][0]).joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )

    op = settings[0]

    # Load detection output
    extract_input = np.load(
        Path(db['data_path'][0]).parent.joinpath('test_outputs/detection/expected_detect_output_1p1c0.npy'),
        allow_pickle=True
    )[()]

    # Run extraction exactly like generate_test_data.py extract_helper function
    with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
        # Compute Fluorescence Extraction
        F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(
            extract_input['stat'],
            f_reg,
            cell_masks=extract_input['cell_masks'],
            neuropil_masks=extract_input['neuropil_masks'],
            settings=op,
            device=torch.device(op['torch_device'])
        )
        # Deconvolve spikes from fluorescence
        dF = F.copy() - op["extraction"]["neuropil_coefficient"] * Fneu
        dF = extraction.preprocess(F=dF, fs=op["fs"], **op["dcnv_preprocess"], device=torch.device(op['torch_device']))
        spks = extraction.oasis(F=dF, batch_size=op["extraction"]["batch_size"], tau=op["tau"], fs=op["fs"])

    # Compare outputs with expected
    expected_dir = Path(db['data_path'][0]).parent.joinpath("test_outputs/extraction/1plane1chan/suite2p/plane0")
    expected_F = np.load(expected_dir / 'F.npy')
    expected_Fneu = np.load(expected_dir / 'Fneu.npy')
    expected_spks = np.load(expected_dir / 'spks.npy')

    assert np.allclose(F, expected_F, rtol=1e-4, atol=5e-2)
    assert np.allclose(Fneu, expected_Fneu, rtol=1e-4, atol=5e-2)
    assert np.allclose(spks, expected_spks, rtol=1e-4, atol=5e-2)


def test_extraction_output_2plane2chan(test_settings):
    """
    Regression test for 2 planes, 2 channels extraction.
    Runs extraction exactly like generate_test_data.py and compares outputs.
    """
    db, settings = test_settings  # Unpack the tuple
    db = {**db, **settings}  # Merge for ExtractionTestUtils
    db.update({
        'nchannels': 2,
        'nplanes': 2,
        'file_list': ['input.tif'],
    })

    detection_dir = db['data_path'][0].joinpath('detection')

    # Prepare settings exactly like generate_test_data.py
    settings = utils.ExtractionTestUtils.prepare(
        db,
        [
            [detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
            [detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
        ],
        (404, 360),
    )

    settings[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
    settings[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))

    # Run extraction for each plane exactly like generate_test_data.py extract_helper function
    for i, op in enumerate(settings):
        # Load detection output
        extract_input = np.load(
            Path(db['data_path'][0]).parent.joinpath(f'test_outputs/detection/expected_detect_output_2p2c{i}.npy'),
            allow_pickle=True
        )[()]

        # Run extraction exactly like generate_test_data.py extract_helper function
        with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
            with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file_chan2']) as f_reg_chan2:
                # Compute Fluorescence Extraction
                F, Fneu, F_chan2, Fneu_chan2 = extraction.extraction_wrapper(
                    extract_input['stat'],
                    f_reg,
                    f_reg_chan2=f_reg_chan2,
                    cell_masks=extract_input['cell_masks'],
                    neuropil_masks=extract_input['neuropil_masks'],
                    settings=op,
                    device=torch.device(op['torch_device'])
                )
                # Deconvolve spikes from fluorescence
                dF = F.copy() - op["extraction"]["neuropil_coefficient"] * Fneu
                dF = extraction.preprocess(F=dF, fs=op["fs"], device=torch.device(op['torch_device']), **op["dcnv_preprocess"])
                spks = extraction.oasis(F=dF, batch_size=op["extraction"]["batch_size"], tau=op["tau"], fs=op["fs"])

        # Compare outputs with expected
        expected_dir = Path(db['data_path'][0]).parent.joinpath(f"test_outputs/extraction/2plane2chan/suite2p/plane{i}")
        expected_F = np.load(expected_dir / 'F.npy')
        expected_Fneu = np.load(expected_dir / 'Fneu.npy')
        expected_F_chan2 = np.load(expected_dir / 'F_chan2.npy')
        expected_Fneu_chan2 = np.load(expected_dir / 'Fneu_chan2.npy')
        expected_spks = np.load(expected_dir / 'spks.npy')

        assert np.allclose(F, expected_F, rtol=1e-4, atol=5e-2)
        assert np.allclose(Fneu, expected_Fneu, rtol=1e-4, atol=5e-2)
        assert np.allclose(F_chan2, expected_F_chan2, rtol=1e-4, atol=5e-2)
        assert np.allclose(Fneu_chan2, expected_Fneu_chan2, rtol=1e-4, atol=5e-2)
        assert np.allclose(spks, expected_spks, rtol=1e-4, atol=5e-2)
