"""
Tests for the Suite2p Extraction module.
"""
import numpy as np
from suite2p import extraction
from suite2p.io import BinaryFile

from pathlib import Path
import utils

def extract_wrapper(ops):
    for plane in range(ops[0]['nplanes']):
        curr_op = ops[plane]
        plane_dir = Path(curr_op['save_path0']).joinpath(f'suite2p/plane{plane}')
        plane_dir.mkdir(exist_ok=True, parents=True)
        extract_input = np.load(
            curr_op['data_path'][0].parent.joinpath(
                'test_outputs',
                'detection',
                'expected_detect_output_{0}p{1}c{2}.npy'.format(curr_op['nplanes'], curr_op['nchannels'], plane)),
            allow_pickle=True
        )[()]
        #extraction.create_masks_and_extract(curr_op, extract_input['stat'])
        stat, F, Fneu, F_chan2, Fneu_chan2 = extraction.create_masks_and_extract(
            curr_op,
            extract_input['stat'],
            extract_input['cell_masks'],
            extract_input['neuropil_masks']
        )
        dF = F - curr_op['neucoeff'] * Fneu
        dF = extraction.preprocess(
            F=dF,
            baseline=curr_op['baseline'],
            win_baseline=curr_op['win_baseline'],
            sig_baseline=curr_op['sig_baseline'],
            fs=curr_op['fs'],
            prctile_baseline=curr_op['prctile_baseline']
        )
        spks = extraction.oasis(F=dF, batch_size=curr_op['batch_size'], tau=curr_op['tau'], fs=curr_op['fs'])
        np.save(plane_dir.joinpath('ops.npy'), curr_op)
        np.save(plane_dir.joinpath('stat.npy'), stat)
        np.save(plane_dir.joinpath('F.npy'), F)
        np.save(plane_dir.joinpath('Fneu.npy'), Fneu)
        np.save(plane_dir.joinpath('F_chan2.npy'), F_chan2)
        np.save(plane_dir.joinpath('Fneu_chan2.npy'), Fneu_chan2)
        np.save(plane_dir.joinpath('spks.npy'), spks)


def run_preprocess(f: np.ndarray, test_ops):
    baseline_vals = ['maximin', 'constant', 'constant_prctile']
    for bv in baseline_vals:
        pre_f = extraction.preprocess(
            F=f,
            baseline=bv,
            win_baseline=test_ops['win_baseline'],
            sig_baseline=test_ops['sig_baseline'],
            fs=test_ops['fs'],
            prctile_baseline=test_ops['prctile_baseline']
        )
        test_f = np.load(test_ops['data_path'][0].parent.joinpath('test_outputs/extraction/{}_f.npy'.format(bv)))
        yield np.allclose(pre_f, test_f, rtol=1e-4, atol=5e-2)


def test_pre_process_baseline(test_ops):
    f = np.load(test_ops['data_path'][0].parent.joinpath('test_outputs/1plane1chan1500/suite2p/plane0/F.npy'))
    assert all(run_preprocess(f, test_ops))


def test_extraction_output_1plane1chan(test_ops):
    test_ops.update({
        'tiff_list': ['input.tif'],
    })
    ops = utils.ExtractionTestUtils.prepare(
        test_ops,
        [[test_ops['data_path'][0].joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )
    extract_wrapper(ops)
    ops = ops[0]
    nplanes = ops['nplanes']
    outputs_to_check = ['F', 'Fneu', 'stat', 'spks']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, Path(ops['data_path'][0]).parent.joinpath(f"test_outputs/extraction/1plane1chan/plane0")),
            utils.get_list_of_data(outputs_to_check, Path(ops['save_path0']).joinpath(f"suite2p/plane0")),
        ))


def test_extraction_output_2plane2chan(test_ops):
    test_ops.update({
        'nchannels': 2,
        'nplanes': 2,
        'tiff_list': ['input.tif'],
    })
    detection_dir = test_ops['data_path'][0].joinpath('detection')
    ops = utils.ExtractionTestUtils.prepare(
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
    outputs_to_check = ['F', 'Fneu', 'F_chan2', 'Fneu_chan2', 'stat', 'spks']
    for i in range(len(ops)):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, Path(ops[i]['data_path'][0].parent.joinpath(f"test_outputs/extraction/2plane2chan/plane{i}"))),
            utils.get_list_of_data(outputs_to_check, Path(ops[i]['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))