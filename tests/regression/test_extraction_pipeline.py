"""
Tests for the Suite2p Extraction module.
"""
import numpy as np
from suite2p import extraction
from suite2p.io import BinaryFile

from pathlib import Path
import utils


def prepare_for_extraction(op, input_file_name_list, dimensions):
    """
    Prepares for extraction by filling out necessary ops parameters. Removes dependence on
    other modules. Creates pre_registered binary file.
    """
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
            curr_op['xrange'], curr_op['yrange'] = [[1, 403], [1, 359]]
        if curr_op['nchannels'] == 2:
            bin2_path = str(plane_dir.joinpath('data_chan2.bin'))
            BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][1]), bin2_path)
            curr_op['reg_file_chan2'] = bin2_path
        curr_op['save_path'] = plane_dir
        curr_op['ops_path'] = plane_dir.joinpath('ops.npy')
        ops.append(curr_op)
    return ops


def extract_wrapper(ops):
    for plane in range(ops[0]['nplanes']):
        curr_op = ops[plane]
        plane_dir = Path(curr_op['save_path0']).joinpath(f'suite2p/plane{plane}')
        plane_dir.mkdir(exist_ok=True, parents=True)
        extract_input = np.load(
            curr_op['data_path'][0].joinpath(
                'detection',
                'detect_output_{0}p{1}c{2}.npy'.format(curr_op['nplanes'], curr_op['nchannels'], plane)),
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
        test_f = np.load('data/test_data/detection/{}_f.npy'.format(bv))
        yield np.allclose(pre_f, test_f, rtol=1e-4, atol=5e-2)


def test_pre_process_baseline(test_ops):
    f = np.load(Path('data/test_data/1plane1chan/suite2p/plane0/F.npy'))
    assert all(run_preprocess(f, test_ops))


def test_extraction_output_1plane1chan(test_ops):
    ops = prepare_for_extraction(
        test_ops,
        [[test_ops['data_path'][0].joinpath('detection/pre_registered.npy')]],
        (404, 360)
    )
    extract_wrapper(ops)
    nplanes = test_ops['nplanes']
    outputs_to_check = ['F', 'Fneu', 'stat', 'spks']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, Path(test_ops['data_path'][0]).joinpath(f"{nplanes}plane{test_ops['nchannels']}chan/suite2p/plane{i}")),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))


def test_extraction_output_2plane2chan(test_ops):
    test_ops.update({
        'nchannels': 2,
        'nplanes': 2,
    })

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
    nplanes = test_ops['nplanes']
    outputs_to_check = ['F', 'Fneu', 'F_chan2', 'Fneu_chan2', 'stat', 'spks']
    for i in range(nplanes):
        assert all(utils.compare_list_of_outputs(
            outputs_to_check,
            utils.get_list_of_data(outputs_to_check, Path(test_ops['data_path'][0].joinpath(f"{nplanes}plane{test_ops['nchannels']}chan/suite2p/plane{i}"))),
            utils.get_list_of_data(outputs_to_check, Path(test_ops['save_path0']).joinpath(f"suite2p/plane{i}")),
        ))