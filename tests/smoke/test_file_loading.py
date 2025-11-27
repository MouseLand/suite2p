from pathlib import Path

import numpy as np
import pytest
import suite2p


def test_bruker(test_settings):
    db, settings = test_settings  # Unpack the tuple
    db['data_path'] = [Path(db['data_path'][0]).joinpath('bruker')]
    db['input_format'] = 'bruker'
    print(db['nchannels'])
    settings['detection']['threshold_scaling'] = 0.5  # Lower threshold
    suite2p.run_s2p(settings=settings, db=db)


def test_h5_file_is_processed_end_to_end(test_settings):
    """Ensure the sample h5 file loads correctly and runs through Suite2p."""
    h5py = pytest.importorskip("h5py")

    db, settings = test_settings  # Unpack the tuple
    data_root = Path(db['data_path'][0])
    db.update({
        'file_list': ['input.h5'],
        'input_format': 'h5',
        'nplanes': 1,
        'nchannels': 1,
    })
    settings['run']['do_deconvolution'] = False  # keep runtimes modest
    settings['run']['do_regmetrics'] = False
    settings['io']['delete_bin'] = True

    suite2p.run_s2p(settings=settings, db=db)

    plane_dir = Path(db['save_path0']) / 'suite2p' / 'plane0'
    assert plane_dir.exists()

    ops = np.load(plane_dir / 'ops.npy', allow_pickle=True).item()
    with h5py.File(data_root / 'input.h5', 'r') as handle:
        expected_frames, expected_ly, expected_lx = handle['data'].shape

    assert ops['nframes'] == expected_frames
    assert ops['Ly'] == expected_ly
    assert ops['Lx'] == expected_lx
    assert (plane_dir / 'F.npy').exists()
    assert (plane_dir / 'iscell.npy').exists()
