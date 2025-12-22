from pathlib import Path

import pytest
import suite2p


def test_bruker(test_settings):
    """Verify Bruker OME-TIFF inputs run end-to-end."""

    db, settings = test_settings  # Unpack the tuple
    data_root = Path(db['data_path'][0])
    db.update({
        'data_path': [data_root.joinpath('bruker')],
        'input_format': 'bruker',
        'nplanes': 1,
        'nchannels': 2,
        'functional_chan': 1,
        'force_sktiff': True,
    })
    settings['detection']['threshold_scaling'] = 0.5  # Lower threshold
    settings['run']['do_regmetrics'] = False
    settings['io']['delete_bin'] = True

    suite2p.run_s2p(settings=settings, db=db)

    plane_dir = Path(db['save_path0']) / 'suite2p' / 'plane0'
    assert plane_dir.exists()

    assert (plane_dir / 'ops.npy').exists()
    assert (plane_dir / 'F.npy').exists()
    assert (plane_dir / 'Fneu.npy').exists()
    assert (plane_dir / 'iscell.npy').exists()
    assert (plane_dir / 'F_chan2.npy').exists()
    assert (plane_dir / 'Fneu_chan2.npy').exists()


def test_h5_file_is_processed_end_to_end(test_settings):
    """Ensure the sample h5 file loads correctly and runs through Suite2p."""
    pytest.importorskip("h5py")

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

    assert (plane_dir / 'ops.npy').exists()
    assert (plane_dir / 'F.npy').exists()
    assert (plane_dir / 'iscell.npy').exists()
