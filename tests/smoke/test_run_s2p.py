import suite2p
import numpy as np
from pathlib import Path


def test_do_registration_do_roi_detect_settings_check_timing(test_settings):
    """
    Tests that do_registration and roidetect parameters don't error. Also, makes sure timing_report
    doesn't error and checks timing dictionary to make sure the times for only the stages specified are provided.
    """
    db, settings = test_settings  # Unpack the tuple
    db.update({
        'file_list': ['input.tif'],
        'batch_size': 100,
    })
    settings['run']['do_detection'] = False
    settings['run']['do_registration'] = False
    settings['run']['do_deconvolution'] = False
    settings['io']['save_ops_orig'] = True  # Make sure ops.npy is saved
    suite2p.run_s2p(settings=settings, db=db)  # conversion only
    conv_only_ops = np.load(Path(db['save_path0']) / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()
    assert list(conv_only_ops['plane_times'].keys()) == ['total_plane_runtime']

    settings['run']['do_registration'] = True
    suite2p.run_s2p(settings=settings, db=db)  # registration only
    reg_only_ops = np.load(Path(db['save_path0']) / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()
    assert list(reg_only_ops['plane_times'].keys()) == ['registration', 'total_plane_runtime']

    settings['run']['do_registration'] = False
    settings['run']['do_detection'] = True
    suite2p.run_s2p(settings=settings, db=db)  # detection step only
    det_only_ops = np.load(Path(db['save_path0']) / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()
    assert list(det_only_ops['plane_times'].keys()) == ['detection', 'extraction', 'classification', 'total_plane_runtime']

    settings['run']['do_deconvolution'] = True
    suite2p.run_s2p(settings=settings, db=db)  # detection & deconvolution
    det_dec_ops = np.load(Path(db['save_path0']) / 'suite2p' / 'plane0' / 'ops.npy', allow_pickle=True).item()
    assert list(det_dec_ops['plane_times'].keys()) == ['detection', 'extraction', 'deconvolution',
                                                     'classification', 'total_plane_runtime']
