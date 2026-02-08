import suite2p
<<<<<<< HEAD


def test_do_registration_do_roi_detect_settings_check_timing(test_ops):
=======
import numpy as np
from pathlib import Path


def test_do_registration_do_roi_detect_settings_check_timing(test_settings):
>>>>>>> suite2p_dev/tomerge
    """
    Tests that do_registration and roidetect parameters don't error. Also, makes sure timing_report
    doesn't error and checks timing dictionary to make sure the times for only the stages specified are provided.
    """
<<<<<<< HEAD
    test_ops.update({
        'tiff_list': ['input.tif'],
        'roidetect': False,
        'do_registration': False,
        'spikedetect': False,
        'batch_size': 100,
    })
    conv_only_ops = suite2p.run_s2p(ops=test_ops)  # conversion only
    assert list(conv_only_ops['timing'].keys()) == ['total_plane_runtime']
    test_ops.update({
        'do_registration': True
    })
    reg_only_ops = suite2p.run_s2p(ops=test_ops)  # registration only
    assert list(reg_only_ops['timing'].keys()) == ['registration', 'total_plane_runtime']
    test_ops.update({
        'do_registration': False,
        'roidetect': True,
    })
    det_only_ops = suite2p.run_s2p(ops=test_ops)  # detection step only
    assert list(det_only_ops['timing'].keys()) == ['detection', 'extraction', 'classification', 'total_plane_runtime']
    test_ops.update({
        'spikedetect': True
    })
    det_dec_ops = suite2p.run_s2p(ops=test_ops)  # detection & deconvolution
    assert list(det_dec_ops['timing'].keys()) == ['detection', 'extraction', 'classification',
                                                     'deconvolution', 'total_plane_runtime']
=======
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
>>>>>>> suite2p_dev/tomerge
