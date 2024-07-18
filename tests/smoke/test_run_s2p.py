import suite2p


def test_do_registration_do_roi_detect_settings_check_timing(test_settings):
    """
    Tests that do_registration and roidetect parameters don't error. Also, makes sure timing_report
    doesn't error and checks timing dictionary to make sure the times for only the stages specified are provided.
    """
    test_settings.update({
        'tiff_list': ['input.tif'],
        'roidetect': False,
        'do_registration': False,
        'spikedetect': False,
        'batch_size': 100,
    })
    conv_only_settings = suite2p.run_s2p(settings=test_settings)  # conversion only
    assert list(conv_only_settings['timing'].keys()) == ['total_plane_runtime']
    test_settings.update({
        'do_registration': True
    })
    reg_only_settings = suite2p.run_s2p(settings=test_settings)  # registration only
    assert list(reg_only_settings['timing'].keys()) == ['registration', 'total_plane_runtime']
    test_settings.update({
        'do_registration': False,
        'roidetect': True,
    })
    det_only_settings = suite2p.run_s2p(settings=test_settings)  # detection step only
    assert list(det_only_settings['timing'].keys()) == ['detection', 'extraction', 'classification', 'total_plane_runtime']
    test_settings.update({
        'spikedetect': True
    })
    det_dec_settings = suite2p.run_s2p(settings=test_settings)  # detection & deconvolution
    assert list(det_dec_settings['timing'].keys()) == ['detection', 'extraction', 'classification',
                                                     'deconvolution', 'total_plane_runtime']
