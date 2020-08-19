import suite2p


def test_do_registration_do_roi_detect_settings_check_timing(test_ops):
    """
    Tests that do_registration and roidetect parameters don't error. Also, makes sure timing_report
    doesn't error and checks timing dictionary to make sure the times for only the stages specified are provided.
    """
    test_ops.update({
        'tiff_list': ['input.tif'],
        'roidetect': False,
        'do_registration': False,
        'spikedetect': False,
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
