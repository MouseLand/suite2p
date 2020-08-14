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
    suite2p.run_s2p(ops=test_ops)  # conversion only
    test_ops.update({
        'do_registration': True
    })
    suite2p.run_s2p(ops=test_ops)  # registration only
    test_ops.update({
        'do_registration': False,
        'roidetect': True,
    })
    suite2p.run_s2p(ops=test_ops)  # detection only
    test_ops.update({
        'spikedetect': True
    })
    suite2p.run_s2p(ops=test_ops)  # detection & deconvolution
