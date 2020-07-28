"""
Tests for the Suite2p Classification module.
"""

import numpy as np
from suite2p import classification


def get_stat_iscell(data_dir_path):
    # stat with standard deviation and skew already calculated
    stat = np.load(data_dir_path.joinpath('classification', 'pre_stat.npy'), allow_pickle=True)
    expected_output = np.load(data_dir_path.joinpath('1plane1chan', 'suite2p', 'plane0', 'iscell.npy'))
    return stat, expected_output


def test_classification_output(test_ops, data_dir):
    """
    Regression test that checks to see if the main_classify function works. Only checks iscell output.
    """
    test_ops['save_path'] = test_ops['save_path0']
    default_cls_file = data_dir.parent.parent.joinpath('suite2p', 'classifiers', 'classifier.npy')
    stat, expected_output = get_stat_iscell(data_dir)
    iscell = classification.classify(test_ops['save_path'], stat, classfile=default_cls_file)
    assert np.allclose(iscell, expected_output, atol=2e-4)


def test_classifier_output(data_dir):
    """
    Regression test that checks to see if classifier works.
    """
    default_cls_file = data_dir.parent.parent.joinpath('suite2p', 'classifiers', 'classifier.npy')
    stat, expected_output = get_stat_iscell(data_dir)
    iscell = classification.Classifier(default_cls_file, keys=['npix_norm', 'compact', 'skew']).run(stat)
    # Logistic Regression has differences in tolerance due to dependence on C
    assert np.allclose(iscell, expected_output, atol=2e-4)
