"""
Tests for the Suite2p Classification module.
"""

import numpy as np
from suite2p import classification


def get_stat_iscell(data_dir_path):
    # stat with standard deviation and skew already calculated
    stat = np.load(data_dir_path.joinpath('classification/pre_stat.npy'), allow_pickle=True)
    expected_output = np.load(data_dir_path.joinpath('1plane1chan/suite2p/plane0/iscell.npy'))
    return stat, expected_output


def test_classification_output(test_ops, data_dir):
    """
    Regression test that checks to see if the main_classify function works. Only checks iscell output.
    """
    test_ops['save_path'] = test_ops['save_path0']
    stat, expected_output = get_stat_iscell(data_dir)
    iscell = classification.classify(stat, classfile=classification.builtin_classfile)
    assert np.allclose(iscell, expected_output, atol=2e-4)
