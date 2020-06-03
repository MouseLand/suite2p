import numpy as np
from pathlib import Path
from suite2p import classification


class TestSuite2pClassificationModule:
    """
    Tests for the Suite2p Classification module.
    """
    def test_classification_output(self, setup_and_teardown, get_test_dir_path):
        """
        Regression test that checks the output of classification.
        """
        op, tmp_dir = setup_and_teardown
        default_cls_file = get_test_dir_path.parent.parent.joinpath('suite2p', 'classifiers', 'classifier.npy')
        stat = np.load(get_test_dir_path.joinpath('classification', 'pre_stat.npy'), allow_pickle=True)
        iscell = classification.Classifier(default_cls_file, keys=['npix_norm', 'compact', 'skew']).run(stat)
        expected_output = np.load(get_test_dir_path.joinpath('1plane1chan', 'suite2p', 'plane0', 'iscell.npy'))
        # Logistic Regression has differences in tolerance due to depndence on C
        assert np.allclose(iscell, expected_output, atol=2e-4)
