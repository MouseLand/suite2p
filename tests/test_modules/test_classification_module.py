import numpy as np
from suite2p import classification


class TestSuite2pClassificationModule:
    """
    Tests for the Suite2p Classification module.
    """
    def test_classification_output(self, data_dir):
        """
        Regression test that checks the output of classification.
        """
        default_cls_file = data_dir.parent.parent.joinpath('suite2p', 'classifiers', 'classifier.npy')
        stat = np.load(data_dir.joinpath('classification', 'pre_stat.npy'), allow_pickle=True)
        iscell = classification.Classifier(default_cls_file, keys=['npix_norm', 'compact', 'skew']).run(stat)
        expected_output = np.load(data_dir.joinpath('1plane1chan', 'suite2p', 'plane0', 'iscell.npy'))
        # Logistic Regression has differences in tolerance due to dependence on C
        assert np.allclose(iscell, expected_output, atol=2e-4)
