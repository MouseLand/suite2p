import pytest
import suite2p

from pathlib import Path


@pytest.fixture()
def data_dir():
    return Path('data/test_data')


@pytest.fixture()
def test_ops(tmpdir, data_dir):
    """Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for each test."""
    ops = suite2p.default_ops()
    ops.update(
        {
            'use_builtin_classifier': True,
            'data_path': [data_dir],
            'save_path0': str(tmpdir),
            'norm_frames': False,
            'circular_neuropil': True,
            'lam_percentile': 0.0,
            'denoise': False,
            'smooth_masks': False
        }
    )
    return ops

