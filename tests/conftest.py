import pytest
import suite2p
import shutil

from pathlib import Path


@pytest.fixture()
def data_dir():
    return Path('data/test_data')


@pytest.fixture()
def test_ops(tmpdir, data_dir):
    """Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for each test."""
    ops = suite2p.default_ops()
    ops['data_path'] = [data_dir]
    ops['save_path0'] = str(tmpdir)
    return ops

