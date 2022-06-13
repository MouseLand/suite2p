import pytest
import suite2p
import os
from cellpose import utils
from pathlib import Path
import zipfile


@pytest.fixture()
def data_dir():
    data_path = Path('data/')
    data_path.mkdir(exist_ok=True)
    cached_file = data_path.joinpath('test_data.zip')
    if not os.path.exists(cached_file):
        url = 'https://www.suite2p.org/static/test_data/test_data.zip'
        utils.download_url_to_file(url, cached_file)        
        with zipfile.ZipFile(cached_file,"r") as zip_ref:
            zip_ref.extractall(data_path)
    return data_path.joinpath('test_data/')


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
            'denoise': False,
            'soma_crop': False
        }
    )
    return ops

