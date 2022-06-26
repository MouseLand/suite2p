import pytest
import suite2p
import os
import zipfile
import tempfile
import shutil
from tqdm import tqdm
from pathlib import Path
from urllib.request import urlopen

@pytest.fixture()
def data_dir():
    data_path = Path('data/')
    data_path.mkdir(exist_ok=True)
    cached_file = data_path.joinpath('test_data.zip')
    if not os.path.exists(cached_file):
        url = 'https://www.suite2p.org/static/test_data/test_data.zip'
        download_url_to_file(url, cached_file)        
        with zipfile.ZipFile(cached_file,"r") as zip_ref:
            zip_ref.extractall(data_path)
    return data_path.joinpath('test_data/')


@pytest.fixture()
def test_ops(tmpdir, data_dir):
    """Initializes ops to be used for tests. Also, uses tmpdir fixture to create a unique temporary dir for each test."""
    return initialize_ops(tmpdir, data_dir)

def initialize_ops(tmpdir, data_dir):
    """Initializes ops. Used for both the test_ops function above and for generate_test_data script. This function was made to accomodate creation of ops for both pytest and non-pytest settings."""
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

def download_url_to_file(url, dst, progress=True):
    r"""Download object at the given URL to a local path.
            Thanks to torch, slightly modified
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    """
    file_size = None
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    # We deliberately save it in a temp file and move it after
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)
    try:
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

