import pytest
import suite2p
import os
import zipfile
import tempfile
import shutil
from tqdm import tqdm
from pathlib import Path
from urllib.request import urlopen
from tenacity import retry

@pytest.fixture()
def data_dir():
    """
    Initializes data input directory for tests. Downloads test_inputs and test_outputs if not present.
    """
    data_path = Path('data/')
    data_path.mkdir(exist_ok=True)
    download_cached_inputs(data_path)
    cached_outputs = data_path.joinpath('test_outputs')
    cached_outputs_url = 'https://osf.io/download/69890549bffbac83ffe2530a'
    if not os.path.exists(cached_outputs):
        extract_zip(data_path.joinpath('test_outputs.zip'), cached_outputs_url, data_path)
    return data_path


@pytest.fixture()
def test_settings(tmpdir, data_dir):
    """Initializes settings to be used for tests. Also, uses tmpdir fixture to create a unique temporary dir for each test."""
    return initialize_settings(tmpdir, data_dir)

def download_cached_inputs(data_path):
    """ Downloads test_input data if not present on machine. This function was created so it can also be used by scripts/generate_test_data.py."""
    cached_inputs = data_path.joinpath('test_inputs')
    cached_inputs_url = 'https://osf.io/download/6984d9ca54398fbd42f5575d'
    if not os.path.exists(cached_inputs):
        print('Cached inputs not found. Downloading now...')
        extract_zip(data_path.joinpath('test_inputs.zip'), cached_inputs_url, data_path)

def get_device():
    """Detect the appropriate device for suite2p processing."""
    import torch
    import platform

    if platform.system() == 'Darwin':  # macOS
        if torch.backends.mps.is_available():
            return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'

    return 'cpu'

def initialize_settings(tmpdir, data_dir):
    """Initializes settings. Used for both the test_settings function above and for generate_test_data script. This function was made to accomodate creation of settings for both pytest and non-pytest settings."""
    settings = suite2p.default_settings()
    db = suite2p.default_db()
    db.update({
        'data_path': [Path(data_dir).joinpath('test_inputs')],
        'save_path0': str(tmpdir),
    })
    settings.update(
        {
            'torch_device': get_device()
        }
    )
    return db, settings

def extract_zip(cached_file, url, data_path):
    download_url_to_file(url, cached_file)        
    with zipfile.ZipFile(cached_file,"r") as zip_ref:
        zip_ref.extractall(data_path)

#@retry
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
    print(f"\nDownloading: {url}")
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

