import pytest
import shutil
import suite2p
from pathlib import Path


@pytest.fixture()
def get_test_dir_path():
    return Path(__file__).parent.parent.joinpath('data/test_data')


@pytest.fixture()
def setup_and_teardown(tmpdir, get_test_dir_path):
    """
    Initializes ops to be used for test. Also, uses tmpdir fixture to create a unique temporary dir for
    each test. Then, removes temporary directory after test is completed.
    """
    ops = suite2p.default_ops()
    ops['data_path'] = [get_test_dir_path]
    ops['save_path0'] = str(tmpdir)
    yield ops, str(tmpdir)
    tmpdir_path = Path(str(tmpdir))
    if tmpdir_path.is_dir():
        shutil.rmtree(tmpdir)
        print('Successful removal of tmp_path {}.'.format(tmpdir))