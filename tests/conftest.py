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


@pytest.fixture(scope="module")
def set_def_classifier(tmpdir_factory):
    """
    Replaces classifier_user.npy with suite2p default classifier (needed for regression tests). Stores user's classifier
    in temporary directory then copies it back when test module is complete.
    """
    tmp_dir = tmpdir_factory.mktemp('classifier') #  can't use tmpdir fixture in module scope fixture
    curr_cls_filepath = str(Path.home().joinpath('.suite2p').joinpath('classifiers', 'classifier_user.npy'))
    default_cls_file = str(Path('data/test_data').parent.parent.joinpath('suite2p', 'classifiers', 'classifier.npy'))
    tmp_cls_filepath = Path(tmp_dir).joinpath('classifier_user.npy')
    user_has_classifier_defined = False
    if Path(curr_cls_filepath).is_file():
        print("User has classifier defined in .suite2p.")
        shutil.copy(curr_cls_filepath, tmp_cls_filepath)  # copy curr classifier to tmp
        user_has_classifier_defined = True
        shutil.copy(default_cls_file, curr_cls_filepath)
    yield default_cls_file
    # Copy user's classifier back during teardown
    if user_has_classifier_defined:
        print("Moved back user's classifier to .suite2p.")
        shutil.copy(tmp_cls_filepath, curr_cls_filepath)
    shutil.rmtree(str(tmp_dir))
