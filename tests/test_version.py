import suite2p
from importlib_metadata import metadata


def test_package_version_number_matches_setuptool_version_number():
    assert suite2p.version == metadata('suite2p')['version']


