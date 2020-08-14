import os
from importlib_metadata import metadata

import suite2p


def test_package_version_number_matches_setuptool_version_number():
    assert suite2p.version == metadata('suite2p')['version']


def test_cli_provides_correct_package_number(capfd):
    os.system('suite2p --version')
    captured = capfd.readouterr()
    assert metadata('suite2p')['version'] in captured.out


def test_ops_suite2p_version_matches_cli_version_number(capfd):
    os.system('suite2p --version')
    captured = capfd.readouterr()
    ops_version = suite2p.default_ops()['suite2p_version']
    assert ops_version in captured.out