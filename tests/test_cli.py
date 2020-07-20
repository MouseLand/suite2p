import os


def test_cli_help_test_appears_when_program_is_called(capfd):
    os.system('suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'parameters' in captured.out


def test_cli_help_test_appears_when_program_is_called2(capfd):
    os.system('python -m suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'parameters' in captured.out
