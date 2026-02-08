import os
import subprocess
from time import sleep


def test_cli_help_test_appears_when_suite2p_is_called(capfd):
    os.system('suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'options' in captured.out


def temp_test_cli_version_test_appears_when_suite2p_is_called_locally(capfd):
    os.system('python -m suite2p --version')
    captured = capfd.readouterr()
    assert 'suite2p v' in captured.out


def temp_test_cli_version_test_appears_when_suite2p_is_called(capfd):
    os.system('suite2p --version')
    captured = capfd.readouterr()
    assert 'suite2p v' in captured.out


def test_cli_help_test_appears_when_suite2p_is_called_locally(capfd):
    os.system('python -m suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'options' in captured.out


def temp_test_cli_suite2p_gui_runs_when_is_called_locally():
    s2p = subprocess.Popen(['python', '-m', 'suite2p'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sleep(2)
    assert s2p.poll() is None  # Make sure process is running
    s2p.terminate()
    output, err = s2p.communicate(b'\n')
    assert "Error" not in err.decode('ascii')
