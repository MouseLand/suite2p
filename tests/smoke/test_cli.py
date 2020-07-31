import os


def test_cli_help_test_appears_when_suite2p_is_called(capfd):
    os.system('suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'parameters' in captured.out


def test_cli_help_test_appears_when_suite2p_is_called2(capfd):
    os.system('python -m suite2p --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'parameters' in captured.out


def test_cli_help_test_appears_when_reg_metrics_is_called(capfd):
    os.system('reg_metrics --help')
    captured = capfd.readouterr()
    assert 'Suite2p' in captured.out
    assert 'usage' in captured.out
    assert 'parameters' in captured.out


def test_cli_reg_metrics_runs_with_output(capfd, tmpdir):
    os.system(f'reg_metrics data/test_data/ --tiff_list input_1500.tif --do_registration 2 --save_path0 {tmpdir}')
    captured = capfd.readouterr()
    assert captured.out
