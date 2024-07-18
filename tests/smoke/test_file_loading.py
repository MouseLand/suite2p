from pathlib import Path
import suite2p


def test_bruker(test_settings):
    test_settings['data_path'] = [Path(test_settings['data_path'][0]).joinpath('bruker')]
    test_settings['input_format'] = 'bruker'
    print(test_settings['nchannels'])
    suite2p.run_s2p(settings=test_settings)