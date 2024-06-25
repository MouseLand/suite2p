from pathlib import Path
import suite2p


def test_bruker(test_ops):
    test_ops['data_path'] = [Path(test_ops['data_path'][0]).joinpath('bruker')]
    test_ops['input_format'] = 'bruker'
    print(test_ops['nchannels'])
    suite2p.run_s2p(ops=test_ops)