from pathlib import Path
import suite2p


def test_bruker(test_settings):
    db, settings = test_settings  # Unpack the tuple
    db['data_path'] = [Path(db['data_path'][0]).joinpath('bruker')]
    db['input_format'] = 'bruker'
    print(db['nchannels'])
    settings['detection']['threshold_scaling'] = 0.5  # Lower threshold
    suite2p.run_s2p(settings=settings, db=db)