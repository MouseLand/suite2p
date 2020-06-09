from pathlib import  Path
import suite2p
from tempfile import TemporaryDirectory

ops = suite2p.default_ops()
ops['nplanes'] = 2
ops['nchannels'] = 2

data_path = Path(__file__).joinpath('../../../data/test_data')
ops['data_path'] = [str(data_path)]
ops['save_path0'] = TemporaryDirectory().name
suite2p.run_s2p(ops)
