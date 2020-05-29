import numpy as np
from pathlib import Path
import suite2p


class TestCommonPipelineUseCases:
    """
    Class that tests common use cases for pipeline.
    """

    def check_output_gt(self, output_root, test_data_dir, nplanes: int, nchannels: int):
        """
        Helper function to check if outputs given by a test are exactly the same
        as the ground truth outputs.
        """
        output_dir = Path(output_root).joinpath("suite2p")
        outputs_to_check = ['F', 'Fneu', 'iscell', 'spks', 'stat']
        if nchannels == 2:
            outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
        for i in range(nplanes):
            for output in outputs_to_check:
                test_data = np.load(
                    str(test_data_dir.joinpath('{}plane{}chan'.format(nplanes, nchannels), 'suite2p',
                                               'plane{}'.format(i), "{}.npy".format(output))), allow_pickle=True
                )
                output_data = np.load(
                    str(output_dir.joinpath('plane{}'.format(i), "{}.npy".format(output))), allow_pickle=True
                )
                print("Comparing {} for plane {}".format(output, i))
                rtol, atol = 1e-6 , 5e-2
                # Handle cases where the elements of npy arrays are dictionaries (e.g: stat.npy)
                if output == 'stat':
                    for gt_dict, output_dict in zip(test_data, output_data):
                        for k in gt_dict.keys():
                            assert np.allclose(gt_dict[k], output_dict[k], rtol=rtol, atol=atol)
                else:
                    assert np.allclose(test_data, output_data, rtol=rtol, atol=atol)

    def test_1plane_1chan(self, setup_and_teardown, get_test_dir_path):
        """
        Tests for case with 1 plane and 1 channel
        """
        # Get ops and unique tmp_dir from fixture
        ops, tmp_dir = setup_and_teardown
        suite2p.run_s2p(ops=ops)
        # Check outputs against ground_truth files
        self.check_output_gt(tmp_dir, get_test_dir_path, ops['nplanes'], ops['nchannels'])

    def test_2plane_1chan(self, setup_and_teardown, get_test_dir_path):
        """
        Tests for case with 2 planes and 1 channel.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        suite2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, get_test_dir_path, ops['nplanes'], ops['nchannels'])

    def test_2plane_2chan(self, setup_and_teardown, get_test_dir_path):
        """
        Tests for case with 2 planes and 2 channels.
        """
        ops, tmp_dir = setup_and_teardown
        ops['nplanes'] = 2
        ops['nchannels'] = 2
        suite2p.run_s2p(ops=ops)
        self.check_output_gt(tmp_dir, get_test_dir_path, ops['nplanes'], ops['nchannels'])
