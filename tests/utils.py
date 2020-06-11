"""Utility functions that can be accessed in tests via the utils fixture below."""

from pathlib import Path

import numpy as np



def get_plane_dir(op, plane):
    suite_dir = Path(op['save_path0']).joinpath('suite2p')
    suite_dir.mkdir(exist_ok=True)
    plane_dir = Path(op['save_path0']).joinpath('suite2p').joinpath('plane{}'.format(plane))
    plane_dir.mkdir(exist_ok=True)
    return plane_dir


def write_data_to_binary(binary_path, data_path):
    input_data = np.load(data_path)
    with open(binary_path, 'wb') as f:
        input_data.tofile(f)
    return binary_path


def check_lists_of_arr_equal(list1, list2):
    for i in range(len(list1)):
        assert np.array_equal(list1[i], list2[i])


def check_dict_dicts_all_close(first_dict, second_dict, r_tol=1e-05, a_tol=1e-08):
    for gt_dict, output_dict in zip(first_dict, second_dict):
        for k in gt_dict.keys():
            assert np.allclose(gt_dict[k], output_dict[k], rtol=r_tol, atol=a_tol)


def check_output(output_root, outputs_to_check, test_data_dir, nplanes: int, nchannels: int):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    output_dir = Path(output_root).joinpath("suite2p")
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
                check_dict_dicts_all_close(test_data, output_data, r_tol=rtol, a_tol=atol)
            else:
                assert np.allclose(test_data, output_data, rtol=rtol, atol=atol)
