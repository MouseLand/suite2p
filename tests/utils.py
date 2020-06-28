"""Utility functions that can be accessed in tests via the utils fixture below."""

from pathlib import Path

import numpy as np


r_tol, a_tol = 1e-6, 5e-2


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


def check_lists_of_arr_all_close(list1, list2):
    for i in range(len(list1)):
        assert np.allclose(list1[i], list2[i], rtol=r_tol, atol=a_tol)


def check_dict_dicts_all_close(first_dict, second_dict):
    for gt_dict, output_dict in zip(first_dict, second_dict):
        for k in gt_dict.keys():
            assert np.allclose(gt_dict[k], output_dict[k], rtol=r_tol, atol=a_tol)


def get_list_of_test_data(outputs_to_check, test_data_dir, nplanes, nchannels, added_tag, curr_plane):
    """
    Gets list of test_data from test data directory matching provided nplanes, nchannels, and added_tag. Returns
    all test_data for given plane number.
    """
    test_data_list = []
    for output in outputs_to_check:
        test_data_list.append(np.load(
            str(test_data_dir.joinpath('{}plane{}chan{}'.format(nplanes, nchannels, added_tag), 'suite2p',
                                       'plane{}'.format(curr_plane), "{}.npy".format(output))), allow_pickle=True
        ))
    return test_data_list


def get_list_of_output_data(outputs_to_check, output_root, curr_plane):
    """
    Gets list of output data from output_directory. Returns all data for given plane number.
    """
    output_dir = Path(output_root).joinpath("suite2p")
    output_data_list = []
    for output in outputs_to_check:
        output_data_list.append(np.load(
            str(output_dir.joinpath('plane{}'.format(curr_plane), "{}.npy".format(output))), allow_pickle=True
        ))
    return output_data_list


def check_output(output_root, outputs_to_check, test_data_dir, nplanes: int, nchannels: int, added_tag=""):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    for i in range(nplanes):
        compare_list_of_outputs(i,
                                outputs_to_check,
                                get_list_of_test_data(outputs_to_check, test_data_dir, nplanes, nchannels, added_tag, i),
                                get_list_of_output_data(outputs_to_check, output_root, i)
        )


def compare_list_of_outputs(plane_num, output_name_list, data_list_one, data_list_two):
    for i in range(len(output_name_list)):
        output = output_name_list[i]
        print("Comparing {} for plane {}".format(output, plane_num))
        # Handle cases where the elements of npy arrays are dictionaries (e.g: stat.npy)
        if output == 'stat':
            check_dict_dicts_all_close(data_list_one[i], data_list_two[i])
        else:
            assert np.allclose(data_list_one[i], data_list_two[i], rtol=r_tol, atol=a_tol)
