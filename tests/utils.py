"""Utility functions that can be accessed in tests via the utils fixture below. """

from typing import Iterator
from pathlib import Path
from tifffile import imread

import numpy as np
from glob import glob

from suite2p.io import BinaryFile

r_tol, a_tol = 1e-4, 5e-2


def get_plane_dir(save_path0: str, plane: int) -> Path:
    plane_dir = Path(save_path0).joinpath(f'suite2p/plane{plane}')
    plane_dir.mkdir(exist_ok=True, parents=True)
    return plane_dir


def get_binary_file_data(op) -> np.ndarray:
    return BinaryFile(read_filename=Path(op['save_path0'], 'suite2p/plane0/data.bin'), Ly=op['Ly'], Lx=op['Lx']).data


def write_data_to_binary(binary_path, data_path):
    input_data = np.load(data_path)
    with open(binary_path, 'wb') as f:
        input_data.tofile(f)
    return binary_path


def check_lists_of_arr_all_close(list1, list2) -> Iterator[bool]:
    for l1, l2 in zip(list1, list2):
        yield np.allclose(l1, l2, rtol=r_tol, atol=a_tol)


def check_dict_dicts_all_close(first_dict, second_dict) -> Iterator[bool]:
    for gt_dict, output_dict in zip(first_dict, second_dict):
        for k in gt_dict.keys():
            yield np.allclose(gt_dict[k], output_dict[k], rtol=r_tol, atol=a_tol)


def get_list_of_test_data(outputs_to_check, test_plane_dir):
    """
    Gets list of test_data from test data directory matching provided nplanes, nchannels, and added_tag. Returns
    all test_data for given plane number.
    """
    test_data = []
    for output in outputs_to_check:
        data_path = test_plane_dir.joinpath(f"{output}")
        if 'reg_tif' in output:
            data = np.concatenate([imread(tif) for tif in glob(str(data_path.joinpath("*.tif")))])
        else:
            data = np.load(str(data_path) + ".npy", allow_pickle=True)
        test_data.append(data)
    return test_data


def get_list_of_output_data(outputs_to_check, output_root, curr_plane):
    """
    Gets list of output data from output_directory. Returns all data for given plane number.
    """
    output_dir = Path(output_root).joinpath(f"suite2p/plane{curr_plane}")
    output_data_list = []
    for output in outputs_to_check:
        data_path = output_dir.joinpath(f"{output}")
        if 'reg_tif' in output:
            filename = np.concatenate([imread(tif) for tif in glob(str(data_path.joinpath("*.tif")))])
        else:
            filename = np.load(str(data_path) + ".npy", allow_pickle=True)
        output_data_list.append(filename)
    return output_data_list


def check_output(output_root, outputs_to_check, test_data_dir, nplanes: int) -> Iterator[bool]:
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    for i in range(nplanes):
        yield all(compare_list_of_outputs(
            outputs_to_check,
            get_list_of_test_data(outputs_to_check, test_data_dir.joinpath(f'plane{i}')),
            get_list_of_output_data(outputs_to_check, output_root, i),
        ))


def compare_list_of_outputs(output_name_list, data_list_one, data_list_two) -> Iterator[bool]:
    for output, data1, data2 in zip(output_name_list, data_list_one, data_list_two):
        if output == 'stat':  # where the elements of npy arrays are dictionaries (e.g: stat.npy)
            yield check_dict_dicts_all_close(data1, data2)
        elif output == 'iscell':  # just check the first column; are cells/noncells classified the same way?
            yield np.array_equal(data1[:, 0], data2[:, 0])
        else:
            yield np.allclose(data1, data2, rtol=r_tol, atol=a_tol)
