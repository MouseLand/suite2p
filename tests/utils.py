"""Utility functions that can be accessed in tests via the utils fixture below. """

from typing import Iterator
from tifffile import imread

import numpy as np
from glob import glob

r_tol, a_tol = 1e-4, 5e-2

def check_dict_dicts_all_close(first_dict, second_dict) -> Iterator[bool]:
    for gt_dict, output_dict in zip(first_dict, second_dict):
        for k in gt_dict.keys():
            yield np.allclose(gt_dict[k], output_dict[k], rtol=r_tol, atol=a_tol)


def get_list_of_data(outputs_to_check, output_dir):
    """Gets list of output data from output_directory."""
    for output in outputs_to_check:
        data_path = output_dir.joinpath(f"{output}")
        if 'reg_tif' in output:
            yield np.concatenate([imread(tif) for tif in glob(str(data_path.joinpath("*.tif")))])
        else:
            yield np.load(str(data_path) + ".npy", allow_pickle=True)


def compare_list_of_outputs(output_name_list, data_list_one, data_list_two) -> Iterator[bool]:
    for output, data1, data2 in zip(output_name_list, data_list_one, data_list_two):
        if output == 'stat':  # where the elements of npy arrays are dictionaries (e.g: stat.npy)
            yield check_dict_dicts_all_close(data1, data2)
        elif output == 'iscell':  # just check the first column; are cells/noncells classified the same way?
            yield np.array_equal(data1[:, 0], data2[:, 0])
        else:
            yield np.allclose(data1, data2, rtol=r_tol, atol=a_tol)
