"""Utility functions that can be accessed in tests via the utils fixture below. """

from typing import Iterator
from tifffile import imread
from pathlib import Path
from glob import glob
from suite2p.io import BinaryFile
from suite2p.parameters import default_settings, convert_settings_orig

import numpy as np
import json


def get_list_of_data(outputs_to_check, output_dir) -> Iterator[np.ndarray]:
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
            for gt_dict, output_dict in zip(data1, data2):
                for k in gt_dict.keys():
                    if k=='ypix' or k=='xpix' or k=='lam':
                        yield np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)
        elif output == 'iscell':  # just check the first column; are cells/noncells classified the same way?
            yield (data1[:, 0] != data2[:, 0]).sum() < 5
        elif output == 'redcell':
            yield True
        else:
            yield np.allclose(data1, data2, rtol=1e-4, atol=5e-2)

class FullPipelineTestUtils:
    """
    Utility functions specific to test_full_pipeline.py. Mostly contains settings initialization
    functions that can be used by both test_full_pipeline.py and generate_test_data.py.
    This is to ensure both the generation script and the tests use the same settings.
    """
    @staticmethod
    def initialize_settings_test1plane_1chan_with_batches(db, settings):
        db.update({
            "file_list": ["input_1500.tif"], 
            "input_format": "tif",
            "nplanes": 1, 
            "nchannels": 1,
            'keep_movie_raw': True,
        })
        settings["run"]['do_regmetrics'] = True
        settings["io"]['save_NWB'] = True
        settings["io"]["save_mat"] = True
        settings["io"]["delete_bin"] = True
        return db, settings

    @staticmethod
    def initialize_settings_test_1plane_2chan_sourcery(db, settings):
        db.update({
            'nchannels': 2,
            'file_list': ['input.tif'],
            'keep_movie_raw': True
        })
        settings["detection"]["sparsery_settings"]["spatial_scale"] = 0
        return db, settings

    @staticmethod
    def initialize_settings_test2plane_2chan_with_batches(db, settings):
        db.update({
            'file_list': ['input_1500.tif'],
            'batch_size': 200,
            'nplanes': 2,
            'nchannels': 2,
        })
        settings["registration"]["reg_tif"] = True
        settings["registration"]["reg_tif_chan2"] = True
        settings["io"]["save_mat"] = True
        settings["io"]["delete_bin"] = True
        return db, settings 

    @staticmethod
    def initialize_settings_test_mesoscan_2plane_2z(db, settings):
        mesoscan_dir = Path(db['data_path'][0]).joinpath('mesoscan')
        mesoscan_dir = Path(db['data_path'][0]).joinpath('mesoscan')
        with open(mesoscan_dir.joinpath('ops.json')) as f:
            meso_settings = json.load(f)
        db, settings, settings_in = convert_settings_orig(meso_settings, settings=settings)
        db['data_path'] = [mesoscan_dir]
        db['save_path0'] = str(mesoscan_dir)
        db['nplanes'] = 2
        settings["io"]["delete_bin"] = True
        return db, settings

    @staticmethod
    def initialize_settings_bruker(db, settings):
        bruker_dir = Path(db['data_path'][0]).joinpath('bruker')
        db['data_path'] = [bruker_dir]
        db.update({
            'input_format': 'bruker',
            'nplanes': 1,
            'nchannels': 2,
        })
        # Adjust detection parameters for bruker data (keep ROI count very low for fast NWB test)
        settings["detection"]["threshold_scaling"] = 0.3  # Moderately lower threshold
        settings["detection"]["max_iterations"] = 30
        settings["detection"]["sparsery_settings"]["spatial_scale"] = 0  # Auto-detect scale
        settings["detection"]["sparsery_settings"]["max_ROIs"] = 15  # Limit to 15 ROIs for fast test
        settings["io"]["save_mat"] = True
        settings["io"]["delete_bin"] = True
        return db, settings

class DetectionTestUtils:
    @staticmethod
    def prepare(op, input_file_name_list, dimensions):
        """
        Prepares for detection by filling out necessary settings parameters. Removes dependence on
        other modules. Creates pre_registered binary file.
        """
        # Set appropriate settings parameters
        detection_defaults = default_settings()['detection']
        op.update({
            'Lx': dimensions[0],
            'Ly': dimensions[1],
            'nframes': 500 // op['nplanes'] // op['nchannels'],
            'frames_per_file': 500 // op['nplanes'] // op['nchannels'],
            'xrange': [2, 402],
            'yrange': [2, 358],
            **detection_defaults,
            # Override detection thresholds for test data
            'threshold_scaling': 0.5,  # Lower threshold to find more ROIs
            'sparsery_settings': {
                **detection_defaults['sparsery_settings'],
                'spatial_scale': 0,  # Let algorithm auto-determine scale
                'max_ROIs': 5000,
            }
        })
        settings = []
        for plane in range(op['nplanes']):
            curr_op = op.copy()
            plane_dir = Path(op['save_path0']).joinpath(f'suite2p/plane{plane}')
            plane_dir.mkdir(exist_ok=True, parents=True)
            bin_path = str(plane_dir.joinpath('data.bin'))
            BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][0]), bin_path)
            curr_op['meanImg'] = np.reshape(
                np.load(str(input_file_name_list[plane][0])), (-1, op['Ly'], op['Lx'])
            ).mean(axis=0)
            curr_op['reg_file'] = bin_path
            if plane == 1: # Second plane result has different crop.
                curr_op['xrange'] = [1, 403]
                curr_op['yrange'] = [1, 359]
            if curr_op['nchannels'] == 2:
                bin2_path = str(plane_dir.joinpath('data_chan2.bin'))
                BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][1]), bin2_path)
                curr_op['reg_file_chan2'] = bin2_path
            curr_op['save_path'] = plane_dir
            curr_op['settings_path'] = plane_dir.joinpath('settings.npy')
            settings.append(curr_op)
        return settings

class ExtractionTestUtils:
    @staticmethod
    def prepare(op, input_file_name_list, dimensions):
        """
        Prepares for extraction by filling out necessary settings parameters. Removes dependence on
        other modules. Creates pre_registered binary file.
        """
        # Get extraction settings from default_settings
        extraction_defaults = default_settings()['extraction']

        op.update({
            'Lx': dimensions[0],
            'Ly': dimensions[1],
            'nframes': 500 // op['nplanes'] // op['nchannels'],
            'frames_per_file': 500 // op['nplanes'] // op['nchannels'],
            'xrange': [2, 402],
            'yrange': [2, 358],
            **extraction_defaults,
        })

        settings = []
        for plane in range(op['nplanes']):
            curr_op = op.copy()
            plane_dir = Path(op['save_path0']).joinpath(f'suite2p/plane{plane}')
            plane_dir.mkdir(exist_ok=True, parents=True)
            bin_path = str(plane_dir.joinpath('data.bin'))
            BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][0]), bin_path)
            curr_op['meanImg'] = np.reshape(
                np.load(str(input_file_name_list[plane][0])), (-1, op['Ly'], op['Lx'])
            ).mean(axis=0)
            curr_op['reg_file'] = bin_path
            if plane == 1: # Second plane result has different crop.
                curr_op['xrange'], curr_op['yrange'] = [[1, 403], [1, 359]]
            if curr_op['nchannels'] == 2:
                bin2_path = str(plane_dir.joinpath('data_chan2.bin'))
                BinaryFile.convert_numpy_file_to_suite2p_binary(str(input_file_name_list[plane][1]), bin2_path)
                curr_op['reg_file_chan2'] = bin2_path
            curr_op['save_path'] = plane_dir
            curr_op['settings_path'] = plane_dir.joinpath('settings.npy')
            settings.append(curr_op)
        return settings
