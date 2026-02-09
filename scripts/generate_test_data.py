import os
import suite2p
import shutil
import numpy as np
import copy
import torch


from pathlib import Path
import sys
sys.path.insert(0, '../')
from conftest import initialize_settings, download_cached_inputs #Guarantees that tests and this script use the same settings
from tests.regression.utils import FullPipelineTestUtils, DetectionTestUtils, ExtractionTestUtils
from suite2p.extraction import masks

# =============================================================================
# Configuration Constants
# =============================================================================

class TestDataConfigs:
    """Test data generation configurations."""

    # Full Pipeline Test Configurations
    FULL_PIPELINE = {
        '1plane1chan1500': {
            'name': '1plane1chan1500',
            'output_dir': '1plane1chan1500',
            'description': 'Single plane, single channel, 1500 frames test data'
        },
        '2plane2chan1500': {
            'name': '2plane2chan1500',
            'output_dir': '2plane2chan1500',
            'description': 'Two planes, two channels, 1500 frames test data'
        },
        'mesoscan': {
            'name': 'mesoscan',
            'output_dir': 'mesoscan',
            'description': 'Mesoscan two planes, two ROIs test data'
        },
        'bruker': {
            'name': 'bruker',
            'output_dir': 'bruker',
            'description': 'Bruker OME-TIFF format test data'
        }
    }

    # Detection Test Configurations
    DETECTION = {
        'output_dir': 'detection',
        'expected_files': [
            'expected_detect_output_1p1c0.npy',
            'expected_detect_output_2p2c0.npy',
            'expected_detect_output_2p2c1.npy'
        ],
        'configs': {
            '1plane1chan': {
                'nplanes': 1,
                'nchannels': 1,
                'dimensions': (404, 360),
                'file_list': ['input.tif'],
                'input_files': [['detection/pre_registered.npy']],
                'output_filename': 'expected_detect_output_1p1c0.npy',
                'cleanup_files': [(0, '')]
            },
            '2plane2chan': {
                'nplanes': 2,
                'nchannels': 2,
                'dimensions': (404, 360),
                'file_list': ['input.tif'],
                'input_files': [
                    ['detection/pre_registered01.npy', 'detection/pre_registered02.npy'],
                    ['detection/pre_registered11.npy', 'detection/pre_registered12.npy']
                ],
                'output_filename': 'expected_detect_output_2p2c{}.npy',
                'cleanup_files': [(0, ''), (0, '_chan2'), (1, ''), (1, '_chan2')],
                'extra_mean_imgs': ['detection/meanImg_chan2p0.npy', 'detection/meanImg_chan2p1.npy']
            }
        }
    }

    # Classification Test Configurations
    CLASSIFICATION = {
        'output_dir': 'classification',
        'expected_files': ['expected_classify_output_1p1c0.npy']
    }

    # Extraction Test Configurations
    EXTRACTION = {
        'output_dir': 'extraction',
        'baseline_methods': ['maximin', 'constant', 'constant_prctile'],
        'subdirs': ['1plane1chan', '2plane2chan']
    }

current_dir = Path(os.getcwd())
# Assumes the input file has already been downloaded
test_input_dir_path = current_dir.parent.joinpath('data')
# Output directory where suite2p results are kept
test_data_dir_path = current_dir.parent.joinpath('data/test_outputs')

class GenerateFullPipelineTestData:
	# Full Pipeline Tests
	@staticmethod
	def _generate_test_data(db, settings, initialize_func, config_key):
		"""Generic function to generate test data for full pipeline tests."""
		test_db, test_settings = initialize_func(copy.deepcopy(db), copy.deepcopy(settings))
		output_dir = TestDataConfigs.FULL_PIPELINE[config_key]['output_dir']

		# Set save_path0 in both settings and db to the correct final location
		save_path0 = str(test_data_dir_path.joinpath(output_dir))
		test_settings['save_path0'] = save_path0
		test_db['save_path0'] = save_path0

		# Create the output directory if it doesn't exist
		Path(save_path0).mkdir(parents=True, exist_ok=True)

		# Clean any existing suite2p output
		save_path_suite2p = Path(save_path0).joinpath('suite2p')
		if save_path_suite2p.exists():
			shutil.rmtree(save_path_suite2p)
			print(f'Deleted existing suite2p output at {save_path_suite2p}')
		print(test_settings['run'])
		suite2p.run_s2p(settings=test_settings, db=test_db)

	@staticmethod
	def generate_1p1c1500_expected_data(db, settings):
		"""
		Generates expected output for test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format
		for test_full_pipeline.py
		"""
		GenerateFullPipelineTestData._generate_test_data(
			db, settings,
			FullPipelineTestUtils.initialize_settings_test1plane_1chan_with_batches,
			'1plane1chan1500'
		)

	@staticmethod
	def generate_2p2c1500_expected_data(db, settings):
		"""
		Generates expected output for test_2plane_2chan_with_batches of test_full_pipeline.py.
		"""
		GenerateFullPipelineTestData._generate_test_data(
			db, settings,
			FullPipelineTestUtils.initialize_settings_test2plane_2chan_with_batches,
			'2plane2chan1500'
		)

	@staticmethod
	def generate_2p2zmesoscan_expected_data(db, settings):
		"""
		Generates expected output for test_mesoscan_2plane_2z of test_full_pipeline.py.
		"""
		GenerateFullPipelineTestData._generate_test_data(
			db, settings,
			FullPipelineTestUtils.initialize_settings_test_mesoscan_2plane_2z,
			'mesoscan'
		)

	@staticmethod
	def generate_bruker_expected_data(db, settings):
		"""
		Generates expected output for test_nwb_round_trip[bruker] of test_io.py.
		"""
		GenerateFullPipelineTestData._generate_test_data(
			db, settings,
			FullPipelineTestUtils.initialize_settings_bruker,
			'bruker'
		)

	@staticmethod
	def generate_all_data(full_db, full_settings):
		# Expected Data for test_full_pipeline.py
		GenerateFullPipelineTestData.generate_1p1c1500_expected_data(full_db, full_settings)
		GenerateFullPipelineTestData.generate_2p2c1500_expected_data(full_db, full_settings)
		GenerateFullPipelineTestData.generate_2p2zmesoscan_expected_data(full_db, full_settings)
		GenerateFullPipelineTestData.generate_bruker_expected_data(full_db, full_settings)

class GenerateDetectionTestData:
	@staticmethod
	def _run_detection(db, input_files, dimensions, output_filename):
		settings = DetectionTestUtils.prepare(db, input_files, dimensions)

		if len(settings) == 1:
			with suite2p.io.BinaryFile(Ly=settings[0]['Ly'], Lx=settings[0]['Lx'], filename=settings[0]['reg_file']) as f_reg:
				settings[0]['neuropil_extract'] = True
				_, stat, _ = suite2p.detection.detection_wrapper(f_reg, settings=settings[0])
				cell_masks, neuropil_masks = masks.create_masks(stat, settings[0]['Ly'], settings[0]['Lx'], neuropil_extract=True)
				np.save(output_filename, {'stat': stat, 'cell_masks': cell_masks, 'neuropil_masks': neuropil_masks})
		else:
			detection_dir = Path(db['data_path'][0]).joinpath('detection')
			settings[0]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p0.npy'))
			settings[1]['meanImg_chan2'] = np.load(detection_dir.joinpath('meanImg_chan2p1.npy'))

			for i, op in enumerate(settings):
				with suite2p.io.BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
					op['neuropil_extract'] = True
					_, stat, _ = suite2p.detection.detection_wrapper(f_reg, settings=op)
					cell_masks, neuropil_masks = masks.create_masks(stat, op['Ly'], op['Lx'], neuropil_extract=True)
					np.save(output_filename % (db['nchannels'], db['nplanes'], i),
						   {'stat': stat, 'cell_masks': cell_masks, 'neuropil_masks': neuropil_masks})

			for plane_num in range(len(settings)):
				for suffix in ['', '_chan2']:
					try:
						remove_binary_file(test_data_dir_path, plane_num, suffix)
					except:
						pass

		if (test_data_dir_path / 'suite2p').exists():
			shutil.rmtree(test_data_dir_path / 'suite2p')

	@staticmethod
	def generate_detection_1plane1chan_test_data(db, settings):
		db = copy.deepcopy(db)
		db.update({'file_list': ['input.tif']})
		GenerateDetectionTestData._run_detection(
			db, [[Path(db['data_path'][0]).joinpath('detection/pre_registered.npy')]],
			(404, 360), 'expected_detect_output_1p1c0.npy'
		)

	@staticmethod
	def generate_detection_2plane2chan_test_data(db, settings):
		db = copy.deepcopy(db)
		db.update({'file_list': ['input.tif'], 'nchannels': 2, 'nplanes': 2})
		detection_dir = Path(db['data_path'][0]).joinpath('detection')
		GenerateDetectionTestData._run_detection(
			db, [
				[detection_dir.joinpath('pre_registered01.npy'), detection_dir.joinpath('pre_registered02.npy')],
				[detection_dir.joinpath('pre_registered11.npy'), detection_dir.joinpath('pre_registered12.npy')]
			], (404, 360), 'expected_detect_output_%ip%ic%i.npy'
		)

	@staticmethod
	def generate_all_data(db, settings):
		GenerateDetectionTestData.generate_detection_1plane1chan_test_data(db, settings)
		GenerateDetectionTestData.generate_detection_2plane2chan_test_data(db, settings)

		detection_dir = test_data_dir_path.joinpath(TestDataConfigs.DETECTION['output_dir'])
		detection_dir.mkdir(exist_ok=True)
		for expected_file in TestDataConfigs.DETECTION['expected_files']:
			if os.path.exists(expected_file):
				dest_file = detection_dir / expected_file
				if os.path.exists(dest_file):
					os.remove(dest_file)
				shutil.move(expected_file, detection_dir)

class GenerateClassificationTestData:
	# Classification Tests
	@staticmethod
	def generate_classification_test_data(db, settings):
		stat = np.load(test_input_dir_path.joinpath('test_inputs/classification/pre_stat.npy'), allow_pickle=True)
		iscell = suite2p.classification.classify(stat, classfile=suite2p.classification.builtin_classfile)
		output_dir = test_data_dir_path.joinpath(TestDataConfigs.CLASSIFICATION['output_dir'])
		output_file = output_dir.joinpath(TestDataConfigs.CLASSIFICATION['expected_files'][0])
		np.save(str(output_file), iscell)

	@staticmethod
	def generate_all_data(db, settings):
		classification_dir = test_data_dir_path.joinpath(TestDataConfigs.CLASSIFICATION['output_dir'])
		classification_dir.mkdir(parents=True, exist_ok=True)
		GenerateClassificationTestData.generate_classification_test_data(db, settings)

class GenerateExtractionTestData:
	# Extraction Tests
	@staticmethod
	def generate_preprocess_baseline_test_data(db, settings):
		# Relies on full pipeline test data generation being completed
		pipeline_config = TestDataConfigs.FULL_PIPELINE['1plane1chan1500']
		f = np.load(test_data_dir_path.joinpath(f"{pipeline_config['output_dir']}/suite2p/plane0/F.npy"))
		extraction_dir = test_data_dir_path.joinpath(TestDataConfigs.EXTRACTION['output_dir'])
		for bv in TestDataConfigs.EXTRACTION['baseline_methods']:
			pre_f = suite2p.extraction.preprocess(
				F=f,
				baseline=bv,
				win_baseline=settings['dcnv_preprocess']['win_baseline'],
				sig_baseline=settings['dcnv_preprocess']['sig_baseline'],
				fs=settings['fs'],
				prctile_baseline=settings['dcnv_preprocess']['prctile_baseline'],
				device=torch.device(settings['torch_device'])
			)
			np.save(str(extraction_dir.joinpath(f'{bv}_f.npy')), pre_f)

	@staticmethod
	def generate_extraction_output_1plane1chan(db, settings):
		settings = copy.deepcopy(settings)
		extraction_subdir = TestDataConfigs.EXTRACTION['subdirs'][0]  # '1plane1chan'
		extraction_dir = TestDataConfigs.EXTRACTION['output_dir']     # 'extraction'
		save_path0 = str(test_data_dir_path.joinpath(extraction_dir, extraction_subdir))

		# Create the output directory
		Path(save_path0).mkdir(parents=True, exist_ok=True)

		settings.update({
			'nplanes': 1,
			'nchannels': 1,
			'save_path0': save_path0,
		})
		settings = ExtractionTestUtils.prepare(
			settings,
			[[Path(db['data_path'][0]).joinpath('detection/pre_registered.npy')]],
			(404, 360)
		)
		op = settings[0]
		extract_input = np.load(
			str(test_data_dir_path.joinpath('detection/expected_detect_output_1p1c0.npy')),
			allow_pickle=True
		)[()]
		extract_helper(op, extract_input, 0)
		remove_binary_file(save_path0, 0, '')

	@staticmethod
	def generate_extraction_output_2plane2chan(db, settings):
		settings = copy.deepcopy(settings)
		extraction_subdir = TestDataConfigs.EXTRACTION['subdirs'][1]  # '2plane2chan'
		extraction_dir = TestDataConfigs.EXTRACTION['output_dir']     # 'extraction'
		save_path0 = str(test_data_dir_path.joinpath(extraction_dir, extraction_subdir))

		# Create the output directory
		Path(save_path0).mkdir(parents=True, exist_ok=True)

		settings.update({
			'nchannels': 2,
			'nplanes': 2,
			'save_path0': save_path0,
		})
		# Create multiple settings for multiple plane extraction
		settings = ExtractionTestUtils.prepare(
			settings,
			[
				[Path(db['data_path'][0]).joinpath('detection/pre_registered01.npy'),
				Path(db['data_path'][0]).joinpath('detection/pre_registered02.npy')],
				[Path(db['data_path'][0]).joinpath('detection/pre_registered11.npy'),
				Path(db['data_path'][0]).joinpath('detection/pre_registered12.npy')]
			]
			, (404, 360),
		)
		settings[0]['meanImg_chan2'] = np.load(Path(db['data_path'][0]).joinpath('detection/meanImg_chan2p0.npy'))
		settings[1]['meanImg_chan2'] = np.load(Path(db['data_path'][0]).joinpath('detection/meanImg_chan2p1.npy'))
		# 2 separate inputs for each plane (but use outputs of detection generate function)
		extract_inputs = [
			np.load(
				str(test_data_dir_path.joinpath('detection/expected_detect_output_2p2c0.npy')),allow_pickle=True
			)[()],
			np.load(
				str(test_data_dir_path.joinpath('detection/expected_detect_output_2p2c1.npy')),allow_pickle=True
			)[()],
		]
		for i in range(len(settings)):
			extract_helper(settings[i], extract_inputs[i], i)
			# Assumes second channel binary file is present
			remove_binary_file(save_path0, i, '')
			remove_binary_file(save_path0, i, '_chan2')

	@staticmethod
	def generate_all_data(db, settings):
		extraction_dir = test_data_dir_path.joinpath(TestDataConfigs.EXTRACTION['output_dir'])
		extraction_dir.mkdir(parents=True, exist_ok=True)
		GenerateExtractionTestData.generate_preprocess_baseline_test_data(db, settings)
		GenerateExtractionTestData.generate_extraction_output_1plane1chan(db, settings)
		GenerateExtractionTestData.generate_extraction_output_2plane2chan(db, settings)

def extract_helper(settings, extract_input, plane):
	plane_dir = Path(settings['save_path0']).joinpath(f'suite2p/plane{plane}')
	print(plane_dir)
	plane_dir.mkdir(exist_ok=True, parents=True)
	with suite2p.io.BinaryFile(Ly=settings['Ly'], Lx=settings['Lx'], filename=settings['reg_file']) as f_reg:
		# Open channel 2 file if it exists
		f_reg_chan2 = None
		if 'reg_file_chan2' in settings and settings['reg_file_chan2'] is not None:
			f_reg_chan2 = suite2p.io.BinaryFile(Ly=settings['Ly'], Lx=settings['Lx'], filename=settings['reg_file_chan2'])

		try:
			# Compute Fluorescence Extraction
			F, Fneu, F_chan2, Fneu_chan2 = suite2p.extraction.extraction_wrapper(
				extract_input['stat'],
				f_reg,
				f_reg_chan2=f_reg_chan2,
				cell_masks=extract_input['cell_masks'],
				neuropil_masks=extract_input['neuropil_masks'],
				settings=settings,
				device=torch.device(settings['torch_device'])
			)
			# Deconvolve spikes from fluorescence
			dF = F.copy() - settings["extraction"]["neuropil_coefficient"] * Fneu
			dF = suite2p.extraction.preprocess(F=dF, fs=settings["fs"], device=torch.device(settings['torch_device']), 
									  		   **settings["dcnv_preprocess"])
			spks = suite2p.extraction.oasis(F=dF, batch_size=settings["extraction"]["batch_size"], tau=settings["tau"], fs=settings["fs"])
			np.save(plane_dir.joinpath('ops.npy'), settings)
			np.save(plane_dir.joinpath('F.npy'), F)
			np.save(plane_dir.joinpath('Fneu.npy'), Fneu)
			np.save(plane_dir.joinpath('F_chan2.npy'), F_chan2)
			np.save(plane_dir.joinpath('Fneu_chan2.npy'), Fneu_chan2)
			np.save(plane_dir.joinpath('spks.npy'), spks)
		finally:
			# Close channel 2 file if it was opened
			if f_reg_chan2 is not None:
				f_reg_chan2.close()


def remove_binary_file(dir_path, plane_num, bin_file_suffix):
	os.remove(os.path.join(dir_path, 'suite2p/plane{}/data{}.bin'.format(plane_num, bin_file_suffix)))

def main():
	# Check if test_input data directory is present. Download if not.
	test_input_dir_path.mkdir(exist_ok=True)
	print('Downloading test input data if not present...')
	download_cached_inputs(test_input_dir_path)
	# Create test_data directory if necessary
	test_data_dir_path.mkdir(parents=True, exist_ok=True)
	full_db, full_settings = initialize_settings(test_data_dir_path, test_input_dir_path)
	GenerateFullPipelineTestData.generate_all_data(full_db, full_settings)
	det_db, det_settings = initialize_settings(test_data_dir_path, test_input_dir_path)
	GenerateDetectionTestData.generate_all_data(det_db, det_settings)
	class_db, class_settings = initialize_settings(test_data_dir_path, test_input_dir_path)
	GenerateClassificationTestData.generate_all_data(class_db, class_settings)
	ext_db, ext_settings = initialize_settings(test_data_dir_path, test_input_dir_path)
	GenerateExtractionTestData.generate_all_data(ext_db, ext_settings)
	return

if __name__ == '__main__':
	main()
