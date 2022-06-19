"""
Utility functions specific to test_full_pipeline.py. Mostly contains ops initialization
functions that can be used by both test_full_pipeline.py and generate_test_data.py.
This is to ensure both the generation script and the tests use the same ops.
"""
import json 
from pathlib import Path

def initialize_ops_test1plane_1chan_with_batches(ops):
	ops.update({
		'tiff_list': ['input_1500.tif'],
		'do_regmetrics': True,
		'save_NWB': True,
		'save_mat': True,
		'keep_movie_raw': True
	})
	return ops

def initialize_ops_test_1plane_2chan_sourcery(ops):
	ops.update({
		'nchannels': 2,
		'sparse_mode': 0,
		'tiff_list': ['input.tif'],
		'keep_movie_raw': True
	})
	return ops

def initialize_ops_test2plane_2chan_with_batches(ops):
	ops.update({
		'tiff_list': ['input_1500.tif'],
		'batch_size': 200,
		'nplanes': 2,
		'nchannels': 2,
		'reg_tif': True,
		'reg_tif_chan2': True,
		'save_mat': True
	})
	return ops 

def initialize_ops_test_mesoscan_2plane_2z(ops):
	with open('data/test_data/mesoscan/ops.json') as f:
		meso_ops = json.load(f)
	ops['data_path'] = [Path(ops['data_path'][0]).joinpath('mesoscan')]
	for key in meso_ops.keys():
		if key not in ['data_path', 'save_path0', 'do_registration', 'roidetect']:
			ops[key] = meso_ops[key]
	ops['delete_bin'] = False
	return ops