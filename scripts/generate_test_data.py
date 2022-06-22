import os
import suite2p 
import shutil
from conftest import initialize_ops #Guarantees that tests and this script use the same ops
from tests.regression.utils import FullPipelineTestUtils
test_data_dir = 'test_data'
# Assumes the input file has already been downloaded
test_input_dir_path = '/home/stringlab/Desktop/suite2p/data/test_data/'
# Output directory where suite2p results are kept
test_output_dir_path = '/home/stringlab/Desktop/suite2p/scripts/test_data'

def generate_1p1c1500_expected_data(ops):
	"""
	Generates expected output for test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format
	for test_full_pipeline.py
	"""
	test_ops = initialize_ops_test1plane_1chan_with_batches(ops.copy())
	suite2p.run_s2p(ops=test_ops)
	rename_output_dir('1plane1chan1500')

# def generate_1p2c_expected_data(ops):
# 	"""
# 	Generates expected output for test_1plane_2chan_sourcery of test_full_pipeline.py.
# 	"""
# 	test_ops = initialize_ops_test_1plane_2chan_sourcery(ops.copy())
# 	suite2p.run_s2p(ops=test_ops)
# 	rename_output_dir('1plane2chan')

def generate_2p2c1500_expected_data(ops):
	"""
	Generates expected output for test_2plane_2chan_with_batches of test_full_pipeline.py.
	"""
	test_ops = initialize_ops_test2plane_2chan_with_batches(ops.copy())
	suite2p.run_s2p(ops=test_ops)
	rename_output_dir('2plane2chan1500')

def generate_2p2zmesoscan_expected_data(ops):
	"""
	Generate expected output for test_mesoscan_2plane_2z of test_full_pipeline.py.
	"""
	test_ops = initialize_ops_test_mesoscan_2plane_2z(ops.copy())
	suite2p.run_s2p(ops=test_ops)
	rename_output_dir('mesoscan')

def generate_full_pipeline_test_data(full_ops):
	# Expected Data for test_full_pipeline.py
	generate_1p1c1500_expected_data(ops)
	# generate_1p2c_expected_data(ops)
	generate_2p2c1500_expected_data(ops)
	generate_2p2zmesoscan_expected_data(ops)

def rename_output_dir(new_dir_name):
	curr_dir_path = os.path.abspath(os.getcwd())
	if os.path.exists(os.path.join(test_output_dir_path, new_dir_name)):
		shutil.rmtree(os.path.join(test_output_dir_path, new_dir_name))
	os.rename(os.path.join(test_output_dir_path, 'suite2p'), os.path.join(test_output_dir_path, new_dir_name))

def main():
	#Create test_data directory if necessary
	if not os.path.exists(test_data_dir):
		os.makedirs(test_data_dir)
		print('Created test directory at ' + os.path.abspath(test_data_dir))
	full_ops = initialize_ops(test_data_dir, test_input_dir_path)
	generate_full_pipeline_test_data(full_ops)

	return 

if __name__ == '__main__':
	main()
