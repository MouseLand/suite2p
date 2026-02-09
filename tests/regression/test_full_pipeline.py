"""
Class that tests common use cases for pipeline.
"""

from suite2p import io
from pathlib import Path
import numpy as np
import suite2p, utils, json


def test_1plane_1chan_with_batches_metrics_and_exported_to_nwb_format(test_settings):
	"""
	Tests for case with 1 plane and 1 channel with multiple batches. Results are saved to nwb format
	then checked to see if it contains the necessary parts for use with GUI.
	"""
	db, settings = test_settings
	db, settings = utils.FullPipelineTestUtils.initialize_settings_test1plane_1chan_with_batches(db.copy(), settings.copy())
	suite2p.run_s2p(settings=settings, db=db)
	nplanes = db['nplanes']
	outputs_to_check = ['F', 'iscell', 'stat']
	for i in range(nplanes):
		# assert all(utils.compare_list_of_outputs(
		# 	outputs_to_check,
		# 	utils.get_list_of_data(outputs_to_check, db['data_path'][0].parent.joinpath(f"test_outputs/{nplanes}plane{db['nchannels']}chan1500/suite2p/plane{i}")),
		# 	utils.get_list_of_data(outputs_to_check, Path(db['save_path0']).joinpath(f"suite2p/plane{i}")),
		# ))
		ogts = utils.get_list_of_data(outputs_to_check, db['data_path'][0].parent.joinpath(f"test_outputs/{nplanes}plane{db['nchannels']}chan1500/suite2p/plane{i}"))
		otests = utils.get_list_of_data(outputs_to_check, Path(db['save_path0']).joinpath(f"suite2p/plane{i}"))
		for j, (oc, ogt, otest) in enumerate(zip(outputs_to_check, ogts, otests)):
			print(oc)
			assert utils.compare_list_of_outputs([oc], [ogt], [otest])
		
	# Read Nwb data and make sure it's identical to output data
	stat, nwb_settings, F, Fneu, spks, iscell, probcell, redcell, probredcell = \
		io.read_nwb(str(Path(db['save_path0']).joinpath('suite2p/ophys.nwb')))
	output_dir = Path(db['save_path0']).joinpath(f"suite2p/plane0")
	output_name_list = ['F', 'iscell', 'stat']
	data_list_one = utils.get_list_of_data(output_name_list, output_dir)
	data_list_two = [F, iscell, stat]
	for output, data1, data2 in zip(output_name_list, data_list_one, data_list_two):
		if output == 'stat':  # where the elements of npy arrays are dictionaries (e.g: stat.npy)
			for gt_dict, output_dict in zip(data1, data2):
				for k in gt_dict.keys():
					if k=='ypix' or k=='xpix':  # todo: these both are different from the original; footprint and overlap are different, std key doesn't exist in output_dict.
						assert np.allclose(gt_dict[k], output_dict[k], rtol=1e-4, atol=5e-2)
		elif output == 'iscell':  # just check the first column; are cells/noncells classified the same way?
			assert np.array_equal(data1[:, 0], data2[:, 0])
		else:
			assert np.allclose(data1, data2, rtol=1e-4, atol=5e-2)


def test_2plane_2chan_with_batches(test_settings):
	"""
	Tests for case with 2 planes and 2 channels with multiple batches.
	"""
	db, settings = test_settings
	db, settings = utils.FullPipelineTestUtils.initialize_settings_test2plane_2chan_with_batches(db.copy(), settings.copy())
	nplanes = db['nplanes']
	suite2p.run_s2p(settings=settings, db=db)

	outputs_to_check = ['F', 'iscell']
	if db['nchannels'] == 2:
		outputs_to_check.extend(['F_chan2', 'Fneu_chan2'])
	for i in range(nplanes):
		ogts = utils.get_list_of_data(outputs_to_check, db['data_path'][0].parent.joinpath(f"test_outputs/{nplanes}plane{db['nchannels']}chan1500/suite2p/plane{i}"))
		otests = utils.get_list_of_data(outputs_to_check, Path(db['save_path0']).joinpath(f"suite2p/plane{i}"))
		for j, (oc, ogt, otest) in enumerate(zip(outputs_to_check, ogts, otests)):
			print(oc)
			assert utils.compare_list_of_outputs([oc], [ogt], [otest])
		# assert all(utils.compare_list_of_outputs(
		# 	outputs_to_check,
		# 	utils.get_list_of_data(outputs_to_check, db['data_path'][0].parent.joinpath(f"test_outputs/{nplanes}plane{db['nchannels']}chan1500/suite2p/plane{i}")),
		# 	utils.get_list_of_data(outputs_to_check, Path(db['save_path0']).joinpath(f"suite2p/plane{i}")),
		# ))


# def test_mesoscan_2plane_2z(test_settings):
# 	"""
# 	Tests for case with 2 planes and 2 ROIs for a mesoscan.
# 	"""
# 	db, settings = test_settings
# 	db, settings = utils.FullPipelineTestUtils.initialize_settings_test_mesoscan_2plane_2z(db, settings)
# 	print(settings['run'])
# 	suite2p.run_s2p(settings=settings, db=db)

# 	nplanes = db['nplanes'] * db['nrois']
# 	outputs_to_check = ['F', 'iscell', 'stat']
# 	for i in range(nplanes):
# 		assert all(utils.compare_list_of_outputs(
# 			outputs_to_check,
# 			# Need additional parent for since db['data_path'][0] is data/test_inputs/mesoscan
# 			utils.get_list_of_data(outputs_to_check, db['data_path'][0].parent.parent.joinpath(f'test_outputs/mesoscan/suite2p/plane{i}')),
# 			utils.get_list_of_data(outputs_to_check, Path(db['save_path0']).joinpath(f"suite2p/plane{i}")),
# 		))
