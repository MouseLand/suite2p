import time
import numpy as np
from pathlib import Path
from typing import Dict, Any

from . import sourcery, sparsedetect, chan2detect, utils
from .stats import roi_stats
from .denoise import pca_denoise
from ..io.binary import BinaryFile
from ..classification import classify, user_classfile
from .. import default_ops

def detect(ops, classfile=None):
	
	t0 = time.time()
	bin_size = int(max(1, ops['nframes'] // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
	print('Binning movie in chunks of length %2.2d' % bin_size)
	with BinaryFile(read_filename=ops['reg_file'], Ly=ops['Ly'], Lx=ops['Lx']) as f:
		mov = f.bin_movie(
			bin_size=bin_size,
			bad_frames=ops.get('badframes'),
			y_range=ops['yrange'],
			x_range=ops['xrange'],
		)
		print('Binned movie [%d,%d,%d] in %0.2f sec.' % (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
	
		ops, stat = detection_wrapper(f, mov=mov, ops=ops, classfile=classfile)

	return ops, stat

def bin_movie(f_reg, bin_size, yrange=None, xrange=None, badframes=None):
	""" bin registered movie """
	n_frames = f_reg.shape[0]
	good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
	batch_size = min(good_frames.sum(), 500)
	Lyc = yrange[1] - yrange[0]
	Lxc = xrange[1] - xrange[0]
	# Need to adjust binned movie size after calculating bad frames
	num_good_frames = good_frames.sum()
	mov = np.zeros((num_good_frames//bin_size, Lyc, Lxc), np.float32)
	ik = 0
	t0 = time.time()
	for k in np.arange(0, n_frames, batch_size):
		data = f_reg[k : min(k + batch_size, n_frames)]

		# exclude badframes
		good_indices = good_frames[k : min(k + batch_size, n_frames)]
		if good_indices.mean() > 0.5:
			data = data[good_indices]

		# crop to valid region
		if yrange is not None and xrange is not None:
			data = data[:, slice(*yrange), slice(*xrange)]

		# bin in time
		if data.shape[0] > bin_size:
			n_d = data.shape[0]
			data = data[:(n_d // bin_size) * bin_size]
			data = data.reshape(-1, bin_size, Lyc, Lxc).astype(np.float32).mean(axis=1)
		n_bins = data.shape[0]
		mov[ik : ik + n_bins] = data
		ik += n_bins

	print('Binned movie of size [%d,%d,%d] created in %0.2f sec.' % (mov.shape[0], mov.shape[1], mov.shape[2], time.time() - t0))
	return mov


def detection_wrapper(f_reg, mov=None, yrange=None, xrange=None, 
					  ops=default_ops(), classfile=None):
	"""
	Main detection function. 

	Identifies ROIs. 

	Parameters
	----------------

	f_reg : np.ndarray or io.BinaryRWFile,
		n_frames x Ly x Lx

	mov : ndarray (t x Lyc x Lxc)
			binned movie

	yrange : list of length 2
		Range of pixels along the y-axis of mov the detection module will be run on 
	
	xrange : list of length 2
		Range of pixels along the x-axis of mov the detection module will be run on 

	ops : dictionary or list of dicts

	classfile: string (optional, default None)
		path to saved classifier

	Returns
	----------------

	ops : dictionary or list of dicts
		
	stat : dictionary 'ypix', 'xpix', 'lam'
		Dictionary containing statistics for ROIs


	"""
	n_frames, Ly, Lx = f_reg.shape
	yrange = ops.get('yrange', [0, Ly]) if yrange is None else yrange
	xrange = ops.get('xrange', [0, Lx]) if xrange is None else xrange
	
	if mov is None:
		bin_size = int(max(1, n_frames // ops['nbinned'], np.round(ops['tau'] * ops['fs'])))
		print('Binning movie in chunks of length %2.2d' % bin_size)
		mov = bin_movie(f_reg, bin_size, yrange=yrange, 
						xrange=xrange, badframes=ops.get('badframes', None))
	else:    
		if mov.shape[1] != yrange[-1] - yrange[0]:
			raise ValueError('mov.shape[1] is not same size as yrange')
		elif mov.shape[2] != xrange[-1] - xrange[0]:
			raise ValueError('mov.shape[2] is not same size as xrange')
		
	if ops.get('inverted_activity', False):
		mov -= mov.min()
		mov *= -1
		mov -= mov.min()

	if ops.get('denoise', 1):
		mov = pca_denoise(mov, block_size=[ops['block_size'][0]//2, ops['block_size'][1]//2],
							n_comps_frac = 0.5)

	if ops.get('anatomical_only', 0):
		try:
			from . import anatomical
			CELLPOSE_INSTALLED = True
		except Exception as e:
			print('Warning: cellpose did not import')
			print(e)
			print('cannot use anatomical mode, but otherwise suite2p will run normally')
			CELLPOSE_INSTALLED = False
		if not CELLPOSE_INSTALLED:
			print('~~~ tried to import cellpose to run anatomical but failed, install with: ~~~')
			print('$ pip install cellpose')
		else:
			print('>>>> CELLPOSE finding masks in ' + ['max_proj / mean_img', 'mean_img', 'enhanced_mean_img', 'max_proj'][int(ops['anatomical_only'])-1])
			stat = anatomical.select_rois(
						ops=ops,
						mov=mov,
						diameter=ops.get('diameter', None))
	else:            
		stat = select_rois(
			ops=ops,
			mov=mov,
			sparse_mode=ops['sparse_mode'],
			classfile=classfile,
		)

	ymin = int(yrange[0])
	xmin = int(xrange[0])
	if len(stat) > 0:
		for s in stat:
			s['ypix'] += ymin
			s['xpix'] += xmin
			s['med'][0] += ymin
			s['med'][1] += xmin    
		
		if ops['preclassify'] > 0:
			if classfile is None:
				print(f'NOTE: Applying user classifier at {str(user_classfile)}')
				classfile = user_classfile

			stat =  roi_stats(stat, Ly, Lx, aspect=ops.get('aspect', None), 
							diameter=ops.get('diameter', None), do_crop=ops.get('soma_crop', 1))
			if len(stat) == 0:
				iscell = np.zeros((0, 2))
			else:
				iscell = classify(stat=stat, classfile=classfile)
			np.save(Path(ops['save_path']).joinpath('iscell.npy'), iscell)
			ic = (iscell[:,0]>ops['preclassify']).flatten().astype('bool')
			stat = stat[ic]
			print('Preclassify threshold %0.2f, %d ROIs removed' % (ops['preclassify'], (~ic).sum()))
		
		stat = roi_stats(stat, Ly, Lx, aspect=ops.get('aspect', None), 
						diameter=ops.get('diameter', None), 
						max_overlap=ops['max_overlap'], 
						do_crop=ops.get('soma_crop', 1))
		print('After removing overlaps, %d ROIs remain' % (len(stat)))

	# if second channel, detect bright cells in second channel
	if 'meanImg_chan2' in ops:
		if 'chan2_thres' not in ops:
			ops['chan2_thres'] = 0.65
		ops, redcell = chan2detect.detect(ops, stat)
		np.save(Path(ops['save_path']).joinpath('redcell.npy'), redcell)

	return ops, stat

def select_rois(ops: Dict[str, Any], mov: np.ndarray, 
				sparse_mode: bool = True,
				classfile: Path = None):
	
	t0 = time.time()
	if sparse_mode:
		ops.update({'Lyc': mov.shape[1], 'Lxc': mov.shape[2]})
		new_ops, stat = sparsedetect.sparsery(
			mov=mov,
			high_pass=ops['high_pass'],
			neuropil_high_pass=ops['spatial_hp_detect'],
			batch_size=ops['batch_size'],
			spatial_scale=ops['spatial_scale'],
			threshold_scaling=ops['threshold_scaling'],
			max_iterations=250 * ops['max_iterations'],
			percentile=ops.get('active_percentile', 0.0),
		)
		ops.update(new_ops)
	else:
		ops, stat = sourcery.sourcery(mov=mov, ops=ops)

	print('Detected %d ROIs, %0.2f sec' % (len(stat), time.time() - t0))
	stat = np.array(stat)
	
	if len(stat)==0:
		raise ValueError("no ROIs were found -- check registered binary and maybe change spatial scale")
		
	# add ROI stat to stat
	#stat = roi_stats(stat, dy, dx, Ly, Lx, max_overlap=max_overlap, do_crop=do_crop)

	return stat

