# Db and Settings for Suite2p

Suite2p can be run with different configurations using the db and settings dictionaries. The db dictionary contains recording specific parameters, and the settings dictionary contains pipeline parameters. Here is a summary of all the parameters that the pipeline takes and their default values.

## db.npy

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `data_path` | Data path | `<class 'list'>` | `[]` | List of folders with tiffs or other files to process. |
| `look_one_level_down` | Look one level down | `<class 'bool'>` | `False` | Whether to look in all subfolders of all data_path folders when searching for tiffs. |
| `input_format` | Input format | `<class 'str'>` | `tif` | Can be ['tif', 'h5', 'nwb', 'bruker', 'movie', 'dcimg']. |
| `keep_movie_raw` | Keep movie raw | `<class 'bool'>` | `False` | Whether to keep binary file of non-registered frames. |
| `nplanes` | Number of planes | `<class 'int'>` | `1` | Each tiff / file has this many planes in sequence. |
| `nrois` | Number of ScanImage ROIs | `<class 'int'>` | `1` | Each tiff / file has this many different ROIs. |
| `nchannels` | Number of channels | `<class 'int'>` | `1` | Specify one- or two- channel recording. |
| `swap_order` | Swap the order of channels and planes for multiplexed mesoscope recordings. | `<class 'bool'>` | `False` | Swap the order of channels and planes for multiplexed mesoscope recordings. |
| `functional_chan` | Functional channel | `<class 'int'>` | `1` | This channel is used to extract functional ROIs (1-based). |
| `lines` | Lines for each Scanimage ROI | `<class 'list'>` | `None` | Line numbers for each ScanImage ROI. |
| `dy` | Y position for each Scanimage ROI | `<class 'list'>` | `None` | Y position for each ScanImage ROI. |
| `dx` | X position for each Scanimage ROI | `<class 'list'>` | `None` | X position for each ScanImage ROI. |
| `ignore_flyback` | Ignore flyback | `<class 'list'>` | `None` | List of planes to not process (0-based). |
| `subfolders` | Subfolders | `<class 'list'>` | `None` | If len(data_path)==1, subfolders of data_path[0] to use when look_one_level_down is set to True. |
| `file_list` | File list | `<class 'list'>` | `None` | List of files to process (default is all files in data_path, only supported with one data_path folder). |
| `save_path0` | save_path0 | `<class 'str'>` | `None` | Directory to store results, defaults to data_path[0]. |
| `fast_disk` | Fast disk | `<class 'str'>` | `None` | Directory to store temporary binary file (recommended to be SSD), defaults to save_path0. |
| `save_folder` | Save folder | `<class 'str'>` | `suite2p` | Directory within save_path0 to save results. |
| `h5py_key` | h5py key | `<class 'str'>` | `data` | Key in h5py where data array is stored. |
| `nwb_driver` | nwb driver | `<class 'str'>` | `` | Driver for nwb file (nothing if file is local). |
| `nwb_series` | nwb series | `<class 'str'>` | `` | TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file. |
| `force_sktiff` | Force tifffile reader | `<class 'bool'>` | `False` | Use tifffile for tiff reading instead of scanimage-tiff-reader. |
| `bruker_bidirectional` | Bruker bidirectional | `<class 'bool'>` | `False` | Tiffs in 0, 1, 2, 2, 1, 0 ... order. |
| `batch_size` | Batch size | `<class 'int'>` | `500` | Number of frames per batch when writing binary files. |

## settings.npy

### general settings

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `torch_device` | Torch device | `<class 'str'>` | `cuda` | Torch device using GPU ('cuda') or CPU ('cpu'). |
| `tau` | Ca timescale | `<class 'float'>` | `1.0` | Timescale for deconvolution and binning in seconds. |
| `fs` | Sampling frequency | `<class 'float'>` | `10.0` | Sampling rate per plane. |
| `diameter` | Diameter | `<class 'list'>` | `[12.0, 12.0]` | ROI diameter in Y and X pixels for sourcery and cellpose detection. |

### run

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `do_registration` | Do registration | `<class 'int'>` | `1` | Whether to motion register data (2 forces re-registration). |
| `do_regmetrics` | Compute reg metrics | `<class 'bool'>` | `True` | Whether or not to compute registration metrics (requires 1500 frames). |
| `do_detection` | Do ROI detection | `<class 'bool'>` | `True` | Whether or not to run ROI detection and extraction. |
| `do_deconvolution` | Do spike deconvolution | `<class 'bool'>` | `True` | Whether or not to run spike deconvolution. |
| `multiplane_parallel` | Multiplane parallel | `<class 'bool'>` | `False` | Whether or not to run each plane as a server job. |

### io

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `combined` | Combine planes | `<class 'bool'>` | `True` | Combine multiple planes after processing into a single result / single canvas for GUI. |
| `save_mat` | Save mat | `<class 'bool'>` | `False` | Whether to save output as matlab file. |
| `save_NWB` | Save NWB | `<class 'bool'>` | `False` | Whether to save output as NWB file. |
| `save_ops_orig` | Save ops orig | `<class 'bool'>` | `True` | Whether to save db, settings, reg_outputs, detection_outputs into ops.npy. |
| `delete_bin` | Delete binary | `<class 'bool'>` | `False` | Whether to delete binary file after processing. |
| `move_bin` | Move binary | `<class 'bool'>` | `False` | If True, and fast_disk is different than save_path, binary file is moved to save_path. |

### registration

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `align_by_chan2` | Align by chan2 (non-func) | `<class 'bool'>` | `False` | When two-channel, you can align by non-functional channel (called chan2). |
| `nimg_init` | # of frames for refImg | `<class 'int'>` | `400` | Number of subsampled frames for finding reference image - choose more if reference image is poor. |
| `maxregshift` | Max registration shift | `<class 'float'>` | `0.1` | Max allowed registration shift, as a fraction of frame max(width and height). |
| `do_bidiphase` | Compute bidiphase offset | `<class 'bool'>` | `False` | Whether or not to compute bidirectional phase offset from recording and apply to all frames in recording (applies to 2P recordings only). |
| `bidiphase` | Bidiphase offset | `<class 'float'>` | `0.0` | Bidirectional phase offset from line scanning (set by user). Applied to all frames in recording. |
| `batch_size` | # of frames per batch | `<class 'int'>` | `100` | Number of frames per batch - choose fewer if using GPU and running out of memory. |
| `nonrigid` | Use nonrigid registration | `<class 'bool'>` | `True` | Whether to use nonrigid registration. |
| `maxregshiftNR` | Nonrigid max pixel shift | `<class 'int'>` | `5` | Maximum pixel shift allowed for nonrigid, relative to rigid, may need to increase value for unstable recordings. |
| `block_size` | Nonrigid block size | `<class 'tuple'>` | `(128, 128)` | Block size for non-rigid registration (** keep this a multiple of 2, 3, and/or 5 **). |
| `smooth_sigma_time` | Time smoothing | `<class 'float'>` | `0` | Gaussian smoothing in time to compute registration shifts (may be necessary with low SNR). |
| `smooth_sigma` | Smoothing in XY | `<class 'float'>` | `1.15` | Gaussian smoothing in XY; ~1 good for 2P recordings, 3-5 may work well for 1P recordings. |
| `spatial_taper` | Edge tapering width | `<class 'float'>` | `3.45` | Edge tapering width in pixels, may want larger for 1P recordings. |
| `th_badframes` | Bad frame threshold | `<class 'float'>` | `1.0` | Determines which frames to exclude when determining cropping - set it smaller to exclude more frames, particularly if crop seems small. |
| `norm_frames` | Normalize frames | `<class 'bool'>` | `True` | Normalize frames when detecting shifts. |
| `snr_thresh` | Nonrigid SNR threshold | `<class 'float'>` | `1.2` | If any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing. |
| `subpixel` | Nonrigid subpixel reg | `<class 'int'>` | `10` | Precision of subpixel registration for nonrigid (1/subpixel steps). |
| `two_step_registration` | Run registration twice | `<class 'bool'>` | `False` | Whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True. |
| `reg_tif` | Save registered tiffs | `<class 'bool'>` | `False` | Whether to save registered tiffs. |
| `reg_tif_chan2` | Save chan2 registered tiffs | `<class 'bool'>` | `False` | Whether to save chan2 registered tiffs. |

### detection

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `algorithm` | Detection algorithm | `<class 'str'>` | `sparsery` | Algorithm used for cell detection ['sparsery', 'sourcery', 'cellpose']. |
| `denoise` | Denoise | `<class 'bool'>` | `False` | Whether to use PCA denoising for cell detection. |
| `block_size` | Denoise block size | `<class 'tuple'>` | `(64, 64)` | Block size for denoising. |
| `nbins` | Max binned frames | `<class 'int'>` | `5000` | Max number of binned frames for cell detection (may need to reduce if reduced RAM). |
| `bin_size` | Bin size | `<class 'int'>` | `None` | Size of bins for cell detection (default is tau * fs). |
| `highpass_time` | Highpass time | `<class 'int'>` | `100` | Running mean subtraction across bins with a window of size highpass_time (may want to use low values for 1P). |
| `threshold_scaling` | Threshold scaling | `<class 'float'>` | `1.0` | Adjust the automatically determined threshold in sparsery and sourcery by this scalar multiplier - set it smaller to find more cells. |
| `npix_norm_min` | Min npix norm | `<class 'float'>` | `0.0` | Minimum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix). |
| `npix_norm_max` | Max npix norm | `<class 'float'>` | `100` | Maximum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix). |
| `max_overlap` | Max overlap | `<class 'float'>` | `0.75` | ROIs with more overlap than this fraction with other ROIs are discarded. |
| `soma_crop` | Soma crop | `<class 'bool'>` | `True` | Crop dendrites from ROI to determine ROI npix_norm and compactness. |
| `chan2_threshold` | Chan2 threshold | `<class 'float'>` | `0.25` | IoU threshold between anatomical ROI and functional ROI to define as 'redcell' |
| `cellpose_chan2` | Cellpose chan2 | `<class 'bool'>` | `False` | Use Cellpose to detect ROIs in anatomical channel and overlap with functional ROIs |
| `sparsery_settings` | N/A | `N/A` | `N/A` | N/A |
| `sourcery_settings` | N/A | `N/A` | `N/A` | N/A |
| `cellpose_settings` | N/A | `N/A` | `N/A` | N/A |

### classification

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `classifier_path` | Classifier path | `<class 'str'>` | `None` | Path to classifier file for ROIs (default is ~/.suite2p/classifiers/classifier_user.npy). |
| `use_builtin_classifier` | Use built-in classifier | `<class 'bool'>` | `False` | Use built-in classifier (classifier.npy) instead of user classifier (classifier_user.npy) for ROIs. |
| `preclassify` | Pre-classify | `<class 'float'>` | `0.0` | Remove ROIs with classifier probability below preclassify before extraction to minimize overlaps |

### extraction

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `snr_threshold` | SNR threshold | `<class 'float'>` | `0.0` | SNR threshold for ROIs. |
| `batch_size` | Batch size | `<class 'int'>` | `500` | Batch size for extraction. |
| `neuropil_extract` | Extract neuropil | `<class 'bool'>` | `True` | Whether or not to extract neuropil; if False, Fneu is set to zero. |
| `neuropil_coefficient` | Neuropil coefficient | `<class 'float'>` | `0.7` | Coefficient for neuropil subtraction. |
| `inner_neuropil_radius` | Inner neuropil radius | `<class 'int'>` | `2` | Number of pixels to exclude from neuropil next to ROI. |
| `min_neuropil_pixels` | Min neuropil pixels | `<class 'int'>` | `350` | Minimum number of pixels in the per ROI neuropil. |
| `lam_percentile` | Lambda percentile | `<class 'float'>` | `50.0` | Percentile of ROI lam weights to ignore when excluding cell pixels for neuropil extraction. |
| `allow_overlap` | Allow overlap | `<class 'bool'>` | `False` | Pixels that are overlapping are thrown out (False) or used for both ROIs (True). |
| `circular_neuropil` | Circular neuropil | `<class 'bool'>` | `False` | Force neuropil_masks to be circular instead of square (slow). |

### dcnv preprocess

| Key | GUI Name | Type | Default | Description |
|---|---|---|---|---|
| `baseline` | Baseline type | `<class 'str'>` | `maximin` | Method for baseline estimation ['maximin', 'prctile', 'constant']. |
| `win_baseline` | Baseline window | `<class 'float'>` | `60.0` | Window (in seconds) for max filter. |
| `sig_baseline` | Baseline sigma | `<class 'float'>` | `10.0` | Width of Gaussian filter in frames (applied to find constant or before maximin filter). |
| `prctile_baseline` | Baseline percentile | `<class 'float'>` | `8.0` | Percentile of trace to use as baseline if using 'prctile' for baseline. |

