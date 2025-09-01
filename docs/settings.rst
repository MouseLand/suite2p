Settings (settings.npy)
------------------

Suite2p can be run with different configurations using the ``db`` and ``settings`` dictionaries. The ``db`` dictionary has the file/binary settings, and the ``settings`` dictionary will describe the settings used for a particular run of the pipeline. 

file settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

db["data_path"]
	List of folders with tiffs or other files to process.
                    Default value: [] ;   
                    Min, max: (None, None) ; 
                    Type: list
                
db["look_one_level_down"]
	Whether to look in all subfolders of all data_path folders when searching for tiffs.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
db["input_format"]
	Can be ['tif', 'h5', 'nwb', 'bruker', 'mesoscan', 'movie', 'dcimg'].
                    Default value: tif ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["keep_movie_raw"]
	Whether to keep binary file of non-registered frames.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
db["nplanes"]
	Each tiff / file has this many planes in sequence.
                    Default value: 1 ;   
                    Min, max: (1, 100) ; 
                    Type: int
                
db["nchannels"]
	Specify one- or two- channel recording.
                    Default value: 1 ;   
                    Min, max: (1, 2) ; 
                    Type: int
                
db["functional_chan"]
	This channel is used to extract functional ROIs (1-based).
                    Default value: 1 ;   
                    Min, max: (1, 2) ; 
                    Type: int
                
db["ignore_flyback"]
	List of planes to not process (0-based).
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: list
                
db["subfolders"]
	If len(data_path)==1, subfolders of data_path[0] to use when look_one_level_down is set to True.
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: list
                
db["file_list"]
	List of files to process (default is all files in data_path, only supported with one data_path folder).
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: list
                
db["save_path0"]
	Directory to store results, defaults to data_path[0].
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["fast_disk"]
	Directory to store temporary binary file (recommended to be SSD), defaults to save_path0.
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["save_folder"]
	Directory within save_path0 to save results.
                    Default value: suite2p ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["h5py_key"]
	Key in h5py where data array is stored.
                    Default value: data ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["nwb_driver"]
	Driver for nwb file (nothing if file is local).
                    Default value:  ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["nwb_series"]
	TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file.
                    Default value:  ;   
                    Min, max: (None, None) ; 
                    Type: str
                
db["force_sktiff"]
	Whether or not to use scikit-image for tiff reading.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
db["bruker_bidirectional"]
	Tiffs in 0, 1, 2, 2, 1, 0 ... order.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
db["batch_size"]
	Number of frames per batch when writing binary files.
                    Default value: 500 ;   
                    Min, max: (None, None) ; 
                    Type: int
                
general settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["torch_device"]
	Torch device using GPU ('cuda') or CPU ('cpu').
                    Default value: cuda ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["tau"]
	Timescale for deconvolution and binning in seconds.
                    Default value: 1.0 ;   
                    Min, max: (0.0, 10.0) ; 
                    Type: float
                
settings["fs"]
	Sampling rate per plane.
                    Default value: 10.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["diameter"]
	ROI diameter in pixels for sourcery and cellpose detection.
                    Default value: 12.0 ;   
                    Min, max: (1.0, inf) ; 
                    Type: float
                
settings["aspect"]
	Pixel X/Y aspect ratio.
                    Default value: 1.0 ;   
                    Min, max: (0.1, 10.0) ; 
                    Type: float
                
settings["classifier_path"]
	Path to classifier file for ROIs.
                    Default value: None ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["use_builtin_classifier"]
	Whether or not to use built-in classifier for ROIs.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                


run settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["run"]["multiplane_parallel"]
	Whether or not to run each plane as a server job.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["run"]["do_registration"]
	Whether to motion register data (2 forces re-registration).
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["run"]["do_regmetrics"]
	Whether to register data (2 forces re-registration).
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["run"]["do_detection"]
	Whether or not to run ROI detection and extraction.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["run"]["do_deconvolution"]
	Whether or not to run spike deconvolution.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                


io settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["io"]["delete_bin"]
	Whether to delete binary file after processing.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["io"]["move_bin"]
	If True, and fast_disk is different than save_path, binary file is moved to save_path.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["io"]["combined"]
	Combine multiple planes after processing into a single result / single canvas for GUI.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["io"]["save_mat"]
	Whether to save output as matlab file.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["io"]["save_NWB"]
	Whether to save output as NWB file.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                


registration settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["registration"]["do_bidiphase"]
	Whether or not to compute bidirectional phase offset from recording and apply to all frames in recording (applies to 2P recordings only).
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["bidiphase"]
	Bidirectional phase offset from line scanning (set by user). Applied to all frames in recording.
                    Default value: 0.0 ;   
                    Min, max: (0, inf) ; 
                    Type: float
                
settings["registration"]["nimg_init"]
	Subsampled frames for finding reference image.
                    Default value: 300 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                
settings["registration"]["batch_size"]
	Number of frames per batch.
                    Default value: 100 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["registration"]["align_by_chan2"]
	When two-channel, you can align by non-functional channel (called chan2).
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["maxregshift"]
	Max allowed registration shift, as a fraction of frame max(width and height).
                    Default value: 0.1 ;   
                    Min, max: (0.0, 1.0) ; 
                    Type: float
                
settings["registration"]["two_step_registration"]
	Whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["reg_tif"]
	Whether to save registered tiffs.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["reg_tif_chan2"]
	Whether to save chan2 registered tiffs.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["subpixel"]
	Precision of subpixel registration (1/subpixel steps).
                    Default value: 10 ;   
                    Min, max: (1, 100) ; 
                    Type: int
                
settings["registration"]["smooth_sigma_time"]
	Gaussian smoothing in time to compute registration shifts.
                    Default value: 0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["registration"]["smooth_sigma"]
	Gaussian smoothing in XY; ~1 good for 2P recordings, 3-5 may work well for 1P recordings.
                    Default value: 1.15 ;   
                    Min, max: (0.25, inf) ; 
                    Type: float
                
settings["registration"]["spatial_taper"]
	Edge tapering width in pixels (may want larger for 1P recordings).
                    Default value: 3.45 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["registration"]["th_badframes"]
	Determines which frames to exclude when determining cropping - set it smaller to exclude more frames.
                    Default value: 1.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["registration"]["norm_frames"]
	Normalize frames when detecting shifts.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["nonrigid"]
	Whether to use nonrigid registration.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["registration"]["block_size"]
	Block size for non-rigid registration (** keep this a multiple of 2, 3, and 5 **).
                    Default value: (128, 128) ;   
                    Min, max: (None, None) ; 
                    Type: tuple
                
settings["registration"]["snr_thresh"]
	If any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing.
                    Default value: 1.2 ;   
                    Min, max: (1.0, inf) ; 
                    Type: float
                
settings["registration"]["maxregshiftNR"]
	Maximum pixel shift allowed for nonrigid, relative to rigid, may need to increase for unstable recordings.
                    Default value: 5 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                


detection settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["detection"]["algorithm"]
	Algorithm used for cell detection (sparsery, sourcery or cellpose).
                    Default value: sparsery ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["detection"]["denoise"]
	Whether to use PCA denoising for cell detection.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["detection"]["block_size"]
	Block size for denoising.
                    Default value: (64, 64) ;   
                    Min, max: (None, None) ; 
                    Type: tuple
                
settings["detection"]["nbins"]
	Max number of binned frames for cell detection (may need to reduce if reduced RAM).
                    Default value: 5000 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["detection"]["bin_size"]
	Size of bins for cell detection (default is tau * fs).
                    Default value: None ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["detection"]["highpass_time"]
	Running mean subtraction across bins with a window of size highpass_time (may want to use low values for 1P).
                    Default value: 100 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                
settings["detection"]["threshold_scaling"]
	Adjust the automatically determined threshold in sparsery and sourcery by this scalar multiplier (smaller=more cells).
                    Default value: 1.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["detection"]["npix_norm_min"]
	Minimum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix).
                    Default value: 0.25 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["detection"]["npix_norm_max"]
	Maximum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix).
                    Default value: 3.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["detection"]["max_overlap"]
	ROIs with more overlap than this fraction with other ROIs are discarded.
                    Default value: 0.75 ;   
                    Min, max: (0.0, 1.0) ; 
                    Type: float
                
settings["detection"]["soma_crop"]
	Crop dendrites from ROI to determine ROI npix_norm and compactness.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["detection"]["chan2_threshold"]
	IoU threshold between red ROI and functional ROI to define as 'redcell'
                    Default value: 0.25 ;   
                    Min, max: (0.0, 1.0) ; 
                    Type: float
                
settings["detection"]["sparsery_settings"]["highpass_neuropil"]
	Highpass filter on binned movie to subtract neuropil signal (may want smaller/larger values depending on ROI diameter).
                    Default value: 25 ;   
                    Min, max: (5, inf) ; 
                    Type: int
                
settings["detection"]["sparsery_settings"]["max_ROIs"]
	Maximum number of ROIs to detect.
                    Default value: 5000 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["detection"]["sparsery_settings"]["spatial_scale"]
	Spatial scale for cell detection (0 for auto-detect scaling, 1=6 pixels, 2=12 pixels, 3=24 pixels, 4=48 pixels).
                    Default value: 0 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                
settings["detection"]["sparsery_settings"]["active_percentile"]
	Percentile of active pixels in the movie to use for thresholding; default is zero (recommended), which instead uses threshold.
                    Default value: 0.0 ;   
                    Min, max: (0.0, 1.0) ; 
                    Type: float
                
settings["detection"]["sourcery_settings"]["connected"]
	Whether or not to keep ROIs fully connected (set to 0 for dendrites).
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["detection"]["sourcery_settings"]["max_iterations"]
	Maximum number of iterations for ROI detection.
                    Default value: 20 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["detection"]["sourcery_settings"]["smooth_masks"]
	Whether or not to smooth masks.
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["detection"]["cellpose_settings"]["cellpose_model"]
	Cellpose model name or model path to use for cell detection (cyto, nuclei, etc).
                    Default value: cpsam ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["detection"]["cellpose_settings"]["img"]
	Cellpose image to use for cell detection (1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj).
                    Default value: 1 ;   
                    Min, max: (1, 4) ; 
                    Type: int
                
settings["detection"]["cellpose_settings"]["highpass_spatial"]
	Highpass image with sigma before running cellpose.
                    Default value: 0 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                
settings["detection"]["cellpose_settings"]["params"]
	Parameters for cellpose, provided as a dict.
                    Default value: {} ;   
                    Min, max: (None, None) ; 
                    Type: dict
                
settings["detection"]["cellpose_settings"]["model_chan2"]
	Cellpose model name or model path to use for channel 2.
                    Default value: nuclei ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["detection"]["cellpose_settings"]["params_chan2"]
	Parameters for cellpose chan2, provided as a dict.
                    Default value: {} ;   
                    Min, max: (None, None) ; 
                    Type: dict
                


extraction settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["extraction"]["snr_threshold"]
	SNR threshold for ROIs.
                    Default value: 0.25 ;   
                    Min, max: (-inf, 1.0) ; 
                    Type: float
                
settings["extraction"]["batch_size"]
	Batch size for extraction.
                    Default value: 500 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["extraction"]["neuropil_extract"]
	Whether or not to extract neuropil; if False, Fneu is set to zero.
                    Default value: True ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["extraction"]["neuropil_coefficient"]
	Coefficient for neuropil subtraction.
                    Default value: 0.7 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["extraction"]["inner_neuropil_radius"]
	Number of pixels to exclude from neuropil next to ROI.
                    Default value: 2 ;   
                    Min, max: (0, inf) ; 
                    Type: int
                
settings["extraction"]["min_neuropil_pixels"]
	Minimum number of pixels in the per ROI neuropil.
                    Default value: 350 ;   
                    Min, max: (1, inf) ; 
                    Type: int
                
settings["extraction"]["lam_percentile"]
	Percentile of ROI lam weights to ignore when excluding cell pixels for neuropil extraction.
                    Default value: 50.0 ;   
                    Min, max: (0.0, 100.0) ; 
                    Type: float
                
settings["extraction"]["allow_overlap"]
	Pixels that are overlapping are thrown out (False) or used for both ROIs (True).
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                
settings["extraction"]["circular_neuropil"]
	Force neuropil_masks to be circular instead of square (slow).
                    Default value: False ;   
                    Min, max: (None, None) ; 
                    Type: bool
                


dcnv_preprocess settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

settings["dcnv_preprocess"]["baseline"]
	Method for baseline estimation ('maximin', 'prctile' or 'constant').
                    Default value: maximin ;   
                    Min, max: (None, None) ; 
                    Type: str
                
settings["dcnv_preprocess"]["win_baseline"]
	Window (in seconds) for max filter.
                    Default value: 1.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["dcnv_preprocess"]["sig_baseline"]
	Width of Gaussian filter in frames (applied to find constant or before maximin filter).
                    Default value: 1.0 ;   
                    Min, max: (0.0, inf) ; 
                    Type: float
                
settings["dcnv_preprocess"]["prctile_baseline"]
	Percentile of trace to use as baseline if using 'constant_prctile' for baseline.
                    Default value: 8.0 ;   
                    Min, max: (0.0, 100.0) ; 
                    Type: float
                
