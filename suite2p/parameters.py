"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from .version import version

def default_db():
    """ default recording setup and paths used for creating binaries """
    return {
        "data_path":  [], # list of folders with tiffs or other files to process
        "look_one_level_down": False,  # whether to look in all subfolders of all data_path folders when searching for tiffs
        "input_format": "tif", # can be "tif", "h5", "nwb", "bruker", "mesoscan", "movie", or "dcimg"
        "keep_movie_raw":
                False,  # whether to keep binary file of non-registered frames. 
        "nplanes": 1,  # each tiff / file has this many planes in sequence
        "nchannels": 1,  # specify one- or two- channel recording
        "functional_chan":
                1,  # this channel is used to extract functional ROIs (1-based)
        "ignore_flyback": None,
        "subfolders": None,  # (optional) if len(data_path)==1, subfolders of db["data_path"][0] to use when look_one_level_down is set to True
        "file_list": None, # list of files to process (default is all files in data_path, only supported with one data_path folder)
        "save_path0": None,  # pathname where you'd like to store results, defaults to first item in data_path
        "fast_disk": None,  # used to store temporary binary file, defaults to save_path0
        "save_folder": "suite2p",  # directory you"d like suite2p results to be saved to (defaults to suite2p subfolder)
        "h5py_key": "data",  #key in h5py where data array is stored
        "nwb_driver": "",  # driver for nwb file (nothing if file is local)
        "nwb_series":
                 "",  # TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file
        "force_sktiff": False,  # whether or not to use scikit-image for tiff reading
        "bruker_bidirectional": False, # tiffs in 0, 1, 2, 2, 1, 0 ... order    
        "batch_size": 500, # number of frames per batch when writing binary files
    }

def default_ops():
    """ default options to run pipeline """
    return {
        # Suite2p version
        "suite2p_version": version,  #current version of suite2p used for pipeline

        "torch_device": "cuda",  # torch device using GPU ("cuda") or CPU ("cpu")
        "tau": 1.,  # this is the timescale for deconvolution
        "fs":
            10., # sampling rate per plane, e.g. 10 for standard recordings with 3 planes
        "diameter": 12.,  # this is the main parameter for cell detection
        "aspect": 1, # pixel X/Y aspect ratio
            
        # run settings
        "run": {
            "multiplane_parallel": False,  # whether or not to run on server
            "do_registration": True,  # whether to register data (2 forces re-registration)
            "do_regmetrics": True,  # whether to register data (2 forces re-registration)
            "do_detection": True,  # whether or not to run ROI detection + extraction
            "do_deconvolution": True,  # whether or not to run spike deconvolution
        },

        "io": {
            # file input/output settings
            "delete_bin": False,  # whether to delete binary file after processing
            "move_bin":
                False,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk
            "combined":
                True,  # combine multiple planes into a single result /single canvas for GUI
            "save_mat": False,  # whether to save output as matlab files
            "save_NWB": False,  # whether to save output as NWB file
         },
        
        "registration": {
            # bidirectional phase offset
            "do_bidiphase":
                False,  #whether or not to compute bidirectional phase offset (applies to 2P recordings only)
            "bidiphase":
                0,  # Bidirectional Phase offset from line scanning (set by user). Applied to all frames in recording.
            "bidi_corrected":
                False,  # Whether to do bidirectional correction during registration
            "two_step_registration":
                False,  # whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True. 
            "nimg_init": 300,  # subsampled frames for finding reference image
            "batch_size": 100,  # number of frames per batch
            "align_by_chan2": False,  # when two-channel, you can align by non-functional channel (called chan2)
            "maxregshift":
                0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
            "reg_tif": False,  # whether to save registered tiffs
            "reg_tif_chan2": False,  # whether to save channel 2 registered tiffs
            "subpixel": 10,  # precision of subpixel registration (1/subpixel steps)
            "smooth_sigma_time": 0,  # gaussian smoothing in time
            "smooth_sigma":
                1.15,  # gaussian smoothing in XY; ~1 good for 2P recordings, recommend 3-5 for 1P recordings
            "spatial_taper": 3.45, 
                # edge tapering width in pixels (may want larger for 1P recordings)
            "th_badframes":
                1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
            "norm_frames": True,  # normalize frames when detecting shifts
            # non rigid registration settings
            "nonrigid": True,  # whether to use nonrigid registration
            "block_size": [128,
                        128],  # block size to register (** keep this a multiple of 2, 3, and 5 **)
            "snr_thresh":
                1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
            "maxregshiftNR":
                5,  # maximum pixel shift allowed for nonrigid, relative to rigid
        },
        
        # cell detection settings with suite2p
        "detection": {
            "algorithm": "sparsery", # ["sparsery", "sourcery", "cellpose"],
            "denoise": False,  # whether to use denoising,
            "block_size": (64, 64), # block size for denoising
            "nbins": 5000, # max number of binned frames for cell detection
            "batch_size": 500, # number of frames per batch
            "diameter": 10, # approximate diameter of cells in pixels
            "bin_size": None, # size of bins for cell detection (default is tau * fs)
            "highpass_time": 100, # running mean subtraction across bins with a window of size "highpass_time" (use low values for 1P) - used before max_proj computation and detection
            "threshold_scaling": 1.0, # adjust the automatically determined threshold in sparsery and sourcery by this scalar multiplier
            "npix_norm_min": 0.25,
            "npix_norm_max": 3.0,
            "max_overlap":
                0.75,  # cells with more overlap than this get removed during triage, before refinement
            "soma_crop": True,  # crop dendrites for cell classification stats like compactness
            "chan2_threshold": 0.65, # minimum for detection of brightness on channel 2, if cellpose is not used
            "sparsery_settings": {"highpass_neuropil": 25,
                            "max_ROIs": 5000,
                            "spatial_scale": 0,
                            "active_percentile": 0.,},
            "sourcery_settings": {"connected": True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
                            "max_iterations": 20,
                            "smooth_masks": False
                            },
            "cellpose_settings": {"model": "cyto", # cellpose model to use
                            "img": 1, # run cellpose to get masks on 1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj
                            "highpass_spatial": 0, # highpass img before running cellpose
                            "params": {}, # parameters for cellpose
                            "model_chan2": "nuclei", # cellpose model to use for channel 2
                            "params_chan2": {} # parameters for cellpose chan2
                            }
        },
        
        # ROI extraction and deconvolution parameters
        "extraction": {
            "snr_threshold": 0.25, # snr threshold for ROIs
            "batch_size": 500, # batch size for extraction
            "neuropil_extract":
                True,  # whether or not to extract neuropil; if False, Fneu is set to zero
            "neuropil_coefficient": 0.7,
            "inner_neuropil_radius":
                2,  # number of pixels to keep between ROI and neuropil donut
            "min_neuropil_pixels": 350,  # minimum number of pixels in the neuropil
            "lam_percentile":
                50.,  # percentile of lambda within area to ignore when excluding cell pixels for neuropil extraction
            "allow_overlap":
                False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
            "circular_neuropil": False, # force neuropil_masks to be circular instead of square (slow)
            "use_builtin_classifier":
                False,  # whether or not to use built-in classifier for cell detection (overrides
            },
            
        "dcnv_preprocess": {
            "baseline": "maximin",  # baselining mode (can also choose "prctile")
            "win_baseline": 60.,  # window for maximin
            "sig_baseline": 10.,  # smoothing constant for gaussian filter
            "prctile_baseline": 8.,  # optional (whether to use a percentile baseline)
        },
        
        # classifier specified in classifier_path if set to True)
        "classifier_path": None,  # path to classifier
        "use_builtin_classifier": False,  # whether or not to use built-in classifier for cell detection (overrides user classifier)
            
    }
