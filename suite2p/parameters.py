"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from .version import version
from pathlib import Path
import numpy as np

# Format for parameter specification:
# parameter: {
#     "gui_name": text displayed next to edit box in GUI
#     "type": callable datatype for this parameter, like int or float.
#     "min": minimum value allowed (inclusive).
#     "max": maximum value allowed (inclusive).
#     "default": default value used by gui and API
#     "description": Explanation of parameter's use. Populates parameter help
#                    in GUI.
# }

### recording setup and paths for creating binaries

SETTINGS_FOLDER = Path.home() / ".suite2p" / "settings" 
SETTINGS_FOLDER.mkdir(exist_ok=True, parents=True)

DB = {
        "data_path": {
            "gui_name": "Data path",
            "type": list,
            "min": None,
            "max": None,
            "default": [],
            "description": "List of folders with tiffs or other files to process.",
        },
        "look_one_level_down": {
            "gui_name": "Look one level down",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to look in all subfolders of all data_path folders when searching for tiffs.",
        },
        "input_format": {
            "gui_name": "Input format",
            "type": str,
            "min": None,
            "max": None,
            "default": "tif",
            "description": "Can be ['tif', 'h5', 'nwb', 'bruker', 'movie', 'dcimg'].",
        },
        "keep_movie_raw": {
            "gui_name": "Keep movie raw",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to keep binary file of non-registered frames.",
        },
        "nplanes": {
            "gui_name": "Number of planes",
            "type": int,
            "min": 1,
            "max": 100,
            "default": 1,
            "description": "Each tiff / file has this many planes in sequence.",
        },
        "nrois": {
            "gui_name": "Number of ScanImage ROIs",
            "type": int,
            "min": 1,
            "max": 100,
            "default": 1,
            "description": "Each tiff / file has this many different ROIs.",
        },
        "nchannels": {
            "gui_name": "Number of channels",
            "type": int,
            "min": 1,
            "max": 2,
            "default": 1,
            "description": "Specify one- or two- channel recording.",
        },
        "swap_order": {
            "gui_name": "Swap the order of channels and planes for multiplexed mesoscope recordings.",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Swap the order of channels and planes for multiplexed mesoscope recordings.",
        },
        "functional_chan": {
            "gui_name": "Functional channel",
            "type": int,
            "min": 1,
            "max": 2,
            "default": 1,
            "description": "This channel is used to extract functional ROIs (1-based).",
        },
        "lines": {
            "gui_name": "Lines for each Scanimage ROI",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "Line numbers for each ScanImage ROI.",
        },
        "dy": {
            "gui_name": "Y position for each Scanimage ROI",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "Y position for each ScanImage ROI.",
        },
        "dx": {
            "gui_name": "X position for each Scanimage ROI",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "X position for each ScanImage ROI.",
        },
        "ignore_flyback": {
            "gui_name": "Ignore flyback",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "List of planes to not process (0-based).",
        },
        "subfolders": {
            "gui_name": "Subfolders",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "If len(data_path)==1, subfolders of data_path[0] to use when look_one_level_down is set to True.",
        },
        "file_list": {
            "gui_name": "File list",
            "type": list,
            "min": None,
            "max": None,
            "default": None,
            "description": "List of files to process (default is all files in data_path, only supported with one data_path folder).",
        },
        "save_path0": {
            "gui_name": "save_path0",
            "type": str,
            "min": None,
            "max": None,
            "default": None,
            "description": "Directory to store results, defaults to data_path[0].",
        },
        "fast_disk": {
            "gui_name": "Fast disk",
            "type": str,
            "min": None,
            "max": None,
            "default": None,
            "description": "Directory to store temporary binary file (recommended to be SSD), defaults to save_path0.",
        },
        "save_folder": {
            "gui_name": "Save folder",
            "type": str,
            "min": None,
            "max": None,
            "default": "suite2p",
            "description": "Directory within save_path0 to save results.",
        },
        "h5py_key": {
            "gui_name": "h5py key",
            "type": str,
            "min": None,
            "max": None,
            "default": "data",
            "description": "Key in h5py where data array is stored.",
        },
        "nwb_driver": {
            "gui_name": "nwb driver",
            "type": str,
            "min": None,
            "max": None,
            "default": "",
            "description": "Driver for nwb file (nothing if file is local).",
        },
        "nwb_series": {
            "gui_name": "nwb series",
            "type": str,
            "min": None,
            "max": None,
            "default": "",
            "description": "TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file.",
        },
        "force_sktiff": {
            "gui_name": "Force tifffile reader",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Use tifffile for tiff reading instead of scanimage-tiff-reader.",
        },
        "bruker_bidirectional": {
            "gui_name": "Bruker bidirectional",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Tiffs in 0, 1, 2, 2, 1, 0 ... order.",
        },
        "batch_size": {
            "gui_name": "Batch size",
            "type": int,
            "min": None,
            "max": None,
            "default": 500,
            "description": "Number of frames per batch when writing binary files.",
        },
    }

### options for running the pipeline
SETTINGS = {
    "torch_device": {
        "gui_name": "Torch device",
        "type": str,
        "min": None,
        "max": None,
        "default": "cuda",
        "description": "Torch device using GPU ('cuda') or CPU ('cpu').",
    },
    "tau": {
        "gui_name": "Ca timescale",
        "type": float,
        "min": 0.,
        "max": 10.,
        "default": 1.,
        "description": "Timescale for deconvolution and binning in seconds.",
    },
    "fs": {
        "gui_name": "Sampling frequency",
        "type": float,
        "min": 0.01,
        "max": np.inf,
        "default": 10.,
        "description": "Sampling rate per plane.",
    },
    "diameter": {
        "gui_name": "Diameter",
        "type": list,
        "min": 1.,
        "max": np.inf,
        "default": [12., 12.],
        "description": "ROI diameter in Y and X pixels for sourcery and cellpose detection.",
    },  
    "run": {
        "do_registration": {
            "gui_name": "Do registration",
            "type": int,
            "min": 0,
            "max": 2,
            "default": 1,
            "description": "Whether to motion register data (2 forces re-registration).",
        },
        "do_regmetrics": {
            "gui_name": "Compute reg metrics",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether or not to compute registration metrics (requires 1500 frames).",
        },
        "do_detection": {
            "gui_name": "Do ROI detection",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether or not to run ROI detection and extraction.",
        },
        "do_deconvolution": {
            "gui_name": "Do spike deconvolution",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether or not to run spike deconvolution.",
        },
        "multiplane_parallel": {
            "gui_name": "Multiplane parallel",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether or not to run each plane as a server job.",
        },
    },
    "io": {
        "combined": {
            "gui_name": "Combine planes",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Combine multiple planes after processing into a single result / single canvas for GUI.",
        },
        "save_mat": {
            "gui_name": "Save mat",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to save output as matlab file.",
        },
        "save_NWB": {
            "gui_name": "Save NWB",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to save output as NWB file.",
        },
        "save_ops_orig": {
            "gui_name": "Save ops orig",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether to save db, settings, reg_outputs, detection_outputs into ops.npy.",
        },
        "delete_bin": {
            "gui_name": "Delete binary",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to delete binary file after processing.",
        },
        "move_bin": {
            "gui_name": "Move binary",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "If True, and fast_disk is different than save_path, binary file is moved to save_path.",
        },
    },
    "registration": {
        "align_by_chan2": {
            "gui_name": "Align by chan2 (non-func)",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "When two-channel, you can align by non-functional channel (called chan2).",
        },
        "nimg_init": {
            "gui_name": "# of frames for refImg",
            "type": int,
            "min": 0,
            "max": np.inf,
            "default": 400,
            "description": "Number of subsampled frames for finding reference image - choose more if reference image is poor.",
        },
        "maxregshift": {
            "gui_name": "Max registration shift",
            "type": float,
            "min": 0.,
            "max": 1.,
            "default": 0.1,
            "description": "Max allowed registration shift, as a fraction of frame max(width and height).",
        },
        "do_bidiphase": {
            "gui_name": "Compute bidiphase offset",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether or not to compute bidirectional phase offset from recording and apply to all frames in recording (applies to 2P recordings only).",
        },
        "bidiphase": {
            "gui_name": "Bidiphase offset",
            "type": float,
            "min": 0,
            "max": np.inf,
            "default": 0.,
            "description": "Bidirectional phase offset from line scanning (set by user). Applied to all frames in recording.",
        },
        "batch_size": {
            "gui_name": "# of frames per batch",
            "type": int,
            "min": 1,
            "max": np.inf,
            "default": 100,
            "description": "Number of frames per batch - choose fewer if using GPU and running out of memory.",
        },
        "nonrigid": {
            "gui_name": "Use nonrigid registration",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether to use nonrigid registration.",
        },
        "maxregshiftNR": {
            "gui_name": "Nonrigid max pixel shift",
            "type": int,
            "min": 0,
            "max": np.inf,
            "default": 5,
            "description": "Maximum pixel shift allowed for nonrigid, relative to rigid, may need to increase value for unstable recordings.",
        },
        "block_size": {
            "gui_name": "Nonrigid block size",
            "type": tuple,
            "min": None,
            "max": None,
            "default": (128, 128),
            "description": "Block size for non-rigid registration (** keep this a multiple of 2, 3, and/or 5 **).",
        },
        "smooth_sigma_time": {
            "gui_name": "Time smoothing",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 0,
            "description": "Gaussian smoothing in time to compute registration shifts (may be necessary with low SNR).",
        },
        "smooth_sigma": {
            "gui_name": "Smoothing in XY",
            "type": float,
            "min": 0.25,
            "max": np.inf,
            "default": 1.15,
            "description": "Gaussian smoothing in XY; ~1 good for 2P recordings, 3-5 may work well for 1P recordings.",
        },
        "spatial_taper": {
            "gui_name": "Edge tapering width",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 3.45,
            "description": "Edge tapering width in pixels, may want larger for 1P recordings.",
        },
        "th_badframes": {
            "gui_name": "Bad frame threshold",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 1.,
            "description": "Determines which frames to exclude when determining cropping - set it smaller to exclude more frames, particularly if crop seems small.",
        },
        "norm_frames": {
            "gui_name": "Normalize frames",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Normalize frames when detecting shifts.",
        },
        "snr_thresh": {
            "gui_name": "Nonrigid SNR threshold",
            "type": float,
            "min": 1.,
            "max": np.inf,
            "default": 1.2,
            "description": "If any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing.",
        },
        "subpixel": {
            "gui_name": "Nonrigid subpixel reg",
            "type": int,
            "min": 1,
            "max": 100,
            "default": 10,
            "description": "Precision of subpixel registration for nonrigid (1/subpixel steps).",
        },
        "two_step_registration": {
            "gui_name": "Run registration twice",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True.",
        },
        "reg_tif": {
            "gui_name": "Save registered tiffs",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to save registered tiffs.",
        },
        "reg_tif_chan2": {
            "gui_name": "Save chan2 registered tiffs",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to save chan2 registered tiffs.",
        },
        
    },
    "detection": {
        "algorithm": {
            "gui_name": "Detection algorithm",
            "type": str,
            "min": None,
            "max": None,
            "default": "sparsery",
            "description": "Algorithm used for cell detection ['sparsery', 'sourcery', 'cellpose'].",
        },
        "denoise": {
            "gui_name": "Denoise",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Whether to use PCA denoising for cell detection.",
        },
        "block_size": {
            "gui_name": "Denoise block size",
            "type": tuple,
            "min": None,
            "max": None,
            "default": (64, 64),
            "description": "Block size for denoising.",
        },
        "nbins": {
            "gui_name": "Max binned frames",
            "type": int,
            "min": 1,
            "max": np.inf,
            "default": 5000,
            "description": "Max number of binned frames for cell detection (may need to reduce if reduced RAM).",
        },
        "bin_size": {
            "gui_name": "Bin size",
            "type": int,
            "min": 1,
            "max": np.inf,
            "default": None,
            "description": "Size of bins for cell detection (default is tau * fs).",
        },
        "highpass_time": {
            "gui_name": "Highpass time",
            "type": int,
            "min": 0,
            "max": np.inf,
            "default": 100,
            "description": "Running mean subtraction across bins with a window of size highpass_time (may want to use low values for 1P).",
        },
        "threshold_scaling": {
            "gui_name": "Threshold scaling",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 1.0,
            "description": "Adjust the automatically determined threshold in sparsery and sourcery by this scalar multiplier - set it smaller to find more cells.",
        },
        "npix_norm_min": {
            "gui_name": "Min npix norm",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 0.,
            "description": "Minimum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix).",
        },
        "npix_norm_max": {
            "gui_name": "Max npix norm",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 100,
            "description": "Maximum npix norm for ROI (npix_norm = per ROI npix normalized by highest variance ROIs' mean npix).",
        },
        "max_overlap": {
            "gui_name": "Max overlap",
            "type": float,
            "min": 0.,
            "max": 1.,
            "default": 0.75,
            "description": "ROIs with more overlap than this fraction with other ROIs are discarded.",
        },
        "soma_crop": {
            "gui_name": "Soma crop",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Crop dendrites from ROI to determine ROI npix_norm and compactness.",
        },
        "chan2_threshold": {
            "gui_name": "Chan2 threshold",
            "type": float,
            "min": 0.,
            "max": 1.,
            "default": 0.25,
            "description": "IoU threshold between anatomical ROI and functional ROI to define as 'redcell'",
        },
        "cellpose_chan2": {
            "gui_name": "Cellpose chan2",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Use Cellpose to detect ROIs in anatomical channel and overlap with functional ROIs",
        },
        "sparsery_settings": {
            "highpass_neuropil": {
                "gui_name": "Highpass neuropil",
                "type": int,
                "min": 5,
                "max": np.inf,
                "default": 25,
                "description": "Highpass filter in pixels on binned movie to subtract neuropil signal (may want smaller/larger values depending on ROI diameter).",
            },
            "max_ROIs": {
                "gui_name": "Max ROIs",
                "type": int,
                "min": 1,
                "max": np.inf,
                "default": 5000,
                "description": "Maximum number of ROIs to detect.",
            },
            "spatial_scale": {
                "gui_name": "Spatial scale",
                "type": int,
                "min": 0,
                "max": 4,
                "default": 0,
                "description": "Spatial scale for cell detection (0 for auto-detect scaling, 1=6 pixels, 2=12 pixels, 3=24 pixels, 4=48 pixels).",
            },
            "active_percentile":{
                "gui_name": "Active percentile",
                "type": float,
                "min": 0.,
                "max": 1.,
                "default": 0.,
                "description": "Percentile of active pixels in the movie to use for thresholding; default is zero (recommended), which instead uses threshold.",
            },
        },
        "sourcery_settings": {
            "connected": {
                "gui_name": "Connected",
                "type": bool,
                "min": None,
                "max": None,
                "default": True,
                "description": "Whether or not to keep ROIs fully connected (set to 0 for dendrites).",
            },
            "max_iterations": {
                "gui_name": "Max iterations",
                "type": int,
                "min": 1,
                "max": np.inf,
                "default": 20,
                "description": "Maximum number of iterations for ROI detection.",
            },
            "smooth_masks": {
                "gui_name": "Smooth masks",
                "type": bool,
                "min": None,
                "max": None,
                "default": False,
                "description": "Whether or not to smooth masks.",
            },
        },
        "cellpose_settings": {
            "cellpose_model": {
                "gui_name": "Cellpose model",
                "type": str,
                "min": None,
                "max": None,
                "default": "cpsam",
                "description": "Cellpose model name or model path to use for cell detection (cyto, nuclei, etc).",
            },
            "img": {
                "gui_name": "Cellpose image",
                "type": str,
                "min": None,
                "max": None,
                "default": "max_proj / meanImg",
                "description": "Cellpose image to use for cell detection ['max_proj / meanImg', 'meanImg', 'max_proj'].",
            },
            "highpass_spatial": {
                "gui_name": "Highpass spatial",
                "type": int,
                "min": 0,
                "max": np.inf,
                "default": 0,
                "description": "Highpass image with sigma before running cellpose.",
            },
            "flow_threshold": {
                "gui_name": "Flow threshold",
                "type": float,
                "min": 0.,
                "max": 1.,
                "default": 0.4,
                "description": "Flow threshold for cellpose.",
            },
            'cellprob_threshold': {
                "gui_name": "Cellprob threshold",
                "type": float,
                "min": 0.,
                "max": 1.,
                "default": 0.0,
                "description": "Cell probability threshold for cellpose.",
            },
            "params": {
                "gui_name": "Additional cellpose parameters",
                "type": dict,
                "min": None,
                "max": None,
                "default": None,
                "description": "Parameters for cellpose, provided as a dict.",
            },
            "flow_threshold": {
                "gui_name": "Flow threshold",
                "type": float,
                "min": 0.,
                "max": 1.,
                "default": 0.4,
                "description": "Flow threshold for cellpose.",
            },
            'cellprob_threshold': {
                "gui_name": "Cellprob threshold",
                "type": float,
                "min": 0.,
                "max": 1.,
                "default": 0.0,
                "description": "Cell probability threshold for cellpose.",
            },
            "params": {
                "gui_name": "Additional cellpose parameters",
                "type": dict,
                "min": None,
                "max": None,
                "default": None,
                "description": "Parameters for cellpose, provided as a dict.",
            },
            "params_chan2": {
                "gui_name": "Additional cellpose parameters for chan2",
                "type": dict,
                "min": None,
                "max": None,
                "default": None,
                "description": "Parameters for cellpose chan2, provided as a dict.",
            },
        },
    },
    "classification": {
        "classifier_path": {
            "gui_name": "Classifier path",
            "type": str,
            "min": None,
            "max": None,
            "default": None,
            "description": "Path to classifier file for ROIs (default is ~/.suite2p/classifiers/classifier_user.npy).",
        },
        "use_builtin_classifier": {
            "gui_name": "Use built-in classifier",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Use built-in classifier (classifier.npy) instead of user classifier (classifier_user.npy) for ROIs.",
        },
        "preclassify": {
            "gui_name": "Pre-classify",
            "type": float,
            "min": 0.,
            "max": 1.,
            "default": 0.,
            "description": "Remove ROIs with classifier probability below preclassify before extraction to minimize overlaps",
        }
    },
    "extraction": {
        "snr_threshold": {
            "gui_name": "SNR threshold",
            "type": float,
            "min": -np.inf,
            "max": 1.,
            "default": 0.,
            "description": "SNR threshold for ROIs.",
        },
        "batch_size": {
            "gui_name": "Batch size",
            "type": int,
            "min": 1,
            "max": np.inf,
            "default": 500,
            "description": "Batch size for extraction.",
        },
        "neuropil_extract": {
            "gui_name": "Extract neuropil",
            "type": bool,
            "min": None,
            "max": None,
            "default": True,
            "description": "Whether or not to extract neuropil; if False, Fneu is set to zero.",
        },
        "neuropil_coefficient": {
            "gui_name": "Neuropil coefficient",
            "type": float,
            "min": 0.,
            "max": np.inf,
            "default": 0.7,
            "description": "Coefficient for neuropil subtraction.",
        },
        "inner_neuropil_radius": {
            "gui_name": "Inner neuropil radius",
            "type": int,
            "min": 0,
            "max": np.inf,
            "default": 2,
            "description": "Number of pixels to exclude from neuropil next to ROI.",
        },
        "min_neuropil_pixels": {
            "gui_name": "Min neuropil pixels",
            "type": int,
            "min": 1,
            "max": np.inf,
            "default": 350,
            "description": "Minimum number of pixels in the per ROI neuropil.",
        },
        "lam_percentile": {
            "gui_name": "Lambda percentile",
            "type": float,
            "min": 0.,
            "max": 100.,
            "default": 50.,
            "description": "Percentile of ROI lam weights to ignore when excluding cell pixels for neuropil extraction.",
        },
        "allow_overlap": {
            "gui_name": "Allow overlap",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Pixels that are overlapping are thrown out (False) or used for both ROIs (True).",
        },
        "circular_neuropil": {
            "gui_name": "Circular neuropil",
            "type": bool,
            "min": None,
            "max": None,
            "default": False,
            "description": "Force neuropil_masks to be circular instead of square (slow).",
        },
    },
    "dcnv_preprocess": {
        "baseline": {
            "gui_name": "Baseline type",
            "type": str,
            "min": None,
            "max": None,
            "default": "maximin",
            "description": "Method for baseline estimation ['maximin', 'prctile', 'constant'].",
        },
        "win_baseline": {
            "gui_name": "Baseline window",
            "type": float,
            "min": 1.,
            "max": np.inf,
            "default": 60.,
            "description": "Window (in seconds) for max filter.",
        },
        "sig_baseline": {
            "gui_name": "Baseline sigma",
            "type": float,
            "min": 1.,
            "max": np.inf,
            "default": 10.,
            "description": "Width of Gaussian filter in frames (applied to find constant or before maximin filter).",
        },
        "prctile_baseline": {
            "gui_name": "Baseline percentile",
            "type": float,
            "min": 0.,
            "max": 100.,
            "default": 8.,
            "description": "Percentile of trace to use as baseline if using 'prctile' for baseline.",
        },
    },
}

def default_dict(d):
    dnew = {}
    for k, v in d.items():
        if "description" in v:
            dnew[k] = v["default"] 
        else:
            dnew[k] = default_dict(v)
    return dnew

def default_db():
    """ default options to run pipeline """
    return default_dict(DB)

def default_settings():
    """ default options to run pipeline """
    settings = default_dict(SETTINGS)
    settings["version"] = version  
    return settings

def user_settings():
    """ user-default options to run pipeline """
    if (SETTINGS_FOLDER / "settings_user.npy").exists():
        settings = np.load(SETTINGS_FOLDER / "settings_user.npy", allow_pickle=True).item()
        settings = {**default_settings(), **settings}
    else:
        settings = default_settings()
    return settings

def add_descriptions(d, dstr="settings", k0=None):
    all_params = []
    kstr = "" if k0 is None else k0
    
    for k, v in d.items():
        if "description" in v:
            pname = f'{dstr}{kstr}["{k}"]'
            if v["min"] is not None:
                s = f"""
                        {pname} ; 
                        Default value: {str(v["default"])} ;   
                        Min, max: ({str(v['min'])}, {str(v['max'])}) ; 
                        Type: {v['type'].__name__}
                    """
            else:
                s = f"""
                        {pname} ; 
                        Default value: {str(v["default"])} ;   
                        Type: {v['type'].__name__}
                    """
            v["description"] += s
            all_params.append([pname, v["description"]])
        else:
            all_params0 = add_descriptions(v, k0=kstr + f'["{k}"]')
            all_params.append(k)
            all_params.extend(all_params0)
    return all_params

def print_all_params():
    all_dbs = add_descriptions(DB, dstr="db")
    all_params = add_descriptions(SETTINGS, dstr="settings")

    n = 0
    print(f"file settings")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    for kv in all_dbs:
        print(kv[0])
        print("\t"+kv[1])
        n += 1 

    print(f"general settings")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    for kv in all_params:
        if isinstance(kv, list):
            print(kv[0])
            print("\t"+kv[1])
            n += 1
        else:
            if "settings" not in kv:
                print("\n")
                print(f"{kv} settings")
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    return n 

def set_db(db, db_in):
    for key in db.keys():
        if key in db_in:
            db[key] = db_in[key]
            del db_in[key]

def set_settings(settings, settings_in):
    for key in settings.keys():
        if key in settings_in:
            if isinstance(settings[key], dict) and len(settings[key].keys()) > 0:
                set_settings(settings[key], settings_in[key])
                del settings_in[key]
            else:
                settings[key] = settings_in[key]   
                del settings_in[key]

def set_settings_orig(settings, settings_in):
    # also support flattened keys from old suite2p
    for key in settings.keys():
        if isinstance(settings[key], dict) and len(settings[key].keys()) > 0:
            set_settings_orig(settings[key], settings_in)
        elif key in settings_in:
            settings[key] = settings_in[key]   
            del settings_in[key]

def convert_settings_orig(settings_in, db=default_db(), settings=default_settings()):
    """ convert settings from old suite2p to new suite2p db and settings format """
    set_db(db, settings_in)
    set_settings(settings, settings_in)
    set_settings_orig(settings, settings_in)
    if "roidetect" in settings_in:
        settings["run"]["do_detection"] = settings_in.pop("roidetect")
    if "spikedetect" in settings_in:
        settings["run"]["do_deconvolution"] = settings_in.pop("spikedetect")
    return db, settings, settings_in

                