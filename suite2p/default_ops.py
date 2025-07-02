"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from .version import version


def default_ops():
    """ default options to run pipeline """
    return {
        # Suite2p version
        "suite2p_version": version,  #current version of suite2p used for pipeline

        # file input/output settings
        "look_one_level_down":
            False,  # whether to look in all subfolders when searching for tiffs
        "fast_disk": [],  # used to store temporary binary file, defaults to save_path0
        "delete_bin": False,  # whether to delete binary file after processing
        "mesoscan": False,  # for reading in scanimage mesoscope files
        "bruker": False,  # whether or not single page BRUKER tiffs!
        "bruker_bidirectional":
            False,  # bidirectional multiplane in bruker: 0, 1, 2, 2, 1, 0 (True) vs 0, 1, 2, 0, 1, 2 (False)
        "h5py": [],  # take h5py as input (deactivates data_path)
        "h5py_key": "data",  #key in h5py where data array is stored
        "nwb_file": "",  # take nwb file as input (deactivates data_path)
        "nwb_driver": "",  # driver for nwb file (nothing if file is local)
        "nwb_series":
            "",  # TwoPhotonSeries name, defaults to first TwoPhotonSeries in nwb file
        "save_path0": '',  # pathname where you'd like to store results, defaults to first item in data_path
        "save_folder": [],  # directory you"d like suite2p results to be saved to
        "subfolders": [
        ],  # subfolders you"d like to search through when look_one_level_down is set to True
        "move_bin":
            False,  # if 1, and fast_disk is different than save_disk, binary file is moved to save_disk

        # main settings
        "nplanes": 1,  # each tiff has these many planes in sequence
        "nchannels": 1,  # each tiff has these many channels per plane
        "functional_chan":
            1,  # this channel is used to extract functional ROIs (1-based)
        "tau": 1.,  # this is the main parameter for deconvolution
        "fs":
            10.,  # sampling rate (PER PLANE e.g. for 12 plane recordings it will be around 2.5)
        "force_sktiff": False,  # whether or not to use scikit-image for tiff reading
        "frames_include": -1,
        "multiplane_parallel": False,  # whether or not to run on server
        "ignore_flyback": [],

        # output settings
        "preclassify":
            0.0,  # apply classifier before signal extraction with probability 0.3
        "save_mat": False,  # whether to save output as matlab files
        "save_NWB": False,  # whether to save output as NWB file
        "combined":
            True,  # combine multiple planes into a single result /single canvas for GUI
        "aspect":
            1.0,  # um/pixels in X / um/pixels in Y (for correct aspect ratio in GUI)

        # bidirectional phase offset
        "do_bidiphase":
            False,  #whether or not to compute bidirectional phase offset (applies to 2P recordings only)
        "bidiphase":
            0,  # Bidirectional Phase offset from line scanning (set by user). Applied to all frames in recording.
        "bidi_corrected":
            False,  # Whether to do bidirectional correction during registration

        # registration settings
        "do_registration": True,  # whether to register data (2 forces re-registration)
        "two_step_registration":
            False,  # whether or not to run registration twice (useful for low SNR data). Set keep_movie_raw to True if setting this parameter to True. 
        "keep_movie_raw":
            False,  # whether to keep binary file of non-registered frames. 
        "nimg_init": 300,  # subsampled frames for finding reference image
        "batch_size": 500,  # number of frames per batch
        "maxregshift":
            0.1,  # max allowed registration shift, as a fraction of frame max(width and height)
        "align_by_chan":
            1,  # when multi-channel, you can align by non-functional channel (1-based)
        "reg_tif": False,  # whether to save registered tiffs
        "reg_tif_chan2": False,  # whether to save channel 2 registered tiffs
        "subpixel": 10,  # precision of subpixel registration (1/subpixel steps)
        "smooth_sigma_time": 0,  # gaussian smoothing in time
        "smooth_sigma":
            1.15,  # ~1 good for 2P recordings, recommend 3-5 for 1P recordings
        "th_badframes":
            1.0,  # this parameter determines which frames to exclude when determining cropping - set it smaller to exclude more frames
        "norm_frames": True,  # normalize frames when detecting shifts
        "force_refImg": False,  # if True, use refImg stored in ops if available
        "pad_fft": False,  # if True, pads image during FFT part of registration

        # non rigid registration settings
        "nonrigid": True,  # whether to use nonrigid registration
        "block_size": [128,
                       128],  # block size to register (** keep this a multiple of 2 **)
        "snr_thresh":
            1.2,  # if any nonrigid block is below this threshold, it gets smoothed until above this threshold. 1.0 results in no smoothing
        "maxregshiftNR":
            5,  # maximum pixel shift allowed for nonrigid, relative to rigid

        # 1P settings
        "1Preg": False,  # whether to perform high-pass filtering and tapering
        "spatial_hp_reg":
            42,  # window for spatial high-pass filtering before registration
        "pre_smooth":
            0,  # whether to smooth before high-pass filtering before registration
        "spatial_taper":
            40,  # how much to ignore on edges (important for vignetted windows, for FFT padding do not set BELOW 3*ops["smooth_sigma"])

        # cell detection settings with suite2p
        "roidetect": True,  # whether or not to run ROI extraction
        "spikedetect": True,  # whether or not to run spike deconvolution
        "sparse_mode": True,  # whether or not to run sparse_mode
        "spatial_scale":
            0,  # 0: multi-scale; 1: 6 pixels, 2: 12 pixels, 3: 24 pixels, 4: 48 pixels
        "connected":
            True,  # whether or not to keep ROIs fully connected (set to 0 for dendrites)
        "nbinned": 5000,  # max number of binned frames for cell detection
        "max_iterations": 20,  # maximum number of iterations to do cell detection
        "threshold_scaling":
            1.0,  # adjust the automatically determined threshold by this scalar multiplier
        "max_overlap":
            0.75,  # cells with more overlap than this get removed during triage, before refinement
        "high_pass":
            100,  # running mean subtraction across bins with a window of size "high_pass" (use low values for 1P)
        "spatial_hp_detect":
            25,  # window for spatial high-pass filtering for neuropil subtraction before detection
        "denoise": False,  # denoise binned movie for cell detection in sparse_mode

        # cell detection settings with cellpose (used if anatomical_only > 0)
        "anatomical_only":
            0,  # run cellpose to get masks on 1: max_proj / mean_img; 2: mean_img; 3: mean_img enhanced, 4: max_proj
        "diameter": 0,  # use diameter for cellpose, if 0 estimate diameter
        "cellprob_threshold": 0.0,  # cellprob_threshold for cellpose
        "flow_threshold": 0.4,  # flow_threshold for cellpose
        "spatial_hp_cp": 0,  # high-pass image spatially by a multiple of the diameter
        "pretrained_model":
            "cpsam",  # path to pretrained model or model type string in Cellpose (can be user model)

        # classification parameters
        "soma_crop":
            True,  # crop dendrites for cell classification stats like compactness
        # ROI extraction parameters
        "neuropil_extract":
            True,  # whether or not to extract neuropil; if False, Fneu is set to zero
        "inner_neuropil_radius":
            2,  # number of pixels to keep between ROI and neuropil donut
        "min_neuropil_pixels": 350,  # minimum number of pixels in the neuropil
        "lam_percentile":
            50.,  # percentile of lambda within area to ignore when excluding cell pixels for neuropil extraction
        "allow_overlap":
            False,  # pixels that are overlapping are thrown out (False) or added to both ROIs (True)
        "use_builtin_classifier":
            False,  # whether or not to use built-in classifier for cell detection (overrides
        # classifier specified in classifier_path if set to True)
        "classifier_path": "",  # path to classifier

        # channel 2 detection settings (stat[n]["chan2"], stat[n]["not_chan2"])
        "chan2_thres": 0.65,  # minimum for detection of brightness on channel 2

        # deconvolution settings
        "baseline": "maximin",  # baselining mode (can also choose "prctile")
        "win_baseline": 60.,  # window for maximin
        "sig_baseline": 10.,  # smoothing constant for gaussian filter
        "prctile_baseline": 8.,  # optional (whether to use a percentile baseline)
        "neucoeff": 0.7,  # neuropil coefficient
    }
