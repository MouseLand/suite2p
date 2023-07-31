Settings (ops.npy)
------------------

Suite2p can be run with different configurations using the ``ops`` dictionary. The ``ops`` dictionary will describe the settings used for a particular run of the pipeline. Here is a summary of all the parameters that the pipeline takes and their default values. 

Main settings
~~~~~~~~~~~~~

These are the essential settings that are dataset-specific.

-  **nplanes**: (*int, default: 1*) each tiff has this many planes in
   sequence

-  **nchannels**: (*int, default: 1*) each tiff has this many channels
   per plane

-  **functional_chan**: (*int, default: 1*) this channel is used to
   extract functional ROIs (1-based, so 1 means first channel, and 2
   means second channel)

-  **tau**: (*float, default: 1.0*) The timescale of the sensor (in
   seconds), used for deconvolution kernel. The kernel is fixed to have
   this decay and is not fit to the data. We recommend:

   -  0.7 for GCaMP6f
   -  1.0 for GCaMP6m
   -  1.25-1.5 for GCaMP6s

-  **force_sktiff**: (*boolean, default: False*) specifies whether or not to use scikit-image for reading in tiffs

-  **fs**: (*float, default: 10.0*) Sampling rate (per plane). For
   instance, if you have a 10 plane recording acquired at 30Hz, then the
   sampling rate per plane is 3Hz, so set ops['fs'] = 3.

-  **do_bidiphase**: (*bool, default: False*) whether or not to compute
   bidirectional phase offset from misaligned line scanning experiment
   (applies to 2P recordings only). suite2p will estimate the
   bidirectional phase offset from ops['nimg_init'] frames if this is
   set to 1 (and ops['bidiphase']=0), and then apply this computed
   offset to all frames.

-  **bidiphase**: (*int, default: 0*) bidirectional phase offset from
   line scanning (set by user). If set to any value besides 0, then this
   offset is used and applied to all frames in the recording.

- **bidi_corrected**: (*bool, default: False*) Specifies whether to do bidi correction. 

- **frames_include**: (*int, default: -1*) if greater than zero, only *frames_include* frames are processed. useful for testing parameters on a subset of data.

- **multiplane_parallel**: (*boolean, default: False*) specifies whether or not to run pipeline on server 

- **ignore_flyback**: (*list[ints], default: empty list*) specifies which planes will be ignored as flyback planes by the pipeline. 

File input/output settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

Suite2p can accomodate many different file formats. Refer to this
:ref:`page <inputs-diff-file-types>` for a detailed list of formats suite2p can work with. 

- **fast_disk**: (*list[str], default: empty list*) specifies location where temporary binary file will be stored. Defaults to ``save_path0`` if no directory is provided by user. 

- **delete_bin** (*bool, default:False*) specifies whether to delete binary file created during registration stage. 

- **mesoscan** (*bool, default: False*) specifies whether file being read in is a scanimage mesoscope recording

- **bruker** (*bool, default: False*) specifies whether provided tif files are single page BRUKER tiffs

- **bruker_bidirectional** (*bool, default: False*) specifies whether BRUKER files are bidirectional multiplane recordings. The ``True`` setting corresponds to the following plane order (first plane is indexed as zero): [0,1,2,2,1,0]. ``False`` corresponds to [0,1,2,0,1,2]. 

- **h5py** (*list[str], default: empty list*) specifies path to h5py file that will be used as inputs. Keep in mind the pathname provided here overwrites the pathname specified in ``ops[data_path]``. 

- **h5py_key** (*str, default: 'data'*) key used to access data array in h5py file. Only use this when the ``h5py`` setting is set to ``True``. 

- **nwb_file** (*str, default: ''*) specifies path to NWB file you use to use as input

- **nwb_driver** (*str, default: ''*) location of driver for NWB file. Leave this empty if the pathname refers to a local file.

- **nwb_series** (*str, default: ''*) Name of TwoPhotonSeries values you wish to retrieve from your NWB file. 

- **save_path0** (*str, default: ''*) String containing pathname of where you'd like to save your pipeline results. If no pathname is provided, the first element of ``ops['data_path']`` is used. 

- **save_folder** (*list[str], default: empty list*) List containing directory name you'd like results to be saved under. Defaults to ``"suite2p"``. 

- **look_one_level_down**: (*bool, default: False*) specifies whether to look in all subfolders when searching for tiffs. Make sure to specify subfolders in the ``subfolders`` parameter below. 

- **subfolders** (*list[str], default: empty list*) Specifies subfolders you'd like to look through. Make sure to have the above parameter ``ops[look_one_level_down] = True`` when using this parameter.

- **move_bin** (*bool, default: False*) If True and ``ops['fast_disk']`` is different from ``ops[save_disk]``, the created binary file is moved to ``ops['save_disk']``. 

Output settings
~~~~~~~~~~~~~~~

-  **preclassify**: (*float, default: 0.0*) (**new**) apply classifier
   before signal extraction with probability threshold of "preclassify".
   If this is set to 0.0, then all detected ROIs are kept and signals
   are computed.

- **save_nwb**: (*bool, default: False*) whether to save output as NWB file 

-  **save_mat**: (*bool, default: False*) whether to save the results in
   matlab format in file "Fall.mat". NOTE the cells you click in the GUI
   will NOT change "Fall.mat". But there is a **new** button in the GUI
   you can click to resave "Fall.mat" in the "File" window.

-  **combined**: (*bool, default: True*) combine results across planes
   in separate folder "combined" at end of processing. This folder will
   allow all planes to be loaded into the GUI simultaneously.

-  **aspect**: (*float, default: 1.0) (**new**) ratio of um/pixels in X
   to um/pixels in Y (ONLY for correct aspect ratio in GUI, not used for
   other processing)

-  **report_time**: (*bool, default: True) (**new**) whether or not to return
   a timing dictionary for each plane. Timing dictionary will contain keys
   corresponding to stages and values corresponding to the duration of that stage.


Registration settings
~~~~~~~~~~~~~~~~~~~~~

These settings are specific to the registration module of suite2p.

- **do_registration**: (*bool, default: True*) whether or not to run
  registration

- **align_by_chan**: (*int, default: 1*) which channel to use for
  alignment (1-based, so 1 means 1st channel and 2 means 2nd channel).
  If you have a non-functional channel with something like td-Tomato
  expression, you may want to use this channel for alignment rather
  than the functional channel.

- **nimg_init**: (*int, default: 300*) how many frames to use to
  compute reference image for registration

- **batch_size**: (*int, default: 500*) how many frames to register
  simultaneously in each batch. This depends on memory constraints - it
  will be faster to run if the batch is larger, but it will require
  more RAM.

- **maxregshift**: (*float, default: 0.1*) the maximum shift as a
  fraction of the frame size. If the frame is Ly pixels x Lx pixels,
  then the maximum pixel shift in pixels will be max(Ly,Lx) \*
  ops['maxregshift'].

- **smooth_sigma**: (*float, default: 1.15*) standard deviation in
  pixels of the gaussian used to smooth the phase correlation between
  the reference image and the frame which is being registered. A value
  of *>4* is recommended for one-photon recordings (with a 512x512
  pixel FOV).

- **smooth_sigma_time**: (*float, default: 0*) standard deviation in time frames
  of the gaussian used to smooth the data before phase correlation is computed.
  Might need this to be set to 1 or 2 for low SNR data.

- **keep_movie_raw**: (*bool, default: False*) whether or not to keep
  the binary file of the non-registered frames. You can view the
  registered and non-registered binaries together in the GUI in the
  "View registered binaries" view if you set this to *True*.

- **two_step_registration**: (*bool, default: False*) whether or not to run
  registration twice (for low SNR data). *keep_movie_raw* must be True for this
  to work.

- **reg_tif**: (*bool, default: False*) whether or not to write the
  registered binary to tiff files

- **reg_tif_chan2**: (*bool, default: False*) whether or not to write
  the registered binary of the non-functional channel to tiff files

- **subpixel**: (*int, default:10*) Precision of Subpixel Registration (1/subpixel steps)

- **th_badframes**: (*float, default: 1.0*) Involved with setting threshold for excluding frames for cropping. Set this smaller to exclude more frames. 

- **norm_frames**: (*bool, default: True*) Normalize frames when detecting shifts

- **force_refImg**: (*bool, default: False*) Specifies whether to use refImg stored in ``ops``. Make sure that ``ops['refImg']`` has a valid file pathname. 

- **pad_fft**: (*bool, default: False*) Specifies whether to pad image or not during FFT portion of registration. 

1P registration
^^^^^^^^^^^^^^^

- **1Preg**: (*bool, default: False*) whether to perform high-pass
  spatial filtering and tapering (parameters set below), which help
  with 1P registration

- **spatial_hp_reg**: (*int, default: 42*) window in pixels for spatial
  high-pass filtering before registration

- **pre_smooth**: (*float, default: 0*) if > 0, defines stddev of
  Gaussian smoothing, which is applied before spatial high-pass
  filtering

- **spatial_taper**: (*float, default: 40*) how many pixels to ignore
  on edges - they are set to zero (important for vignetted windows, for
  FFT padding do not set BELOW 3*ops['smooth_sigma'])

Non-rigid registration
^^^^^^^^^^^^^^^^^^^^^^

- **nonrigid**: (*bool, default: True*) whether or not to perform
  non-rigid registration, which splits the field of view into blocks
  and computes registration offsets in each block separately.

- **block_size**: (*two ints, default: [128,128]*) size of blocks for
  non-rigid registration, in pixels. HIGHLY recommend keeping this a
  power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft

- **snr_thresh**: (*float, default: 1.2*) how big the phase correlation
  peak has to be relative to the noise in the phase correlation map for
  the block shift to be accepted. In low SNR recordings like
  one-photon, I'd recommend a larger value like *1.5*, so that block
  shifts are only accepted if there is significant SNR in the phase
  correlation.

- **maxregshiftNR**: (*float, default: 5.0*) maximum shift in pixels of
  a block relative to the rigid shift

ROI detection settings 
~~~~~~~~~~~~~~~~~~~~~~

- **roidetect**: (*bool, default: True*) whether or not to run ROI
  detect and extraction

- **sparse_mode**: (*bool, default: True*) whether or not to use sparse_mode cell detection

- **spatial_scale**: (*int, default: 0*), what the optimal scale of the
  recording is in pixels. if set to 0, then the algorithm determines it
  automatically (recommend this on the first try). If it seems off, set it yourself to the following values:
  1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).

- **connected**: (*bool, default: True*) whether or not to require ROIs
  to be fully connected (set to *0* for dendrites/boutons)

- **threshold_scaling**: (*float, default: 1.0*) this controls the
  threshold at which to detect ROIs (how much the ROIs have to stand
  out from the noise to be detected). if you set this higher, then
  fewer ROIs will be detected, and if you set it lower, more ROIs will
  be detected.

- **spatial_hp_detect**: (*int, default: 25*) window for spatial high-pass filtering for neuropil subtracation before ROI detection takes place.

- **max_overlap**: (*float, default: 0.75*) we allow overlapping ROIs
  during cell detection. After detection, ROIs with more than
  ops['max_overlap'] fraction of their pixels overlapping with other
  ROIs will be discarded. Therefore, to throw out NO ROIs, set this to
  1.0.

- **high_pass**: (*int, default: 100*) running mean subtraction across
  bins of frames with window of size 'high_pass'. Values of less than 10 are
  recommended for 1P data where there are often large full-field
  changes in brightness.

- **smooth_masks**: (*bool, default: True*) whether to smooth masks in
  final pass of cell detection. This is useful especially if you are in
  a high noise regime.

- **max_iterations**: (*int, default: 20*) how many iterations over
  which to extract cells - at most ops['max_iterations'], but usually
  stops before due to ops['threshold_scaling'] criterion.

- **nbinned**: (*int, default: 5000*) maximum number of binned frames
  to use for ROI detection.

- **denoise**: (*bool, default: False*) Whether or not binned movie should be denoised before cell detection in sparse_mode. If True, make sure to set ``ops['sparse_mode']`` is also set to True. 

Cellpose Detection 
^^^^^^^^^^^^^^^^^^
These settings are only used if ``ops['anatomical_only']`` is set to an integer greater than 0. 

- **anatomical_only**: (*int, default: 0*) If greater than 0, specifies what to use `Cellpose <https://cellpose.readthedocs.io/>`_ on.

    - 1: Will find masks on max projection image divided by mean image.
    - 2: Will find masks on mean image
    - 3: Will find masks on enhanced mean image
    - 4: Will find masks on maximum projection image 

- **diameter**: (*int, default: 0*) Diameter that will be used for cellpose. If set to zero, diameter is estimated. 

- **cellprob_threshold**: (*float, default: 0.0*) specifies threshold for cell detection that will be used by cellpose. 

- **flow_threshold**: (*float, default: 1.5*) specifies flow threshold that will be used for cellpose.

- **spatial_hp_cp**: (*int, default: 0*) Window for spatial high-pass filtering of image to be used for cellpose. 

- **pretrained_model**: (*str, default: 'cyto'*) Path to pretrained model or string for model type (can be user's model ).

Signal extraction settings
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **neuropil_extract**: (*bool, default: True*) Whether or not to extract signal from neuropil. If False, Fneu is set to zero. 

- **allow_overlap**: (*bool, default: False*) whether or not to extract
  signals from pixels which belong to two ROIs. By default, any pixels
  which belong to two ROIs (overlapping pixels) are excluded from the
  computation of the ROI trace.

- **min_neuropil_pixels**: (*int, default: 350*) minimum number of
  pixels used to compute neuropil for each cell

- **inner_neuropil_radius**: (*int, default: 2*) number of pixels to
  keep between ROI and neuropil donut

- **lam_percentile**: (*int, default: 50*)Percentile of Lambda within area to ignore when excluding cell pixels for neuropil extraction

Spike deconvolution settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We neuropil-correct the trace ``Fout = F - ops['neucoeff'] * Fneu``, and
then baseline-correct these traces with an ``ops['baseline']`` filter, and
then detect spikes.

- **spikedetect**: (*bool, default: True*) Whether or not to run spike_deconvolution

- **neucoeff**: (*float, default: 0.7*) neuropil coefficient for all ROIs.

- **baseline**: (*string, default 'maximin'*) how to compute the
  baseline of each trace. This baseline is then subtracted from each
  cell. *'maximin'* computes a moving baseline by filtering the data
  with a Gaussian of width ``ops['sig_baseline'] * ops['fs']``, and then
  minimum filtering with a window of ``ops['win_baseline'] * ops['fs']``,
  and then maximum filtering with the same window. *'constant'*
  computes a constant baseline by filtering with a Gaussian of width
  ``ops['sig_baseline'] * ops['fs']`` and then taking the minimum value of
  this filtered trace. *'constant_percentile'* computes a constant
  baseline by taking the ``ops['prctile_baseline']`` percentile of the
  trace.

- **win_baseline**: (*float, default: 60.0*) window for maximin filter
  in seconds

- **sig_baseline**: (*float, default: 10.0*) Gaussian filter width in
  seconds, used before maximin filtering or taking the minimum value of
  the trace, ``ops['baseline'] = 'maximin'`` or ``'constant'``.

- **prctile_baseline**: (*float, optional, default: 8*) percentile of
  trace to use as baseline if ``ops['baseline'] = 'constant_percentile'``.

Classification settings
~~~~~~~~~~~~~~~~~~~~~~~

- **soma_crop**: (*bool, default: True*) Specifies whether to crop dendrites for cell classification stats (e.g., compactness)

- **use_builtin_classifier**: (*bool, default: False*) Specifies whether or not to use built-in classifier for cell detection. This will override classifier specified in ``ops['classifier_path']`` if set to True. 

- **classifier_path**: (*str, default: ''*) Path to classifier file you want to use for cell classification

Channel 2 specific settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **chan2_thres**: threshold for calling an ROI "detected" on a second
  channel


Miscellaneous settings
~~~~~~~~~~~~~~~~~~~~~~

- **suite2p_version**: specifies version of suite2p pipeline that was run with these settings. Changing this parameter will NOT change the version of suite2p used. 
