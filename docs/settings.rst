Settings (ops.npy)
------------------

Here is a summary of all the parameters that the pipeline takes, and its
default value.

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

- **frames_include**: (*int, default: -1*) if greater than zero, only *frames_include* frames are processed. useful for testing parameters on a subset of data.

Output settings
~~~~~~~~~~~~~~~

-  **preclassify**: (*float, default: 0.3*) (**new**) apply classifier
   before signal extraction with probability threshold of "preclassify".
   If this is set to 0.0, then all detected ROIs are kept and signals
   are computed.

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

Registration
~~~~~~~~~~~~

- **do_registration**: (*bool, default: True*) whether or not to run
  registration

- **align_by_chan**: (*int, default: 1*) which channel to use for
  alignment (1-based, so 1 means 1st channel and 2 means 2nd channel).
  If you have a non-functional channel with something like td-Tomato
  expression, you may want to use this channel for alignment rather
  than the functional channel.

- **nimg_init**: (*int, default: 200*) how many frames to use to
  compute reference image for registration

- **batch_size**: (*int, default: 200*) how many frames to register
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

**1P registration settings**

- **1Preg**: (*bool, default: False*) whether to perform high-pass
  spatial filtering and tapering (parameters set below), which help
  with 1P registration

- **spatial_hp**: (*int, default: 42*) window in pixels for spatial
  high-pass filtering before registration

- **pre_smooth**: (*float, default: 0*) if > 0, defines stddev of
  Gaussian smoothing, which is applied before spatial high-pass
  filtering

- **spatial_taper**: (*float, default: 40*) how many pixels to ignore
  on edges - they are set to zero (important for vignetted windows, for
  FFT padding do not set BELOW 3*ops['smooth_sigma'])

**Non-rigid registration**

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

ROI detection
~~~~~~~~~~~~~

- **roidetect**: (*bool, default: True*) whether or not to run ROI
  detect and extraction

- **sparse_mode**: (*bool, default: False*) whether or not to use sparse_mode cell detection

- **spatial_scale**: (*int, default: 0*), what the optimal scale of the
  recording is in pixels. if set to 0, then the algorithm determines it
  automatically (recommend this on the first try). If it seems off, set it yourself to the following values:
  1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).

- **connected**: (*bool, default: True*) whether or not to require ROIs
  to be fully connected (set to *0* for dendrites/boutons)

- **threshold_scaling**: (*float, default: 5.0*) this controls the
  threshold at which to detect ROIs (how much the ROIs have to stand
  out from the noise to be detected). if you set this higher, then
  fewer ROIs will be detected, and if you set it lower, more ROIs will
  be detected.

- **max_overlap**: (*float, default: 0.75*) we allow overlapping ROIs
  during cell detection. After detection, ROIs with more than
  ops['max_overlap'] fraction of their pixels overlapping with other
  ROIs will be discarded. Therefore, to throw out NO ROIs, set this to
  1.0.

- **high_pass**: (*int, default: 100*) running mean subtraction across
  time with window of size 'high_pass'. Values of less than 10 are
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

Signal extraction
~~~~~~~~~~~~~~~~~

- **allow_overlap**: (*bool, default: False*) whether or not to extract
  signals from pixels which belong to two ROIs. By default, any pixels
  which belong to two ROIs (overlapping pixels) are excluded from the
  computation of the ROI trace.

- **min_neuropil_pixels**: (*int, default: 350*) minimum number of
  pixels used to compute neuropil for each cell

- **inner_neuropil_radius**: (*int, default: 2*) number of pixels to
  keep between ROI and neuropil donut

Spike deconvolution
~~~~~~~~~~~~~~~~~~~

We neuropil-correct the trace Fout = F - ops['neucoeff'] \* Fneu, and
then baseline-correct these traces with an ops['baseline'] filter, and
then detect spikes.

- **neucoeff**: (*float, default: 0.7*) neuropil coefficient for all ROIs.

- **baseline**: (*string, default 'maximin'*) how to compute the
  baseline of each trace. This baseline is then subtracted from each
  cell. *'maximin'* computes a moving baseline by filtering the data
  with a Gaussian of width ops['sig_baseline'] \* ops['fs'], and then
  minimum filtering with a window of ops['win_baseline'] \* ops['fs'],
  and then maximum filtering with the same window. *'constant'*
  computes a constant baseline by filtering with a Gaussian of width
  ops['sig_baseline'] \* ops['fs'] and then taking the minimum value of
  this filtered trace. *'constant_percentile'* computes a constant
  baseline by taking the ops['prctile_baseline'] percentile of the
  trace.

- **win_baseline**: (*float, default: 60.0*) window for maximin filter
  in seconds

- **sig_baseline**: (*float, default: 10.0*) Gaussian filter width in
  seconds, used before maximin filtering or taking the minimum value of
  the trace, ops['baseline'] = 'maximin' or 'constant'.

- **prctile_baseline**: (*float, optional, default: 8*) percentile of
  trace to use as baseline if ops['baseline'] = 'constant_percentile'.

Channel 2 settings
~~~~~~~~~~~~~~~~~~

- **chan2_thres**: threshold for calling an ROI "detected" on a second
  channel
