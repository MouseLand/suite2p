Outputs
-------------------------

``F.npy``: array of fluorescence traces (ROIs by timepoints)

``Fneu.npy``: array of neuropil fluorescence traces (ROIs by timepoints)

``spks.npy``: array of deconvolved traces (ROIs by timepoints)

``stat.npy``: list of statistics computed for each cell (ROIs by 1)

``ops.npy``: options and intermediate outputs (dictionary)

``iscell.npy``: specifies whether an ROI is a cell, first column is 0/1,
and second column is probability that the ROI is a cell based on the
default classifier

All can be loaded in python with numpy

.. code:: python

   import numpy as np

   F = np.load('F.npy', allow_pickle=True)
   Fneu = np.load('Fneu.npy', allow_pickle=True)
   spks = np.load('spks.npy', allow_pickle=True)
   stat = np.load('stat.npy', allow_pickle=True)
   ops =  np.load('ops.npy', allow_pickle=True)
   ops = ops.item()
   iscell = np.load('iscell.npy', allow_pickle=True)

MATLAB output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``'save_mat'=1``, then a MATLAB file is created ``Fall.mat``. This
will contain ops, F, Fneu, stat, spks and iscell. The "iscell"
assignments are only saved ONCE when the pipeline is finished running.
If you make changes in the GUI to the cell assignments, ONLY
``iscell.npy`` changes. To load a modified ``iscell.npy`` into MATLAB, I
recommend using this package: `npy-matlab`_. Alternatively there is a
*new* save button in the GUI (in the file menu) that allows you to save
the iscell again to the ``Fall.mat`` file.

NWB Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``ops['save_NWB']=1``, then an NWB file is created ``ophys.nwb``. This 
will contain the fields from ops and stat required to load back into the GUI, along 
with F, Fneu, spks and iscell. If 
the recording has multiple planes, then they are all saved together like in 
combined view. See fields below:

stat: stat['ypix'], stat['xpix'] (if multiplane `stat['iplane']`) are saved in 
'pixel_mask' (called `'voxel_mask'` in multiplane).

ops: 'meanImg', 'max_proj', 'Vcorr' are saved in Images 'Backgrounds_k' where k is the plane 
number, and have the same names. optionally if two channels, 'meanImg_chan2' is saved.

iscell: saved as an array 'iscell' 

F,Fneu,spks are saved as roi_response_series 'Fluorescence', 'Neuropil', and 'Deconvolved'.


Multichannel recordings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cells are detected on the ops['functional_chan'] and the fluorescence
signals are extracted from both channels. The functional channel signals
are saved to ``F.npy`` and ``F_neu.npy``, and non-functional channel
signals are saved to ``F_chan2.npy`` and ``Fneu_chan2.npy``.

.. _statnpy-fields:

stat.npy fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  ypix: y-pixels of cell
-  xpix: x-pixels of cell
-  med: (y,x) center of cell
-  lam: pixel mask (sum(lam \* frames[ypix,xpix,:]) = fluorescence)
-  npix: number of pixels in ROI
-  npix_norm: number of pixels in ROI normalized by the mean of npix
   across all ROIs
-  radius: estimated radius of cell from 2D Gaussian fit to mask
-  aspect_ratio: ratio between major and minor axes of a 2D Gaussian fit
   to mask
-  compact: how compact the ROI is (1 is a disk, >1 means less compact)
-  footprint: spatial extent of an ROI's functional signal, including
   pixels not assigned to the ROI; a threshold of 1/5 of the max is used
   as a threshold, and the average distance of these pixels from the
   center is defined as the footprint
-  skew: skewness of neuropil-corrected fluorescence trace
-  std: standard deviation of neuropil-corrected fluorescence trace
-  overlap: which pixels overlap with other ROIs (these are excluded
   from fluorescence computation)
-  ipix_neuropil: pixels of neuropil mask for this cell

Here is example code to make an image where each cell (without its
overlapping pixels) is a different "number":

::

   stat = np.load('stat.npy')
   ops = np.load('ops.npy').item()

   im = np.zeros((ops['Ly'], ops['Lx']))

   for n in range(0,ncells):
       ypix = stat[n]['ypix'][~stat[n]['overlap']]
       xpix = stat[n]['xpix'][~stat[n]['overlap']]
       im[ypix,xpix] = n+1

   plt.imshow(im)
   plt.show()

(There is no longer ipix like in the matlab version. In python note you
can access a 2D array like X[ys, xs] = lam. In Matlab, this would cause
a broadcast of all the pairs of ys and xs, which is why ipix = ys +
(xs-1) \* Ly was a useful temporary variable to have around for linear
indexing into arrays. In Python, the equivalent ipix would be ipix = yx
+ xs \* Lxy.)

.. _opsnpy-fields:

ops.npy fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will include all of the options you ran the pipeline with,
including file paths. During the running of the pipeline, some outputs
are added to ``ops.npy``:

    -  **reg_file**: location of registered binary file
    -  **Ly**: size of Y dimension of tiffs/h5
    -  **Lx**: size of X dimension of tiffs/h5
    -  **nframes**: number of frames in recording
    -  **yrange**: valid y-range used for cell detection (excludes edges that were shifted out of the FOV during registration)
    -  **xrange**: valid x-range used for cell detection (excludes edges that were shifted out of the FOV during registration)
    -  **refImg**: reference image used for registration
    -  **yoff**: y-shifts of recording at each timepoint
    -  **xoff**: x-shifts of recording at each timepoint
    -  **corrXY**: peak of phase correlation between frame and reference image at each timepoint
    -  **meanImg**: mean of registered frames
    -  **meanImgE**: a median-filtered version of the mean image
    -  **Vcorr**: correlation map (computed during cell detection)
    -  **filelist**: List of the image file names (e.g. tiff) that were loaded, in the order that Suite2p processed them.
    -  **date_proc**: Date and time that the analysis was run.

.. _npy-matlab: https://github.com/kwikteam/npy-matlab
