Registration
--------------------------------------

You can register your frames using the first channel of the recording,
or using the second channel. Say your first channel shows GCaMP and your
second channel shows td-Tomato, you might want to use the second channel
for registration if it has higher SNR. If so, set
``ops['align_by_chan']=2``. Otherwise, leave ``ops['align_by_chan']=1``
(default).

Finding a target reference image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To perform registration, we need a reference image to align all the
frames to. This requires an initial alignment step. Consider we just
took the average of a subset of frames. Because these frames are not
motion-corrected, the average will not be crisp - there will be fuzzy
edges because objects in the image have been moving around across the
frames. Therefore, we do an initial iterative alignment procedure on a
random subset of frames in order to get a crisp reference image for
registration. We first take ``ops['nimg_init']`` random frames of the
movie. Then from those frames, we take the top 20 frames that are most
correlated to each other and take the mean of those frames as our
initial reference image. Then we refine this reference image iteratively
by aligning all the random frames to the reference image, and then
recomputing the reference image as the mean of the best aligned frames.

The function that performs these steps can be run as follows (where ops
needs the reg_file, Ly, Lx, and nimg_init parameters):

.. code:: python

   from suite2p.registration import register

   refImg = register.pick_initial_reference(ops)

Here is an example reference image on the right, compared to just taking
the average of a random subset of frames (on the left):

.. image:: _static/badrefimg.png
   :width: 600

If the reference image doesn't look good, try increasing
``ops['nimg_init']``.

Registering the frames to the reference image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the reference image is obtained, we align each frame to the
reference image. The frames are registered in batches of size
``ops['batch_size']`` (default is 200 frames per batch).

We first perform rigid registration (assuming that the whole image
shifts by some (dy,dx)), and then optionally after that we perform
non-rigid registration (assuming that subsegments of the image shift by
separate amounts). To turn on non-rigid registration, set
``ops['nonrigid']=True``. We will outline the parameters of each
registration step below.

.. _1-rigid-registration:

1. Rigid registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rigid registration computes the shifts between the frame and the
reference image using phase-correlation. We have found on simulated data
that phase-correlation is more accurate than cross-correlation.
`Phase-correlation <https://en.wikipedia.org/wiki/Phase_correlation>`_ 
is a well-established method to compute the
relative movement between two images. Phase-correlation normalizes the
Fourier spectra of the images before multiplying them (whereas
cross-correlation would just multiply them). This normalization
emphasizes the correlation between the higher frequency components of
the images, which in most cases makes it more robust to noise.

Cross-correlation

.. image:: _static/rigid_cross.png
   :width: 600

Phase-correlation

.. image:: _static/rigid_phase.png
   :width: 600

Comparison

.. image:: _static/phase_vs_cross.png
   :width: 600

You can set a maximum shift size using the option
``ops['maxregshift']``. By default, it is 0.1, which means that the
maximum shift of the frame from the reference in the Y direction is
``0.1 * ops['Ly']`` and in X is ``0.1 * ops['Lx']`` where Ly and Lx are
the Y and X sizes of the frame.

After computing the shifts, the frames are shifted in the Fourier domain
(allowing subpixel shifts of the images). The shifts are saved in
``ops['yoff']`` and ``ops['xoff']`` for y and x shifts respectively. The
peak of the phase-correlation of each frame with the reference image is
saved in ``ops['corrXY']``.

You can run this independently from the pipeline, if you have a
reference image (ops requires the parameters nonrigid=False,
num_workers, and maxregshift):

.. code:: python

   maskMul,maskOffset,cfRefImg = register.prepare_masks(refImg)
   refAndMasks = [maskMul,maskOffset,cfRefImg]
   aligned_data, yshift, xshift, corrXY, yxnr = register.phasecorr(data, refAndMasks, ops)

(see bioRxiv preprint comparing cross/phase `here <https://www.biorxiv.org/content/early/2016/06/30/061507>`_)

.. _2-non-rigid-registration-optional:

2. Non-rigid registration (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run rigid registration and find that there is still motion in
your frames, then you should run non-rigid registration. Non-rigid
registration divides the image into subsections and computes the shift
of each subsection (called a block) separately. Non-rigid registration
will approximately double the registration time.

The size of the blocks to divide the image into is defined by
``ops['block_size'] = [128,128]`` which is the size in Y and X in
pixels. If Y is the direction of line-scanning for 2p imaging, you may
want to divide it into smaller blocks in that direction.

.. image:: _static/overlapping_blocks.png
   :width: 600

Each block is able to shift up to ``ops['maxregshiftNR']`` pixels in Y
and X. We recommend to keep this small unless you're in a very high
signal-to-noise ratio regime and your motion is very large. For subpixel shifts, 
we use Kriging interpolation and run it on each block. 

Phase correlation of each block:

.. image:: _static/block_phasecorr.png
   :width: 600

Shift of each block from phase corr:

.. image:: _static/block_arrows.png
   :width: 600

In a low signal-to-noise ratio regime, there may be blocks which on a
given frame do not have sufficient information from which to align with
the reference image. We compute a given block's maximum phase
correlation with the reference block, and determine how much greater this max is than 
the surrounding phase correlations. The ratio
between these two is defined as the ``snr`` of that block at that given
time point. We smooth over high snr blocks and use bilinear interpolation 
to upsample create the final shifts:

.. image:: _static/block_upsample.png
   :width: 600

We then use bilinear interpolation to warp the frame using these shifts.

Metrics for registration quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inputs required for PC metrics are the following fields in ops:
``nframes``, ``Ly``, ``Lx``, ``reg_file``. Your movie must have at least 1500 frames in each plane
for the metrics to be calculated. You can run on the red channel (ops['reg_file_chan2']) if use_red=True.
The outputs saved from the PC metrics are ``ops['regDX']``, ``ops['tPC']`` and ``ops['regPC']``.

::
   
   from suite2p.registration import metrics 

   ops = metrics.get_pc_metrics(ops, use_red=False)
   
``ops['tPC']`` are the time courses of each of the principal 
components of the registered movie. Note 
the time-course is not the entire movie, it's only the subset of frames used to 
compute the PCs (2000-5000 frames equally sampled throughout the movie). 

``ops['regPC']`` are computed from the spatial principal components of the
registered movie. ``ops['regPC'][0,0,:,:]`` is the average of the top
500 frames of the 1st PC, ``ops['regPC'][1,0,:,:]`` is the average of
the bottom 500 frames of the 1st PC. ``ops['regDX']`` quantifies the
movement in each PC (``iPC``) by registering ``ops['regPC'][0,iPC,:,:]``
and ``ops['regPC'][1,iPC,:,:]`` to the reference image ``ops['refImg']`` (if available, 
if not the mean of all the frames is used as the reference image)
and computing the registration shifts.

Here's a twitter `thread <https://twitter.com/marius10p/status/1051494533786193920>`_ 
with multiple examples.

.. _Phase-correlation: 
.. _here: 
.. |bad-refImg| image:: badrefImg.PNG

CLI Script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suite2p provides a CLI (Command-Line Interface) script that calculates the registration metrics
for a given input tif and outputs some statistics on those metrics. You can use this script to
determine the quality of registration and tune your registration parameters (e.g: determine if
non-rigid registration is necessary).

To run the script, use the following command:

.. prompt:: bash

    reg_metrics <INSERT_OPS_DATA_PATH> # Add --tiff_list <INSERT_INPUT_TIF_FILENAME_HERE>.tif to select a subset of tifs

Once you run the ``reg_metrics`` command, registration will be performed for the input file with default
ops parameters and an output similar to the following will be shown:

::

    # Average NR refers to the average nonrigid offsets of the blocks for a PC
    # Max NR refers to the max nonrigid offsets of the blocks for a PC
    Plane 0:
    Avg_Rigid: 0.000000     Avg_Average NR: 0.028889        Avg_Max NR: 0.120000
    Max_Rigid: 0.000000     Max_Average NR: 0.044444        Max_Max NR: 0.200000

For each ``nplane``, these statistics (Average and Max) are calculated across PCs on the offsets found in ``ops['regDX']``.
If the registration works perfectly and most of the motion is removed from the registered dataset, these scores
should all be very close to zero.

.. Important::

    Make sure to also inspect the registered video to check the quality of registration. You can see an example
    of how this is done in the GUI `here <https://youtu.be/M7UjvCUn74Y?t=810>`_.

You may notice that upon visual inspection, the registered video may look fine/contain little motion even
if the statistics are not close to zero. You should always visually check the registration output and prioritize
what your eyes say over what the CLI script reports.

.. note::

    All suite2p registration `settings <settings.html#registration>`__ can be modified in this CLI script. Just pass
    the setting with its value as an optional argument. For instance,

    .. prompt:: bash

        reg_metrics path_to_data_tif --nplanes 2 --smooth_sigma 1.2

    runs the script with ``ops['nplanes'] = 2`` and ``ops['smooth_sigma'] = 1.2``.
    You can see all the arguments ``reg_metrics`` takes with the following command:

    .. prompt:: bash

        reg_metrics --help
