Cell Detection
==============

Summary
------------

The cell detection algorithm consists of reducing the
dimensionality of the data (principal components computation), smoothing
spatial principal components, finding peaks in these components, and
extending ROIs spatially around these peaks. On each iteration of peak
extraction, the neuropil is estimated from large spatial masks and
subtracted from the spatial components. This is to improve cell
detection and to help avoid extracting neuropil components with large
spatial extents.

SVDs ( = PCs) of data
---------------------

Before computing the principal components of the movie, we bin the data
such that we have at least as many frames to take the SVD of as
specified in the option ``ops['navg_frames_svd']``. The bin size will be
the maximum of ``nframes/ops['navg_frames_svd']`` and
``ops['tau'] * ops['fs']`` (the number of samples per transient). We
then bin the movie into this bin size and subtract the mean of the
binned movie across time. Then we smooth the movie in Y and X with a
gaussian filter of standard deviation ``sig = ops['diameter']/10``. The
we normalize the pixels by their noise variance. The noise variance is
variance of each pixel in the movie across time (at least 1e-10). Then
we compute the covariance of the movie (``mov @ mov.T``). Then we
compute the SVD of the covariance and keep the top
``ops['nsvd_for_roi']`` spatial components (components that are Y x X).

The function that performs this is ``celldetect2.getSVDdata`` and it
requires the ops described above, and Ly, Lx, yrange, xrange, and a
reg\_file location.

Sourcery
--------

After the spatial components are found, we perform an iterative
algorithm to find the cells in the components. Each iteration consists
of the following steps:

1. **Smoothing of spatial components**: The components are smoothed with
   a Gaussian filter in Y and X with standard deviation
   ``sig = ops['diameter']`` (this matrix is called ``us``). Note that
   diameter can be a list (for unequal pixel/um in Y and X). Next the
   mean of the squared smoothed components is computed. The mean of the
   squared un-smoothed components is also computed. The *correlation
   map* is defined as the element-wise division of the smoothed
   components by the unsmoothed components. The function that computes
   the correlation map is ``celldetect2.getVmap``.

2. **Detection of peaks in correlation map**: On each iteration, up to
   200 peaks are extracted from the correlation map. These are the
   largest remaining peaks such that they are greater than the
   threshold, which is set to be proportional to the median of the peaks
   in the whole correlation map:
   ``ops['threshold_scaling'] * np.median(peaks[peaks>1e-4])``. The
   initial activity ``code`` for this newly detected peak is the value
   of ``us`` (Gaussian smoothed PCs) at this peak. This is a vector of
   values across the PCs (nPCs in length).

3. **ROI extension**: The ROI is iteratively extended around its
   currently defined pixels +/- 1 in each direction. First, the new
   pixel weights (``lam``) of the extended ROI are computed. The weights
   ``lam`` are the unsmoothed PCs projected into the ``code`` dimension.
   The pixels that are greater than ``max(lam)/5`` are kept. The
   ``lam``'s are normalized to be unit norm. The new ``code`` is
   recomputed from the new weights, and is the unsmoothed PCs projected
   onto the ``lam`` weights. Then this extension procedure is repeated
   until no pixels are greater than ``max(lam)/5``.

4. **Neuropil computation**: Now that the new codes are computed, the
   neuropil is estimated. We set spatial basis functions for the
   neuropil, which are raised cosines that tile the FOV. The parameter
   ``ops['ratio_neuropil']`` determines how big you expect the neuropil
   basis functions to be relative to the cell diameter
   (``ops['diameter']``). The default is 6. This results in a tiling of
   7x7 raised cosines if your FOV is 512x512 pixels and your diameter is
   12 pixels. For one-photon recordings, we recommend setting
   ``ops['ratio_neuropil']`` to 2 or 3. Next we perform regression to
   compute the contribution of the neuropil on the PCs, and we subtract
   the estimated neuropil contribution from the ``U`` PCs. And these
   steps are repeated until the stopping criterion is reached.

**Stopping criterion**: The number of cells detected in the first
iteration is defined as ``Nfirst``. The cell detection is stopped if the
number of cells detected in the current iteration is less than
``Nfirst/10`` or if the iteration is the last iteration (defined by
``ops['max_iterations']``).

**Refinement step**: We remove masks which have more than a fraction
``ops['max_overlap']`` of their pixels overlapping with other masks.
Also, if ``ops['connected']=1``, then only the connected regions of ROIs
are kept. If you are looking for dendritic components, you may want to
set ``ops['connected']=0``.
