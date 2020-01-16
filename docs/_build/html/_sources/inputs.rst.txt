Inputs
-------------------------

Input format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This applies to all file types!

Directory structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

suite2p looks for all tiffs/hdf5 in the folders listed in
``ops['data_path']``. If you want suite2p to look in those folders AND
all their children folders, set ``ops['look_one_level_down']=True``. If
you want suite2p to only look at some of the folder's children, then set
``ops['subfolders']`` to those folder names.

If you want suite2p to only use specific tiffs in ONE folder, then set
the data path to only have one folder
(``ops['data_path']=['my_folder_path']``), and name the tiffs you want
processed in ``ops['tiff_list']``.

See examples in this `notebook`_.

Frame ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have data with multiple planes and/or multiple channels, suite2p
expects the frames to be interleaved, e.g.

-  frame0 = time0_plane0_channel1
-  frame1 = time0_plane0_channel2
-  frame2 = time0_plane1_channel1
-  frame3 = time0_plane1_channel2
-  frame4 = time1_plane0_channel1
-  frame5 = time1_plane0_channel2
-  ...

channels are ones-based (channel 1 and 2 NOT 0 and 1).

.. _recordings-with-photostim--other-artifacts:

Recordings with photostim / other artifacts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Photostim and other artifacts require you to exclude these frames during
ROI detection. Otherwise there will be "ROIs" that are related to the
stimulation, not actually cells. To exclude them, make an array of
integers corresponding to the frame times of the photostimulation. Save
this array into a numpy array called ``bad_frames.npy``:

::

   import numpy as np

   bad_frames = np.array([20,30,40])
   np.save('bad_frames.npy', bad_frames)

Put this file into the first folder in your ops['data_path'] (the first
folder you choose in the GUI).

Different file types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tiffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most tiffs should work out of the box. suite2p relies on two external
tiff readers: `scanimage-tiff-reader`_ and `sklearn.external.tifffile`_.
The default is the scanimage one, but it will use the other one if it
errors.

You can use single-page tiffs. These will work out of the box if they
end in \*.tif or \*.tiff. If they have a different ending then use the
flag ``ops['all_files_are_tiffs'] = True`` and the pipeline will assume
any files in your folders are tiffs. NOTE that these will be slower to
load in and create the binary, so if you're planning on using the
pipeline extensively you may want to change your acquisition output.

If you save a stack of tiffs using ImageJ, and it's larger than 4GB,
then it won't run through suite2p anymore. A work-around is to save as
an OME-TIFF in FIJI: "File->save as->OME-TIFF->compression type
uncompressed" in FIJI (thanks @kylemxm! see issue `here <https://github.com/MouseLand/suite2p/issues/149#issuecomment-473862374>`_).

If you have old Scanimage tiffs (version <5) that are larger than 2GB,
then most tiff readers will not work. @elhananby has recommended this `repository`_ for reading the data into matlab (see issue `here`_).
After reading it into matlab, you can re-save the tiff in a format that
imageJ and suite2p can recognize (see matlab tiff writing
`here <https://www.mathworks.com/help/matlab/ref/tiff.write.html>`__).

Bruker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using Bruker Prairie View system, .RAW files are batch converted to single .ome.tifs.
Now, you can load the resulting multiple tif files (i.e. one per frame per channel) to suite2p to be converted to binary.
This looks for files containing 'Ch1', and will assume all additional files are 'Ch2'.
Select "input_format" as "bruker" in the drop down menu in the GUI or set ``ops['input_format'] = "bruker"``.

Mesoscope tiffs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have a matlab script
`here <https://github.com/MouseLand/suite2p/blob/master/helpers/mesoscope_json_from_scanimage.m>`__
for extracting the parameters from scanimage tiffs collected from the
Thorlabs mesoscope. The script creates an ``ops.json`` file that you can
then load into the run GUI using the button "load ops file". This should
populate the run GUI with the appropriate parameters. Behind the scenes
there are ``ops['lines']`` loaded and ``ops['dy'],ops['dx']`` that
specify which lines in the tiff correspond to each ROI and where in
space each ROI is respectively. ``ops['nplanes']`` will only be greater
than 1 if you collected in multi-plane mode. Once the pipeline starts
running, this parameter will change to "nplanes \* nrois" and each
"plane" is now an ROI from a specific plane. Please open issues if
you're using this and having trouble because it's not straightforward.

Thorlabs raw files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Christoph Schmidt-Hieber (@neurodroid) has written `haussmeister`_ which
can load and convert ThorLabs \*.raw files to suite2p binary files!
suite2p will automatically use this if you have pip installed it
(``pip install haussmeister``).

.. _hdf5-files-and-sbx:

HDF5 files (and \*.sbx)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These should work out of the box, but are less well-tested. Dario
Ringach has a utility to convert neurolabware \*.sbx files to \*.h5
files (see blog post
`here <https://scanbox.org/2018/08/29/using-suite2p-with-scanbox/>`__).

The H5 loading from the GUI now works the same as it always has for tiffs. Select
"h5" from the drop-down menu and input the h5 KEY for the data as a string. Now
choose the folder with your \*.h5 or \*.hdf5 files and the pipeline will use all
h5 files in that folder. You can use ops['look_one_level_down'] to process all
subfolders of the data_path.


sbx binary files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scanbox binary files (*.sbx) work out of the box if you set ``ops['input_format'] = "sbx"``.

When recording in bidirectional mode some columns might have every other line saturated; to trim these during loading set ``ops['sbx_ndeadcols']``. Set this option to ``-1`` to let suite2p compute the number of columns automatically, a positive integer to specify the number of columns to trim.
Joao Couto (@jcouto) wrote the binary sbx parser.


.. _repository: https://github.com/dgreenberg/read_patterned_tifdata
.. _haussmeister: https://github.com/neurodroid/haussmeister
.. _notebook: https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_tiffs_or_batch.ipynb
.. _scanimage-tiff-reader: http://scanimage.gitlab.io/ScanImageTiffReaderDocs/
.. _sklearn.external.tifffile: http://scikit-image.org/docs/dev/api/skimage.external.tifffile.html
.. _here: https://github.com/MouseLand/suite2p/issues/135#issuecomment-467244278
