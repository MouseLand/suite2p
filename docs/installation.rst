Installation
----------------

Install an `Anaconda`_ distribution of Python -- Choose **Python 3.x**
and your operating system. Note you might need to use an anaconda prompt
if you did not add anaconda to the path.

1. Download the suite2p repository from GitHub using Git:  ``git clone https://github.com/MouseLand/suite2p``
2. Open an anaconda prompt / command prompt with ``conda`` for **python
   3** in the path
3. Change the current directory to the suite2p folder: ``cd suite2p``
4. Run ``conda env create -f environment.yml``
5. To activate this new environment, run ``conda activate suite2p``. Afterwards, You should see ``(suite2p)`` on the left side of the terminal line.
6. Install suite2p into this environment: ``pip install suite2p``
7. Now run ``suite2p`` and you're all set.

If you have an older ``suite2p`` environment you can remove it with
``conda env remove -n suite2p`` before creating a new one.

Note you will always have to run **conda activate suite2p** before you
run suite2p. Conda ensures mkl_fft and numba run correctly and quickly
on your machine. If you want to run jupyter notebooks in this
environment, then also ``conda install jupyter``.

To upgrade suite2p (package `here`_), run the following in the
environment:

::

   pip install suite2p --upgrade

**Common issues**

If when running ``suite2p``, you receive the error:
``No module named PyQt5.sip``, then try uninstalling and reinstalling
pyqt5

::

   pip uninstall pyqt5 pyqt5-tools
   pip install suite2p

If when running ``suite2p``, you receive an error associated
with **matplotlib**, try upgrading it:

::

   pip install matplotlib --upgrade

If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able
to install suite2p. More recent versions of Mac OS are fine.

The software has been heavily tested on Windows 10 and Ubuntu 18.04, and
less well tested on Mac OS. Please post an issue if you have
installation problems. The registration step runs faster on Ubuntu than
Windows, so if you have a choice we recommend using the Ubuntu OS.

Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  `rastermap`_
-  `pyqtgraph`_
-  `PyQt5`_
-  `numpy`_ (>=1.13.0)
-  `scipy`_
-  `h5py`_
-  `scikit-learn`_
-  `scanimage-tiff-reader`_
-  `tifffile`_
-  `natsort`_
-  `matplotlib`_ (not for plotting (only using hsv_to_rgb and colormap
   function), should not conflict with PyQt5)

.. _rastermap: https://github.com/MouseLand/rastermap
.. _pyqtgraph: http://pyqtgraph.org/
.. _PyQt5: http://pyqt.sourceforge.net/Docs/PyQt5/
.. _numpy: http://www.numpy.org/
.. _scipy: https://www.scipy.org/
.. _h5py: https://www.h5py.org/
.. _tifffile: https://pypi.org/project/tifffile/ 
.. _scikit-learn: http://scikit-learn.org/stable/
.. _scanimage-tiff-reader: http://scanimage.gitlab.io/ScanImageTiffReaderDocs/
.. _natsort: https://natsort.readthedocs.io/en/master/
.. _matplotlib: https://matplotlib.org/
.. _Anaconda: https://www.anaconda.com/download/
.. _here: https://pypi.org/project/suite2p/
