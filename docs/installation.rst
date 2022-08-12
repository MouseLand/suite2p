Installation
----------------

Please refer to the suite2p `README`_ for the latest up-to-date installation instructions.

**Common issues**

- If when running ``suite2p``, you receive the error:
  ``No module named PyQt5.sip``, then try uninstalling and reinstalling pyqt5
  
   ::

      pip uninstall pyqt5 pyqt5-tools
      pip install suite2p

- If when running ``suite2p``, you receive an error associated
  with **matplotlib**, try upgrading it:

   ::

      pip install matplotlib --upgrade

- If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able to install suite2p. More recent versions of Mac OS are fine.

- If you are using Ubuntu 22.04 and run into the following issue:

  ::
  
     qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in even though it was found. 
     This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application
     may fix this problem.
  
  Follow this `link`_ to install Qt5 and the issue above should be fixed.

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
.. _link: https://askubuntu.com/questions/1406379/qt5-install-problem-ubuntustudio-22-04/1406503#1406503
.. _README: https://github.com/MouseLand/suite2p#readme