# Installation

Please refer to the suite2p [README](https://github.com/MouseLand/suite2p#readme) for the latest up-to-date installation instructions.

**Common issues**

- If when running `suite2p`, you receive the error:
  `No module named PyQt5.sip`, then try uninstalling and reinstalling pyqt5
  > ```default
  > pip uninstall pyqt5 pyqt5-tools
  > pip install suite2p
  > ```
- If when running `suite2p`, you receive an error associated
  with **matplotlib**, try upgrading it:
  > ```default
  > pip install matplotlib --upgrade
  > ```
- If you are on Yosemite Mac OS, PyQt doesn’t work, and you won’t be able to install suite2p. More recent versions of Mac OS are fine.
- If you are using Ubuntu 22.04 and run into the following issue:
  ```default
  qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in even though it was found.
  This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application
  may fix this problem.
  ```

  Follow this [link](https://askubuntu.com/questions/1406379/qt5-install-problem-ubuntustudio-22-04/1406503#1406503) to install Qt5 and the issue above should be fixed.

The software has been heavily tested on Windows 10 and Ubuntu 18.04, and
less well tested on Mac OS. Please post an issue if you have
installation problems. The registration step runs faster on Ubuntu than
Windows, so if you have a choice we recommend using the Ubuntu OS.

## Dependencies

- [rastermap](https://github.com/MouseLand/rastermap)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.13.0)
- [scipy](https://www.scipy.org/)
- [h5py](https://www.h5py.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [scanimage-tiff-reader](http://scanimage.gitlab.io/ScanImageTiffReaderDocs/)
- [tifffile](https://pypi.org/project/tifffile/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [matplotlib](https://matplotlib.org/) (not for plotting (only using hsv_to_rgb and colormap
  function), should not conflict with PyQt5)
