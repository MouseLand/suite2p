# Installation

Please refer to the suite2p [README](https://github.com/MouseLand/suite2p#readme) for the latest up-to-date installation instructions.

## Common issues

If you receive an issue with Qt "xcb", you may need to install xcb libraries, e.g.:

```bash
sudo apt install libxcb-cursor0
sudo apt install libxcb-xinerama0
```

There is also more advice here: https://github.com/NVlabs/instant-ngp/discussions/300.

If you are having issues with CUDA on Windows, or want to use
Cuda Toolkit 10, please follow these [instructions](https://github.com/MouseLand/cellpose/issues/481#issuecomment-1080137885):

```bash
conda create -n suite2p pytorch=1.8.2 cudatoolkit=10.2 -c pytorch-lts
conda activate suite2p
pip install suite2p
```

If you receive the error: `No module named PyQt5.sip`, then try
uninstalling and reinstalling pyqt5

```bash
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
```

If you are having other issues with the graphical interface and QT, see some advice [here](https://github.com/MouseLand/cellpose/issues/564#issuecomment-1268061118).

If you have errors related to OpenMP and libiomp5, then try

```bash
conda install nomkl
```

If you receive an error associated with **matplotlib**, try upgrading
it:

```bash
pip install matplotlib --upgrade
```

If you receive the error: `ImportError: _arpack DLL load failed`, then try uninstalling and reinstalling scipy

```bash
pip uninstall scipy
pip install scipy
```

If you are on Yosemite Mac OS or earlier, PyQt doesn't work and you won't be able
to use the graphical interface for cellpose. More recent versions of Mac
OS are fine. The software has been heavily tested on Windows 10 and
Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if
you have installation problems.

## Dependencies

Suite2p relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [pytorch](https://pytorch.org/)
- [numpy](http://www.numpy.org/) (>=1.20.0)
- [scipy](https://www.scipy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [tifffile](https://github.com/cgohlke/tifffile)
- [scanimage-tiff-reader](https://vidriotech.gitlab.io/scanimage-tiff-reader/)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) or PySide
- [superqt](https://github.com/pyapp-kit/superqt)

Suite2p also optionally uses our anatomical segmentation tool [Cellpose](https://github.com/mouseland/cellpose). In the GUI our tool [Rastermap](https://github.com/mouseland/rastermap) is used for visualization