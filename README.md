# suite2p <img src="suite2p/logo/logo_unshaded.png" width="250" title="sweet two pea" alt="sweet two pea" align="right" vspace = "50">

[![Documentation Status](https://readthedocs.org/projects/suite2p/badge/?version=dev)](https://suite2p.readthedocs.io/en/dev/?badge=dev)

[![Build Status](https://travis-ci.org/Mouseland/suite2p.svg?branch=dev)](https://travis-ci.org/Mouseland/suite2p)

Pipeline for processing two-photon calcium imaging data.  
Copyright (C) 2018  Howard Hughes Medical Institute Janelia Research Campus  

suite2p includes the following modules:

* Registration
* Cell detection
* Spike detection
* Visualization GUI

This code was written by Carsen Stringer and Marius Pachitariu.  
For support, please open an [issue](https://github.com/MouseLand/suite2p/issues).
The reference paper is [here](https://www.biorxiv.org/content/early/2017/07/20/061507).  
The deconvolution algorithm is based on [this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423), with settings based on [this paper](http://www.jneurosci.org/content/early/2018/08/06/JNEUROSCI.3339-17.2018).

See this **twitter [thread](https://twitter.com/marius10p/status/1032804776633880583)** for GUI demonstrations.

The matlab version is available [here](https://github.com/cortex-lab/Suite2P). Note that the algorithm is older and will not work as well on non-circular ROIs.

## *NEW* detailed github [page](http://mouseland.github.io/suite2p)

## Installation

Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.x** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.

1. Download the [`environment.yml`](https://github.com/MouseLand/suite2p/blob/master/environment.yml) file from the repository. You can do this by cloning the repository, or copy-pasting the text from the file into a text document on your local computer.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Change directories to where the `environment.yml` is and run `conda env create -f environment.yml`
4. Pip install suite2p into the environment: `pip install suite2p`
6. Now run `python -m suite2p` and you're all set.

If you have an older `suite2p` environment you can remove it with `conda env remove -n suite2p` before creating a new one.

Note you will always have to run **conda activate suite2p** before you run suite2p. Conda ensures mkl_fft and numba run correctly and quickly on your machine. If you want to run jupyter notebooks in this environment, then also `conda install jupyter`.

To **upgrade** the suite2p (package [here](https://pypi.org/project/suite2p/)), run the following in the environment:
~~~~
pip install git+https://github.com/MouseLand/suite2p
~~~~


**Common issues**

If you are on Yosemite Mac OS, PyQt doesn't work, and you won't be able to install suite2p. More recent versions of Mac OS are fine.

The software has been heavily tested on Windows 10 and Ubuntu 18.04, and less well tested on Mac OS. Please post an issue if you have installation problems. The registration step runs faster on Ubuntu than Windows, so if you have a choice we recommend using the Ubuntu OS.

## Installation of github version for developers

1. Clone the repository and `cd suite2p` in an anaconda prompt / command prompt with `conda` for **python 3** in the path
2. Run `conda env create -f environment.yml`
3. To activate this new environment, run `conda activate suite2p` (you will have to activate every time you want to run suite2p)
4. Install the local version of suite2p into this environment in develop mode, with everything to get test data: `pip install .[data]`
5. Download the data: ``dvc pull``. Follow the authentication instructions in the output.
6. Run tests: `python setup.py test`


## Examples

An example dataset is provided [here](https://drive.google.com/open?id=1PCJy265NHRWYXUz7CRhbJHtd6B-zJs8f). It's a single-plane, single-channel recording.

## Getting started

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path. Make sure to run this from a directory in which you have **WRITE** access (suite2p saves a couple temporary files in your current directory):
~~~~
python -m suite2p
~~~~
Then:
1. File -> Run suite2p (or ctrl+r)
2. Setup a configuration
    - -> Add directory which contains tiffs to data_path (can be multiple folders, but add them one at a time)
    - -> OR choose an h5 file which has a key with the data, data shape should be time x pixels x pixels (you can type in the key name for the data after you choose the file)
    - -> Add save_path ((otherwise the data directory is used as save path))
    - -> Add fast_disk (this is where the binary file of registered data will be created, choose an SSD for this path) ((otherwise the save path is used as the fast disk path))
    - Set some parameters (see full list below). At the minimum:
		~~~~
		nplanes, nchannels, tau, fs
		~~~~
3. Press run and wait. Messages should start appearing in the embedded command line.
4. When the run is finished, the results will open in the GUI window and there you can visualize and refine the results (see below).

### Using the GUI

![multiselect](gui_images/multiselect.gif)

suite2p output goes to a folder called "suite2p" inside your save_path, which by default is the same as the data_path. If you ran suite2p in the GUI, it loads the results automatically. Otherwise, load the results with File -> Load results.

The GUI serves two main functions:

1. Checking the quality of the data and results.
	* there are currently several views such as the enhanced mean image, the ROI masks, the correlation map, the correlation among cells, and the ROI+neuropil traces
	* by selecting multiple cells (with "Draw selection" or ctrl+left-click), you can view the activity of multiple ROIs simultaneously in the lower plot
	* there are also population-level visualizations, such as [rastermap](https://github.com/MouseLand/rastermap)
2. Classify ROIs into cell / not cell (left and right views respectively)
	* the default classifier included should work well in a variety of scenarios.
	* a user-classifier can be learnt from manual curation, thus adapting to the statistics of your own data.
	* the GUI automatically saves which ROIs are good in "iscell.npy". The second column contains the probability that the ROI is a cell based on the currently loaded classifier.

Main GUI controls (works in all views):

1. Pan  = Left-Click  + drag  
2. Zoom = (Scroll wheel) OR (Right-Click + drag)
3. Full view = Double left-click OR escape key
4. Swap cell = Right-click on the cell
5. Select multiple cells = (Ctrl + left-click) OR (SHIFT + left-click) AND/OR ("Draw selection" button)

You can add your manual curation to a pre-built classifier by clicking "Add current data to classifier". Or you can make a brand-new classifier from a list of "iscell.npy" files that you've manually curated. The default classifier in the GUI is initialized as the suite2p classifier, but you can overwrite it by adding to it, or loading a different classifier and saving it as the default. The default classifier is used in the pipeline to produce the initial "iscell.npy" file.

## Other ways to call Suite2p

1. From the command line:
~~~~
python -m suite2p --ops <path to ops.npy> --db <path to db.npy>
~~~~

2. From Python/Jupyter
~~~~python
from suite2p.run_s2p import run_s2p
ops1 = run_s2p(ops, db)
~~~~

See our example jupyter notebook [here](jupyter/run_pipeline_tiffs_or_batch.ipynb). It also explains how to batch-run suite2p.

## Outputs

~~~~
F.npy: array of fluorescence traces (ROIs by timepoints)  
Fneu.npy: array of neuropil fluorescence traces (ROIs by timepoints)  
spks.npy: array of deconvolved traces (ROIs by timepoints)  
stat.npy: array of statistics computed for each cell (ROIs by 1)  
ops.npy: options and intermediate outputs
iscell.npy: specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
~~~~

## Dependencies
suite2p relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [rastermap](https://github.com/MouseLand/rastermap)
- [pyqtgraph](http://pyqtgraph.org/)
- [PyQt5](http://pyqt.sourceforge.net/Docs/PyQt5/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [mkl_fft](https://anaconda.org/conda-forge/mkl_fft)
- [scanimage-tiff-reader](https://vidriotech.gitlab.io/scanimagetiffreader-python/)
- [scipy](https://www.scipy.org/)
- [h5py](https://www.h5py.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [scanimage-tiff-reader](http://scanimage.gitlab.io/ScanImageTiffReaderDocs/)
- [tifffile](https://pypi.org/project/tifffile/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [matplotlib](https://matplotlib.org/) (not for plotting (only using hsv_to_rgb and colormap function), should not conflict with PyQt5)

### Logo
Logo was designed by Shelby Stringer and [Chris Czaja](http://chrisczaja.com/).
