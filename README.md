# suite2p <img src="suite2p/logo/logo_unshaded.png" width="250" title="sweet two pea" alt="sweet two pea" align="right" vspace = "50">

[![Documentation Status](https://readthedocs.org/projects/suite2p/badge/?version=latest)](https://suite2p.readthedocs.io/en/latest/?badge=latest)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/MouseLand/suite2p)
![tests](https://github.com/mouseland/suite2p/actions/workflows/test_and_deploy.yml/badge.svg)
[![codecov](https://codecov.io/gh/MouseLand/suite2p/branch/main/graph/badge.svg?token=OJEC3mty85)](https://codecov.io/gh/MouseLand/suite2p)
[![PyPI version](https://badge.fury.io/py/suite2p.svg)](https://badge.fury.io/py/suite2p)
[![Downloads](https://static.pepy.tech/badge/suite2p)](https://pepy.tech/project/suite2p)
[![Downloads](https://static.pepy.tech/badge/suite2p/month)](https://pepy.tech/project/suite2p)
[![Python version](https://img.shields.io/pypi/pyversions/suite2p)](https://pypistats.org/packages/suite2p)
[![Licence: GPL v3](https://img.shields.io/github/license/MouseLand/suite2p)](https://github.com/MouseLand/suite2p/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/MouseLand/suite2p)](https://github.com/MouseLand/suite2p/graphs/contributors)
[![repo size](https://img.shields.io/github/repo-size/MouseLand/suite2p)](https://github.com/MouseLand/suite2p/)
[![GitHub stars](https://img.shields.io/github/stars/MouseLand/suite2p?style=social)](https://github.com/MouseLand/suite2p/)
[![GitHub forks](https://img.shields.io/github/forks/MouseLand/suite2p?style=social)](https://github.com/MouseLand/suite2p/)


Pipeline for processing two-photon calcium imaging data.
Copyright (C) 2026  Howard Hughes Medical Institute Janelia Research Campus

suite2p includes the following modules:

* Registration
* ROI detection
* Signal extraction
* ROI classification
* Spike detection
* Visualization GUI

For support, please open an [issue](https://github.com/MouseLand/suite2p/issues). The reference paper is [here](https://www.biorxiv.org/content/10.64898/2026.02.04.703741v1). The deconvolution algorithm is based on [this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423), with settings based on [this paper](http://www.jneurosci.org/content/early/2018/08/06/JNEUROSCI.3339-17.2018).

See this **twitter [thread](https://twitter.com/marius10p/status/1032804776633880583)** for GUI demonstrations. The matlab version is available [here](https://github.com/cortex-lab/Suite2P). Note that the algorithm is older and will not work as well on non-circular ROIs. Lectures on how suite2p works are available [here](https://youtu.be/HpL5XNtC5wU?list=PLutb8FMs2QdNqL4h4NrNhSHgLGk4sXarb).

**Note on pull requests**: we accept very few pull requests due to the maintenance efforts required to support new code, and we do not accept pull requests from automated code checkers. If you wrote code that interfaces/changes suite2p behavior, a common approach would be to keep that in a fork and pull periodically from the main branch to make sure you have the latest updates.

### CITATION

If you use this package in your research, please cite the [paper](https://www.biorxiv.org/content/10.64898/2026.02.04.703741v1):

Carsen Stringer, Chris Ki, Nicholas Del Grosso, Paul LaFosse, Qingqing Zhang, Marius Pachitariu (2026). Extracting large-scale neural activity with Suite2p. *bioRxiv*.

## Read the Documentation at https://suite2p.readthedocs.io/

## Local installation (< 2 minutes)

You can install cellpose using conda or with native python if you have python3.8+ on your machine. 

### System requirements

Linux, Windows and Mac OS are supported for running the code. For running the graphical interface you will need a Mac OS later than Yosemite. At least 8GB of RAM is required to run the software. 16GB-32GB is encouraged for larger recordings. The software has been heavily tested on Windows 10 and Ubuntu 24.04 and less well-tested on Mac OS. Please open an issue if you have problems with installation.

### Dependencies

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

Suite2p also optionally uses our anatomical segmentation tool [Cellpose](https://github.com/mouseland/cellpose). In the GUI our tool [Rastermap](https://github.com/mouseland/rastermap) is used for visualization.


### Option 1: Installation Instructions with conda 

If you have an older `suite2p` environment you can remove it with `conda env remove -n suite2p` before creating a new one (we recommend removing pre-2026 envs and re-creating).

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install a [miniforge](https://github.com/conda-forge/miniforge) distribution of Python. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt which has `conda` for **python 3** in the path
3. Create a new environment with `conda create --name suite2p python=3.11`. We recommend python 3.11, but python 3.9-3.12 will also work.
4. To activate this new environment, run `conda activate suite2p`
5. (option 1) To install cellpose with the GUI, run `python -m pip install suite2p[gui]`.  If you're on a zsh server, you may need to use ' ': `python -m pip install 'suite2p[gui]'`.
6. (option 2) To install cellpose without the GUI, run `python -m pip install suite2p`. 

To upgrade suite2p (package [here](https://pypi.org/project/suite2p/)), run the following in the environment:

~~~sh
python -m pip install suite2p --upgrade
~~~

Note you will always have to run `conda activate suite2p` before you run cellpose. If you want to run jupyter notebooks in this environment, then also `python -m pip install notebook` and `python -m pip install matplotlib`.

You can also try to install Suite2p and the GUI dependencies from your base environment using the command

~~~~sh
python -m pip install suite2p[gui]
~~~~

If you have **issues** with installation, see [here](https://cellpose.readthedocs.io/en/latest/installation.html) for more details. If these suggestions fail, open an issue.

### Option 2: Installation Instructions with python's venv

Venv ([tutorial](https://docs.python-guide.org/dev/virtualenvs/#lower-level-virtualenv), for those interested) is a built-in tool in python for creating virtual environments. It is a good alternative if you don't want to install conda and already have python3 on your machine. The main difference is that you will need to choose where to install the environment and the packages. Suite2p will then live in this environment and not be accessible from other environments. You will need to navigate to the environment directory and activate it each time before running Suite2p. The steps are similar to the conda installation:

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Install python3.8 or later from [python.org](https://www.python.org/downloads/). This will be the version of python that will be used in the environment. You can check your python version with `python --version`.
2. Navigate to the directory where you want to create the environment and run `python3 -m venv suite2p` to create a new environment called `suite2p`.
3. Activate the environment with `source suite2p/bin/activate` on Mac/Linux or `suite2p\Scripts\activate` on Windows. A prefix `(suite2p)` should appear in the terminal.
4. Install suite2p into the `suite2p` venv using pip with `python -m pip install suite2p`.
5. Install the suite2p GUI, with `python -m pip install suite2p[gui]`. Depending on your terminal software, you may need to use quotes like this: `python -m pip install 'suite2p[gui]'`.
6. You can now run suite2p from this environment with `python -m suite2p` or `suite2p` if you are in the suite2p directory.
7. To deactivate the environment, run `deactivate`. 

### GPU version (CUDA) on Windows or Linux

If you plan on running Suite2p on longer recordings, we strongly recommend installing a GPU version of *torch*. To use your NVIDIA GPU with python, you will need to make sure the NVIDIA driver for your GPU is installed, check out this [website](https://www.nvidia.com/Download/index.aspx?lang=en-us) to download it. You can also install the CUDA toolkit, or use the pytorch cudatoolkit (installed below with conda). If you have trouble with the below install, we recommend installing the CUDA toolkit yourself, choosing one of the 12.x releases [here](https://developer.nvidia.com/cuda-toolkit-archive).

With the latest versions of pytorch on Linux, as long as the NVIDIA drivers are installed, the GPU version is installed by default with pip.  

If it's not working, we will need to remove the CPU version of torch:
~~~
pip uninstall torch
~~~

To install the GPU version of torch, follow the instructions [here](https://pytorch.org/get-started/locally/). The pip or conda installs should work across platforms, you will need torch and torchvision, e.g. for windows + cuda 12.6 the command is
~~~
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
~~~

Info on how to install several older versions is available [here](https://pytorch.org/get-started/previous-versions/). After install you can check `conda list` for `pytorch`, and its version info should have `cuXX.X`, not `cpu`.

### Installation of github version

Follow steps from above to install the dependencies. Then run 
~~~
pip install git+https://www.github.com/mouseland/suite2p.git
~~~

If you want edit ability to the code, in the github repository folder, run `pip install -e .`. If you want to go back to the pip version of suite2p, then say `pip install suite2p`.

### Installing the latest github version of the code

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

The simplest way is
~~~
pip install git+https://github.com/MouseLand/suite2p.git
~~~

If you want to download and edit the code, and use that version,
1. Clone the repository with git and `cd suite2p`
2. Run `pip install -e .` in that folder


### Installation for developers

If you are using a GPU, make sure its drivers and the cuda libraries are correctly installed.

1. Clone the repository and `cd suite2p` in an anaconda prompt / command prompt with `conda` for **python 3** in the path
2. Run `conda create --name suite2p python=3.11`
3. To activate this new environment, run `conda activate suite2p` (you will have to activate every time you want to run suite2p)
4. Install the local version of suite2p into this environment in develop mode with the command `pip install -e .[all]`. If you're running a `zsh` shell, you may need to instead use `pip install -e .\[all\]`. 
5. Run tests: `pytest -vs`, this will automatically download the test data into your `suite2p` folder. The test data is split into two parts: test inputs and expected test outputs which will be downloaded in `data/test_inputs` and `data/test_outputs` respectively. The .zip files for these two parts can be downloaded from these links: [test_inputs](https://www.suite2p.org/static/test_data/test_inputs.zip) and [test_outputs](https://www.suite2p.org/static/test_data/test_outputs.zip).

## Examples

An example dataset is provided [here](https://drive.google.com/drive/folders/0B649boZqpYG1R3ota25jdUthSzQ?resourcekey=0-wSoqFv5rnE6TERPcJHwQtQ). It's a single-plane, single-channel recording.

## Getting started

The quickest way to start is to open the GUI from a command line terminal. You might need to open an anaconda prompt if you did not add anaconda to the path. Make sure to run this from a directory in which you have **WRITE** access (suite2p saves a couple temporary files in your current directory):
~~~~
suite2p
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

<img src="https://github.com/MouseLand/MouseLand.github.io/releases/download/v0.1/multiselect.gif" width="800" alt="selecting multiple ROIs in suite2p with Ctrl"/>


The suite2p output goes to a folder called "suite2p" inside your save_path, which by default is the same as the data_path. If you ran suite2p in the GUI, it loads the results automatically. Otherwise, you can load the results with File -> Load results or by dragging and dropping the stat.npy file into the GUI.

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
suite2p --settings <path to settings.npy> --db <path to db.npy>
~~~~

2. From Python/Jupyter
~~~~python
from suite2p.run_s2p import run_s2p
settings1 = run_s2p(settings, db)
~~~~

## Outputs

~~~~
F.npy: array of fluorescence traces (ROIs by timepoints)
Fneu.npy: array of neuropil fluorescence traces (ROIs by timepoints)
spks.npy: array of deconvolved traces (ROIs by timepoints)
stat.npy: array of statistics computed for each cell (ROIs by 1)
settings.npy: options and intermediate outputs
iscell.npy: specifies whether an ROI is a cell, first column is 0/1, and second column is probability that the ROI is a cell based on the default classifier
~~~~

# License

Copyright (C) 2023 Howard Hughes Medical Institute Janelia Research Campus, the labs of Carsen Stringer and Marius Pachitariu.

**This code is licensed under GPL v3 (no redistribution without credit, and no redistribution in private repos, see the [license](LICENSE) for more details).**

### Logo
Logo was designed by Shelby Stringer and [Chris Czaja](http://chrisczaja.com/).
