Suite2p processes 2p calcium imaging data from raw tifs to extracted fluorescence traces and spike times. 

Copyright (C) 2018  Howard Hughes Medical Institute Janelia Research Campus

Suite2p includes the following modules:

* Registration
* Cell detection
* Spike detection 
* Visualization GUI

This code was written by Carsen Stringer and Marius Pachitariu. The reference paper is [here](https://www.biorxiv.org/content/early/2017/07/20/061507). For support, please open an issue directly on github.

## Installation

	pip install suite2p

If you don't already have Python (and pip), you'll need to first install a distribution of Python like [Anaconda](https://www.anaconda.com/download/). Choose Python 3.x 64-bit for your operating system. 

## Getting started

The quickest way to start is to open the GUI:

	python -m suite2p

From here you should:

0. File -> Run suite2p
1. Setup a configuration for your own data
    ** Add folders with tiffs:  -> Add directory to data_path
    ** Add a save path (otherwise the data directory is used as save path) -> Choose save_path
    ** Set some parameters. At the minimum: 
        *** nplanes, nchannels, diameter, tau, fs. See below for what these do. 
2. Press run and wait. Messages should start appearing in the embedded command line. 
3. When the run is finished, use the GUI to visualize and refine the results (see below).

## How to use the GUI

Suite2p outputs all results in a folder called "suite2p" inside your save_path, which by default is the same as the data_path. If you ran suite2p in the GUI, it already loaded the results. Otherwise, load the results with File -> Load results. 

The GUI serves two main functions:

1. Quickly check the quality of the data and results. 
2. Classify ROIs into cell / not cell.
 
To serve the first goal, we provide several views such as the enhanced mean image, the ROI masks, the correlation map, and the ROI+neuropil traces. We will later add more views population-level visualizations. 

To serve the second goal, suite2p builds a classifier based on the statistics of each ROI. This classifier can learn from manual curation (right-click to swap an ROI from the "accepted" to the "rejected" pile), and in this way adapt to the statistics of your own data. Nonetheless, the default classifier included should work well in a wide variety of scenarios. 
 
Main GUI controls (works for all views):

1. Pan  = Left-Click  + drag  
2. Zoom = Right-Click + drag  
3. Default view = Double left-click
4. Swap cell: Right-click on the cell
 
## Other ways to run Suite2p

1. From the command line:

	python -m suite2p --ops 'path to ops.npy' --db 'path to db.npy'
	
2. From Python/Jupyter

	from suite2p.run_sp import run_s2p
	ops1 = run_s2p(ops, db)
