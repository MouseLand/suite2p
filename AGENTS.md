# Suite2p - Agent Reference Guide

## Overview
Suite2p is a comprehensive Python pipeline for processing two-photon calcium imaging data from neuroscience experiments. It automates the detection of neurons, extraction of fluorescence signals, and provides a GUI for manual review and curation of results.

**Authors**: Carsen Stringer and Marius Pachitariu (HHMI Janelia Research Campus)  
**License**: GPL v3  
**Repository**: https://github.com/MouseLand/suite2p

## Main Goal
Process raw two-photon calcium imaging data (TIFF/H5/NWB formats) into clean neuronal activity traces through automated registration, cell detection, signal extraction, and spike deconvolution. Enables researchers to identify active neurons and quantify their calcium dynamics.

## Core Pipeline Stages

### 1. **Registration** (`suite2p/registration/`)
- **Purpose**: Correct motion artifacts in imaging data
- **Approach**: Phase-correlation based rigid and non-rigid registration
- **Key Files**: `register.py`, `rigid.py`, `nonrigid.py`, `bidiphase.py`
- **Process**: 
  - Iteratively finds crisp reference image from random frame subset
  - Performs rigid (whole-image) shifts using phase-correlation
  - Optional non-rigid registration for local deformations using block-based alignment
  - Outputs binary file of registered frames

### 2. **Cell Detection** (`suite2p/detection/`)
- **Purpose**: Identify neuron locations (ROIs) from imaging data
- **Approach**: SVD-based dimensionality reduction + iterative peak detection
- **Key Files**: `detect.py`, `sparsedetect.py`, `anatomical.py`, `stats.py`
- **Process**:
  - Compute spatial principal components (SVD) of binned movie
  - Smooth components and create correlation map
  - Iteratively detect peaks and extend ROI boundaries
  - Estimate and subtract neuropil contamination
  - Optional: Use Cellpose for anatomical segmentation (`anatomical_only` mode)

### 3. **Signal Extraction** (`suite2p/extraction/`)
- **Purpose**: Extract fluorescence time series from detected ROIs
- **Key Files**: `extract.py`, `masks.py`, `dcnv.py`
- **Process**:
  - Create pixel masks for each ROI and surrounding neuropil
  - Extract fluorescence traces (F) and neuropil traces (Fneu)
  - Perform neuropil subtraction
  - Spike deconvolution to estimate neural activity

### 4. **Classification** (`suite2p/classification/`)
- **Purpose**: Distinguish true cells from artifacts
- **Key Files**: `classifier.py`, `classify.py`
- **Features**: Pre-trained classifier based on ROI statistics (compactness, activity, etc.)
- **Output**: `iscell.npy` - binary labels and probability scores

### 5. **GUI** (`suite2p/gui/`)
- **Purpose**: Interactive visualization and manual curation
- **Key Files**: `gui2p.py`, `classgui.py`, `reggui.py`, `visualize.py`
- **Features**:
  - View ROI masks, correlation maps, activity traces
  - Manually classify cells vs. non-cells
  - Train custom classifiers
  - Visualize population activity with rastermap
  - Review registration quality

## Tech Stack

### Core Dependencies
- **Python**: 3.8+ (specified in environment.yml)
- **Numerical Computing**: NumPy, SciPy, Numba (JIT compilation for performance)
- **Deep Learning**: PyTorch (≥1.7.1) - used for neural network-based operations
- **Image Processing**: scikit-image, tifffile, scanimage-tiff-reader
- **GUI Framework**: PyQt5, pyqtgraph
- **Data I/O**: h5py, pynwb (Neurodata Without Borders), sbxreader
- **Visualization**: matplotlib, rastermap

### Performance Optimization
- **Intel MKL**: Math Kernel Library for fast FFT operations
- **Numba**: Just-in-time compilation for numerical functions
- **TBB**: Threading Building Blocks for parallelization
- **Binary format**: Efficient memory-mapped file handling

## Architecture

### Package Structure
```
suite2p/
├── run_s2p.py           # Main pipeline orchestrator
├── io/                  # Input/output handlers (TIFF, H5, NWB, binary)
├── registration/        # Motion correction algorithms
├── detection/           # ROI detection and segmentation
├── extraction/          # Signal extraction and deconvolution
├── classification/      # Cell vs. artifact classification
├── gui/                 # PyQt5-based graphical interface
├── ops/                 # Default parameters and configuration
└── classifiers/         # Pre-trained classification models
```

### Data Flow
1. **Input**: TIFF stacks, H5 files, or NWB files → converted to binary format
2. **Registration**: Binary file → registered binary + reference image
3. **Detection**: Registered data → ROI statistics (`stat.npy`)
4. **Extraction**: ROIs + registered data → fluorescence traces (`F.npy`, `Fneu.npy`)
5. **Deconvolution**: Fluorescence → spike estimates (`spks.npy`)
6. **Classification**: ROI stats → cell labels (`iscell.npy`)
7. **Output**: All results saved as NumPy arrays + optional MATLAB/NWB export

### Key Configuration
- **`ops` dictionary**: Central configuration object containing all pipeline parameters
- **Default parameters**: Defined in `run_s2p.py::default_ops()`
- **Runtime**: Can run via GUI, command line, or Python API (`run_s2p(ops, db)`)

## Entry Points
- **GUI**: `suite2p` (launches PyQt5 GUI)
- **CLI**: `suite2p --ops <ops.npy> --db <db.npy>`
- **Python API**: `from suite2p.run_s2p import run_s2p; run_s2p(ops, db)`
- **Jupyter**: Example notebooks in `jupyter/` directory

## Output Files (in `suite2p/` directory)
- `F.npy`: Fluorescence traces (ROIs × timepoints)
- `Fneu.npy`: Neuropil fluorescence traces
- `spks.npy`: Deconvolved spike estimates
- `stat.npy`: ROI statistics (location, shape, properties)
- `ops.npy`: Final parameters and metadata
- `iscell.npy`: Cell classification (binary label + probability)

## Notable Features
- **Multi-plane support**: Process volumetric recordings with multiple imaging planes
- **Multiple channels**: Can register using non-functional channel (e.g., td-Tomato)
- **Batch processing**: Process multiple datasets via Python scripts
- **Extensible**: Users can train custom classifiers from manually curated data
- **Format agnostic**: Supports ScanImage, Bruker, Mesoscope, standard TIFF, H5, NWB

## Development Notes
- Active development on GitHub (MouseLand/suite2p)
- Test suite: `tests/` with pytest
- Benchmarking: `benchmarks/` for registration metrics
- Documentation: ReadTheDocs (suite2p.readthedocs.io)
- Version management: setuptools_scm (git-based versioning)

## Install
- uv venv --python 3.8 && uv pip install --python .venv/bin/python -e ".[gui]"
- uv pip install setuptools

- source .venv/bin/activate
- python -m suite2p