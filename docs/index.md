<!-- suite2p documentation master file, created by
sphinx-quickstart on Sun Aug 18 15:27:04 2019.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->

# Welcome to suite2p’s documentation!

suite2p is an imaging processing pipeline written in Python 3 which
includes the following modules:

- Registration
- Cell detection
- Spike detection
- Visualization GUI

For examples of how the output looks and how the GUI works, check out
this twitter [thread](https://twitter.com/marius10p/status/1032804776633880583).

This code was written by Carsen Stringer and Marius Pachitariu. For
support, please open an [issue](https://github.com/MouseLand/suite2p/issues).

The reference paper is [here](https://www.biorxiv.org/content/early/2017/07/20/061507). The deconvolution algorithm is based on
[this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423), with settings based on [this
paper](http://www.jneurosci.org/content/early/2018/08/06/JNEUROSCI.3339-17.2018).

We make pip installable releases of suite2p, here is the [pypi](https://pypi.org/project/suite2p/). You
can install it as `pip install suite2p`

* [Module Index](py-modindex.md)
* [Search Page](search.md)
* [Index](genindex.md)

# Basics:

* [Installation](installation.md)
  * [Dependencies](installation.md#dependencies)
* [Inputs](inputs.md)
  * [Input format](inputs.md#input-format)
    * [Directory structure](inputs.md#directory-structure)
    * [Frame ordering](inputs.md#frame-ordering)
    * [Recordings with photostim / other artifacts](inputs.md#recordings-with-photostim-other-artifacts)
  * [Different file types](inputs.md#different-file-types)
    * [Tiffs](inputs.md#tiffs)
    * [Bruker](inputs.md#bruker)
    * [Mesoscope tiffs](inputs.md#mesoscope-tiffs)
    * [Thorlabs raw files](inputs.md#thorlabs-raw-files)
    * [HDF5 files (and \*.sbx)](inputs.md#hdf5-files-and-sbx)
    * [sbx binary files](inputs.md#sbx-binary-files)
    * [Nikon nd2 files](inputs.md#nikon-nd2-files)
  * [BinaryFile](inputs.md#binaryfile)
* [Settings (settings.npy)](settings.md)
  * [file settings](settings.md#file-settings)
  * [general settings](settings.md#general-settings)
  * [run settings](settings.md#run-settings)
  * [io settings](settings.md#io-settings)
  * [registration settings](settings.md#registration-settings)
  * [detection settings](settings.md#detection-settings)
  * [extraction settings](settings.md#extraction-settings)
  * [dcnv_preprocess settings](settings.md#dcnv-preprocess-settings)
* [Using the GUI](gui.md)
  * [Different views and colors for ROI panels](gui.md#different-views-and-colors-for-roi-panels)
    * [Views](gui.md#views)
    * [Colors](gui.md#colors)
    * [Correlations](gui.md#correlations)
    * [Correlations with 1D var](gui.md#correlations-with-1d-var)
    * [Rastermap / custom](gui.md#rastermap-custom)
  * [Buttons / shortcuts for cell selection](gui.md#buttons-shortcuts-for-cell-selection)
    * [Mouse control](gui.md#mouse-control)
    * [Keyboard shortcuts](gui.md#keyboard-shortcuts)
    * [Multi-cell selection](gui.md#multi-cell-selection)
  * [Trace view (bottom row)](gui.md#trace-view-bottom-row)
  * [Classifying cells](gui.md#classifying-cells)
    * [Adding data to a classifier](gui.md#adding-data-to-a-classifier)
    * [Building your own classifier](gui.md#building-your-own-classifier)
    * [Applying a custom classifier](gui.md#applying-a-custom-classifier)
  * [Visualizing activity](gui.md#visualizing-activity)
  * [Manual adding of ROIs](gui.md#manual-adding-of-rois)
  * [Merging ROIs](gui.md#merging-rois)
  * [View registered binary](gui.md#view-registered-binary)
    * [Z-stack Alignment](gui.md#z-stack-alignment)
  * [View registration metrics](gui.md#view-registration-metrics)
* [Outputs](outputs.md)
  * [MATLAB output](outputs.md#matlab-output)
  * [NWB Output](outputs.md#nwb-output)
  * [Multichannel recordings](outputs.md#multichannel-recordings)
  * [stat.npy fields](outputs.md#stat-npy-fields)
  * [settings.npy fields](outputs.md#settings-npy-fields)
* [Multiday recordings](multiday.md)
* [Developer Documentation](developer_doc.md)
  * [Versioning](developer_doc.md#versioning)
  * [Testing](developer_doc.md#testing)
    * [Downloading Test Data](developer_doc.md#downloading-test-data)
    * [Running the tests](developer_doc.md#running-the-tests)
* [Frequently Asked Questions](FAQ.md)
  * [Cropped field-of-view](FAQ.md#cropped-field-of-view)
  * [Deconvolution means what?](FAQ.md#deconvolution-means-what)
  * [Multiple functional channels](FAQ.md#multiple-functional-channels)
  * [Z-drift](FAQ.md#z-drift)
  * [No signals in manually selected ROIs](FAQ.md#no-signals-in-manually-selected-rois)

# How it works:

* [Registration](registration.md)
  * [Finding a target reference image](registration.md#finding-a-target-reference-image)
  * [Registering the frames to the reference image](registration.md#registering-the-frames-to-the-reference-image)
  * [1. Rigid registration](registration.md#rigid-registration)
  * [2. Non-rigid registration (optional)](registration.md#non-rigid-registration-optional)
  * [Metrics for registration quality](registration.md#metrics-for-registration-quality)
    * [CLI Script](registration.md#cli-script)
* [Cell Detection](celldetection.md)
  * [Summary](celldetection.md#summary)
  * [SVDs ( = PCs) of data](celldetection.md#svds-pcs-of-data)
  * [Sourcery](celldetection.md#sourcery)
* [Signal extraction](roiextraction.md)
* [Spike deconvolution](deconvolution.md)
