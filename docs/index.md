# Welcome to suite2p's documentation!

![image](_static/favicon.ico)

suite2p is an imaging processing pipeline written in Python 3 which
includes the following modules:

- Registration
- ROI detection
- Signal extraction
- ROI classification
- Spike deconvolution
- Visualization GUI

For examples of how the output looks and how the GUI works, check out
this twitter [thread](https://twitter.com/marius10p/status/1032804776633880583).

This code was written by Carsen Stringer and Marius Pachitariu. For
support, please open an [issue](https://github.com/MouseLand/suite2p/issues).

The reference paper is [here](https://www.biorxiv.org/content/10.64898/2026.02.04.703741v1). The deconvolution algorithm is based on
[this paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423), with settings based on [this
paper](http://www.jneurosci.org/content/early/2018/08/06/JNEUROSCI.3339-17.2018).

We make pip installable releases of suite2p, here is the [pypi](https://pypi.org/project/suite2p/). You
can install it as `pip install suite2p`

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
    * [HDF5 files (and \*.sbx)](inputs.md#hdf5-files-and-sbx)
    * [sbx binary files](inputs.md#sbx-binary-files)
    * [Nikon nd2 files](inputs.md#nikon-nd2-files)
  * [BinaryFile](inputs.md#binaryfile)
* [Parameters](parameters.md)
  * [db.npy](parameters.md#dbnpy)
  * [general settings](parameters.md#general-settings)
  * [run](parameters.md#run)
  * [io](parameters.md#io)
  * [registration](parameters.md#registration)
  * [detection](parameters.md#detection)
  * [classification](parameters.md#classification)
  * [extraction](parameters.md#extraction)
  * [dcnv preprocess](parameters.md#dcnv-preprocess)
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
  * [Main output files](outputs.md#main-output-files)
  * [MATLAB output](outputs.md#matlab-output)
  * [NWB Output](outputs.md#nwb-output)
  * [Multichannel recordings](outputs.md#multichannel-recordings)
  * [stat.npy fields](outputs.md#statnpy-fields)
  * [reg_outputs.npy fields](outputs.md#reg_outputsnpy-fields)
  * [detect_outputs.npy fields](outputs.md#detect_outputsnpy-fields)
* [Multiday recordings](multiday.md)
* [Developer Documentation](developer_doc.md)
  * [Testing](developer_doc.md#testing)
* [Frequently Asked Questions](FAQ.md)
  * [Cropped field-of-view](FAQ.md#cropped-field-of-view)
  * [Deconvolution means what?](FAQ.md#deconvolution-means-what)
  * [Multiple functional channels](FAQ.md#multiple-functional-channels)
  * [Z-drift](FAQ.md#z-drift)
  * [No signals in manually selected ROIs](FAQ.md#no-signals-in-manually-selected-rois)

# How it works:

* [Registration](registration.md)
  * [Bidirectional phase offset](registration.md#bidirectional-phase-offset-optional)
  * [Reference image computation](registration.md#reference-image-computation)
  * [Rigid registration](registration.md#rigid-registration)
  * [Non-rigid registration](registration.md#non-rigid-registration)
  * [Valid region estimation](registration.md#valid-region-estimation)
  * [Two-step registration](registration.md#two-step-registration-optional)
  * [Registration metrics](registration.md#registration-metrics)
  * [Key parameters](registration.md#key-parameters-registration)
* [ROI Detection](roidetection.md)
  * [Preprocessing](roidetection.md#preprocessing)
  * [Sparsery (default)](roidetection.md#sparsery-default)
  * [Sourcery](roidetection.md#sourcery)
  * [Cellpose](roidetection.md#cellpose)
  * [Post-detection filtering](roidetection.md#post-detection-filtering)
* [Signal Extraction](roiextraction.md)
  * [Cell masks](roiextraction.md#cell-masks)
  * [Neuropil masks](roiextraction.md#neuropil-masks)
  * [Trace extraction](roiextraction.md#trace-extraction)
  * [Neuropil correction and deconvolution](roiextraction.md#neuropil-correction-and-deconvolution)
  * [SNR-based ROI filtering](roiextraction.md#snr-based-roi-filtering)
  * [Key parameters](roiextraction.md#key-parameters-extraction)
* [Spike deconvolution](deconvolution.md)
* [ROI Classification](classification.md)
  * [Features](classification.md#features)
  * [How the classifier works](classification.md#how-the-classifier-works)
  * [Classifier files](classification.md#classifier-files)
  * [Pre-classification](classification.md#pre-classification-preclassify)
  * [Key parameters](classification.md#key-parameters-classification)