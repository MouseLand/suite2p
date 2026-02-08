# ROI Classification

After signal extraction, Suite2p classifies each ROI as a cell or not-cell using a weighted, non-parametric naive Bayes classifier. The settings for classification are in the `classification` dictionary.

## Features

The default classifier uses three features from each ROI's statistics, but the user can train a classifier with more statistics:

- **skew**: skewness of the neuropil-corrected fluorescence trace (cells tend to have positive skew due to transient calcium events)
- **npix_norm**: number of soma pixels normalized by the median ROI size (filters out ROIs that are too small or too large)
- **compact**: compactness of the ROI shape, where 1.0 is a disk and larger values indicate less compact shapes (cells tend to be compact)

## How the classifier works

1. **Non-parametric binning**: Each feature is binned into 100 bins, separately for the cell and not-cell classes from the training data. The probability of being a cell in each bin is smoothed with a Gaussian (standard deviation of 2 bins).

2. **Log-likelihood computation**: For each ROI, the log posterior likelihood ratio is computed per feature by looking up the ROI's feature value in the binned probability distributions.

3. **Logistic regression**: The log-likelihoods from each feature are combined using a logistic regression classifier (L2 regularization of 100, `liblinear` solver). This effectively learns the relative importance of each feature -- a simple naive Bayes classifier would weight all features equally.

4. **Output**: The classifier outputs a probability for each ROI. ROIs with probability > 0.5 are labeled as cells. The results are saved in `iscell.npy` as an (n_rois, 2) array, where column 0 is the binary label and column 1 is the probability. In the GUI the user can change the probability threshold and press enter to apply the new threshold.

## Classifier files

Suite2p selects a classifier in the following priority order:

1. **User-specified**: if `classifier_path` is set and the file exists, that classifier is used
2. **Built-in**: if `use_builtin_classifier` is True, or if no user classifier exists, the built-in classifier at `suite2p/classifiers/classifier.npy` is used
3. **User default**: otherwise, the user's saved classifier at `~/.suite2p/classifiers/classifier_user.npy` is used

### Training a custom classifier

You can train a custom classifier in the Suite2p GUI by manually labeling ROIs as cells or not-cells. The classifier is trained on the labeled data and saved for future use. This allows you to adapt the classifier to your specific recording conditions and cell types.

## Pre-classification (`preclassify`)

By default, classification runs after signal extraction. However, you can optionally apply classification **before extraction** by setting `preclassify` to a value greater than 0 (e.g. 0.5). This removes ROIs with classifier probability below the threshold before extraction proceeds. This step reduces the number of overlapping pixels excluded from trace extraction, which may improve SNR for remaining ROIs.

### How it differs from post-classification

Since `skew` requires an extracted fluorescence trace, pre-classification can only use the shape-based features `npix_norm` and `compact`. The `classify` function intersects the requested keys with the keys available in `stat`, so `skew` is automatically excluded when it hasn't been computed yet.

## Key parameters (`classification`)

| Parameter | Description |
|---|---|
| `preclassify` | Probability threshold for pre-classification; ROIs below this threshold are removed before extraction. Set to 0 to disable (default: 0) |
| `classifier_path` | Path to a custom classifier file (default: None, uses built-in or user classifier) |
| `use_builtin_classifier` | Use the built-in classifier instead of the user's saved classifier (default: False) |