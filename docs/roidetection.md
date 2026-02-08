# ROI Detection

Suite2p provides three algorithms for detecting regions of interest (ROIs), specified by the `algorithm` setting: `'sparsery'` (default), `'sourcery'`, or `'cellpose'`. The first two use functional information from the movie, while Cellpose uses anatomical information from summary images.

## Preprocessing

Before any detection algorithm runs, the registered movie is binned in time. The bin size is set to `fs * tau` (the indicator decay timescale in frames), since consecutive frames within this window contain redundant information. The total number of bins is capped at `settings['detection']['nbins']`. Optionally, PCA denoising can be applied to the binned movie by setting `settings['detection']['denoise']` to True. The movie is then high-pass filtered in time by subtracting a Gaussian-smoothed version of itself (standard deviation `settings['detection']['highpass_time']`), and a maximum projection image (`max_proj`) is computed.

## Sparsery (default)

Sparsery is the main detection algorithm. It performs a greedy matrix decomposition on the movie, assuming **L0-sparse sources in space and L0-sparse traces in time**. This means ROIs are only detected if they are spatially localized and strongly active on at least a few frames.

### How it works

1. **Neuropil removal**: Each binned frame is divided by the per-pixel shot noise estimate and high-pass filtered spatially with a uniform filter of size `settings['detection']['sparsery_settings']['highpass_neuropil']` pixels.

2. **Multi-scale template matching**: Variance explained maps are computed for uniform square templates of sizes 3x3, 6x6, 12x12, 24x24, and 48x48 pixels. The optimal spatial scale is estimated automatically from the peaks of these maps, or can be set manually with `settings['detection']['sparsery_settings']['spatial_scale']` (integer 1-4, corresponding to 6x6 through 48x48).

3. **Iterative ROI extraction**: On each iteration, the location with the highest variance explained is selected as a candidate ROI. Active frames are identified as those exceeding a temporal threshold `Th2 = 5 * spatial_scale * threshold_scaling`. The ROI mask is refined by iteratively extending pixels whose mean intensity on active frames exceeds one-fifth of the maximum. Activity is subtracted from the movie before searching for the next ROI.

4. **Splitting check**: Each candidate ROI is tested to see if splitting it into two ROIs would improve the variance explained, using iterative k-means.

Detection stops when the variance explained falls below a threshold scaled by `settings['detection']['threshold_scaling']` or when `settings['detection']['sparsery_settings']['max_ROIs']` is reached.

### Key parameters (`sparsery_settings`)

| Parameter | Description |
|---|---|
| `threshold_scaling` | Multiplier on the temporal threshold for detecting ROIs. Lower values find more ROIs, higher values find fewer (default: 1.0) |
| `highpass_neuropil` | Spatial high-pass filter size in pixels for neuropil removal, set higher if zoom is high - should be ~3x the diameter of the cell in pixels (default: 25) |
| `max_ROIs` | Maximum number of ROIs to detect, set larger if hitting limit (default: 5000) |
| `spatial_scale` | Override automatic scale estimation, which is used for determining thresholds (valid inputs are 1-4, corresponding to template sizes 6x6 to 48x48). Set to 0 for automatic (default: 0) |

## Sourcery

Sourcery also uses functional information but relies more on **overall pixel correlations** rather than sparse transient activity. It can be useful for **low-SNR data with compact ROIs**.

### How it works

1. **PCA and neuropil projection**: The binned movie is orthogonalized via PCA to obtain spatial pixel-wise PCs. A low-spatial-frequency neuropil basis (sines and cosines) is projected out of the PCs to avoid extracting large neuropil components.

2. **Correlation map**: The neuropil-subtracted PCs are spatially smoothed and a correlation map is computed, highlighting regions where nearby pixels are temporally correlated. Local peaks in this map above `settings['detection']['threshold_scaling']` times the median peak value are used as ROI seeds.

3. **Iterative ROI extraction**: Starting from the highest peak, each ROI is initialized and iteratively extended pixel-by-pixel. Pixel weights are computed as the dot product between the local PCs and the ROI's temporal signature. After each group of 200 ROIs, the neuropil and ROI temporal components are re-estimated via L2-regularized linear regression. This extraction continues until the ROI seeds are below the threshold or when `settings['detection']['sourcery_settings']['max_iterations']` is reached.

4. **Refinement**: Two final iterations refine the spatial components using un-smoothed PCs (if `settings['detection']['sourcery_settings']['smooth_masks']` is False), and ROIs are optionally reduced to their connected components (if `settings['detection']['sourcery_settings']['connected']` is True).

### Key parameters (`sourcery_settings`)

| Parameter | Description |
|---|---|
| `threshold_scaling` | Multiplier on the correlation map threshold for ROI seeds. Lower values find more ROIs |
| `diameter` | Expected ROI diameter in pixels; used for spatial smoothing and neuropil basis frequency |
| `max_iterations` | Maximum number of ROI detection iterations |
| `smooth_masks` | If False, final ROI masks use un-smoothed PCs (default: False) |
| `connected` | If True, reduce ROIs to connected components before final refinement (default: True) |

## Cellpose

Cellpose is an **anatomical segmentation algorithm** that detects ROIs from summary images rather than from the movie dynamics. It uses the Cellpose-SAM model by default (to use older cellpose versions, `pip install cellpose==3.1.1.2`).

### How it works

1. **Image selection**: The image used for segmentation is specified by `settings['detection']['cellpose_settings']['img']` and can be:
   - `'max_proj / meanImg'` -- ratio of the maximum projection and the mean image (default)
   - `'meanImg'` -- the mean image
   - `'max_proj'` -- the maximum projection image

2. **Preprocessing**: The selected image can be high-pass filtered spatially with `settings['detection']['cellpose_settings']['highpass_spatial']` before running Cellpose.

3. **Segmentation**: Cellpose segments the image into individual ROIs using the model specified by `settings['detection']['cellpose_settings']['cellpose_model']`.

4. **Pixel weights**: For `'max_proj / meanImg'` or `'max_proj'`, the pixel weights (`lam`) are set from the maximum projection. For `'meanImg'`, the weights are the mean image clipped at the 1st and 99th percentiles and scaled between 0.1 and 1.1.

### Key parameters

| Parameter | Description |
|---|---|
| `diameter` | Expected cell diameter in pixels, passed to Cellpose |
| `img` | Which image to segment: `'max_proj / meanImg'`, `'meanImg'`, or `'max_proj'` |
| `highpass_spatial` | Spatial high-pass filter size applied to the image before segmentation |
| `cellpose_model` | Cellpose model to use for segmentation |

## Post-detection filtering

After ROI detection (regardless of algorithm), several optional filters can remove ROIs before signal extraction:

| Parameter | Description |
|---|---|
| `max_overlap` | Remove ROIs with more than this fraction of pixels overlapping other ROIs (default: 0.75) |
| `npix_norm_min` | Minimum normalized ROI size for inclusion |
| `npix_norm_max` | Maximum normalized ROI size for inclusion |
| `preclassify` | Threshold for a shape/size classifier applied before extraction (uses `npix_norm` and `compact` features, see [classification](classification.md)) |