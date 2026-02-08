# Signal Extraction

After ROI detection, Suite2p extracts fluorescence time courses for each ROI and estimates the surrounding neuropil signal. The settings for extraction are in the `extraction` dictionary.

## Cell masks

The cell mask for each ROI is defined by its pixel coordinates (`ypix`, `xpix`) and pixel weights (`lam`). The weights are normalized to sum to 1 for each ROI, so that the fluorescence trace is the weighted average of the pixel intensities across frames.

By default, pixels that overlap between multiple ROIs are excluded from extraction. This can be changed by setting `allow_overlap` to True.

## Neuropil masks

The neuropil is the signal from dendrites and axons surrounding each ROI, captured within the point spread function of the microscope. Suite2p estimates the neuropil signal using a surround mask for each ROI.

### How neuropil masks are constructed

1. **Exclusion zone**: The ROI is extended outward by `inner_neuropil_radius` pixels. These pixels are excluded from the neuropil mask to avoid contamination from the cell itself.

2. **Surround region**: A square region is grown around the ROI until it contains at least `min_neuropil_pixels` non-cell pixels. Alternatively, setting `circular_neuropil` to True uses circular extension instead (slower).

3. **Cell pixel exclusion**: Pixels belonging to other ROIs are excluded from the neuropil mask. However, in dense fields of view with many ROIs, excluding all cell pixels would leave too few neuropil pixels. To handle this, an adaptive percentile filter is applied: a percentile filter with percentile `lam_percentile` is computed over a window of 5x the median ROI radius, and only pixels with `lam` weights above this threshold are excluded. This allows low-weight cell pixels to contribute to neuropil estimation.

4. **Equal weighting**: All pixels in the neuropil mask are weighted equally (each weight is 1 / number of neuropil pixels).

## Trace extraction

The fluorescence traces are extracted via sparse matrix multiplication between the mask matrices and the registered frames, processed in batches of `batch_size` frames. This operation is GPU-accelerated using PyTorch sparse tensors.

The outputs are:
- `F`: ROI fluorescence traces (n_rois by n_frames)
- `Fneu`: neuropil fluorescence traces (n_rois by n_frames)

If a second anatomical channel is present, the same masks are applied to extract `F_chan2` and `Fneu_chan2`.

## Neuropil correction and deconvolution

After extraction, the neuropil trace is subtracted from the ROI trace using the `neuropil_coefficient` (default 0.7):

```
Fc = F - neuropil_coefficient * Fneu
```

The corrected trace is then baselined and deconvolved (see [Spike deconvolution](deconvolution.md)).

## SNR-based ROI filtering

After extraction, an SNR estimate is computed for each ROI. If `snr_threshold` is greater than 0, ROIs with SNR below this threshold are removed, overlapping pixels are recomputed, and extraction is run a second time. This ensures that low-quality ROIs do not affect the fluorescence estimation of neighboring cells.

## Key parameters (`extraction`)

| Parameter | Description |
|---|---|
| `batch_size` | Number of frames processed per batch during extraction (default: 500) |
| `neuropil_coefficient` | Coefficient for neuropil subtraction before deconvolution (default: 0.7) |
| `allow_overlap` | If True, include overlapping pixels in cell masks for extraction (default: False) |
| `inner_neuropil_radius` | Number of pixels to pad around each ROI as an exclusion zone for neuropil masks (default: 2) |
| `min_neuropil_pixels` | Minimum number of pixels in each neuropil mask (default: 350) |
| `lam_percentile` | Percentile threshold for excluding cell pixels from neuropil masks; lower values exclude fewer pixels, helping in dense FOVs (default: 50) |
| `circular_neuropil` | If True, extend neuropil masks circularly instead of as rectangles (default: False) |
| `neuropil_extract` | If True, compute neuropil masks and extract neuropil traces (default: True) |
| `snr_threshold` | Minimum SNR for keeping an ROI; set to 0 to disable filtering (default: 0) |