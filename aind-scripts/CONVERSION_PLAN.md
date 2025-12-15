# AIND to Suite2p Data Conversion Plan

## Overview
This document outlines the plan to convert AIND multiplane ophys data format to suite2p-compatible format. The conversion will enable loading AIND processed data into the suite2p GUI for visualization and further analysis.

## Input Data Structure (AIND Format)

### Directory Structure
```
<dataset_name>/
├── VISp_0/                          # Plane 0
│   ├── VISp_0.h5                    # Raw movie data
│   ├── extraction/
│   │   └── VISp_0_extraction.h5     # ROI extraction results
│   ├── classification/
│   │   └── VISp_0_classification.h5 # Cell classification
│   ├── motion_correction/
│   │   └── VISp_0_registered.h5     # Motion corrected data
│   ├── events/
│   │   └── VISp_0_events_oasis.h5   # Spike deconvolution
│   ├── dff/
│   │   └── VISp_0_dff.h5            # Delta F/F
│   └── decrosstalk/
│       └── ...                       # Crosstalk correction
├── VISp_1/                          # Plane 1
├── ...                              # Additional planes
└── metadata files...
```

### H5 File Contents

#### extraction/VISp_X_extraction.h5
- **cellpose/**: Cellpose segmentation outputs
  - `masks` (512, 512): ROI masks
  - `cellprob` (512, 512): Cell probability map
  - `flows` (2, 512, 512): Flow fields
- **rois/**: ROI properties
  - `coords` (3, N): Sparse coordinates [roi_id, y, x]
  - `data` (N,): Pixel weights for each coordinate
  - `med` (n_rois, 2): Center positions [y, x]
  - `npix` (n_rois,): Number of pixels per ROI
  - `npix_soma` (n_rois,): Number of soma pixels
  - `npix_norm` (n_rois,): Normalized pixel count
  - `radius` (n_rois,): ROI radius
  - `aspect_ratio` (n_rois,): Aspect ratio
  - `compact` (n_rois,): Compactness metric
  - `solidity` (n_rois,): Solidity metric
  - `footprint` (n_rois,): Spatial footprint
  - `overlap` (N,): Boolean overlap flags
  - `soma_crop` (N,): Boolean soma crop flags
  - `neuropil_coords` (3, M): Neuropil mask coordinates
- **traces/**: Fluorescence traces
  - `roi` (n_rois, n_frames): Raw ROI traces
  - `neuropil` (n_rois, n_frames): Neuropil traces
  - `corrected` (n_rois, n_frames): Neuropil-corrected traces
  - `neuropil_rcoef` (n_rois,): Neuropil correction coefficients
  - `skew` (n_rois,): Trace skewness
  - `std` (n_rois,): Trace standard deviation
- **Image stats**:
  - `meanImg` (512, 512): Mean image
  - `maxImg` (512, 512): Max projection
- **Cell classification**:
  - `iscell` (n_rois, 2): [is_cell, probability]

#### classification/VISp_X_classification.h5
- `soma/predictions` (n_rois,): Binary soma predictions
- `soma/probabilities` (n_rois, 2): Soma class probabilities
- `dendrites/predictions` (n_rois,): Binary dendrite predictions
- `dendrites/probabilities` (n_rois, 2): Dendrite class probabilities
- `border/labels` (n_rois,): Border cell labels

#### motion_correction/VISp_X_registered.h5
- `data` (n_frames, 512, 512): Registered movie
- `ref_image` (512, 512): Reference image for registration
- `reg_metrics/`: Registration quality metrics
  - `regDX` (n_chunks, 3): [y_shift, x_shift, correlation]
  - `crispness` (2,): Image crispness metrics

#### events/VISp_X_events_oasis.h5
- `events` (n_rois, n_frames): Deconvolved spike events
- `denoised` (n_rois, n_frames): Denoised traces
- `b_hat` (n_rois,): Baseline estimates
- `tau_hat` (n_rois,): Time constant estimates
- `lam_hat` (n_rois,): Lambda parameters

#### dff/VISp_X_dff.h5
- `data` (n_rois, n_frames): Delta F/F traces
- `baseline` (n_rois, n_frames): Baseline estimates
- `noise` (n_rois,): Noise estimates
- `skewness` (n_rois,): Trace skewness

## Output Data Structure (Suite2p Format)

### Directory Structure
```
<output_path>/
├── <dataset_name>/
│   ├── VISp_0/                  # Plane 0
│   │   ├── F.npy
│   │   ├── Fneu.npy
│   │   ├── spks.npy
│   │   ├── stat.npy
│   │   ├── ops.npy
│   │   ├── iscell.npy
│   │   └── iscell_alt.npy      # Alternative classification from soma classifier
│   ├── VISp_1/                  # Plane 1
│   └── ...
```

### Required Output Files

#### F.npy
- **Shape**: (n_rois, n_frames)
- **Type**: float32
- **Description**: Fluorescence traces (neuropil-corrected)
- **Source**: `extraction/traces/corrected`

#### Fneu.npy
- **Shape**: (n_rois, n_frames)
- **Type**: float32
- **Description**: Neuropil fluorescence traces
- **Source**: `extraction/traces/neuropil`

#### spks.npy
- **Shape**: (n_rois, n_frames)
- **Type**: float32
- **Description**: Deconvolved spike estimates
- **Source**: `events/events`

#### iscell.npy
- **Shape**: (n_rois, 2)
- **Type**: float32 [binary, probability]
- **Description**: Cell classification (primary)
- **Source**: `extraction/iscell`
- **Note**: Column 0 = binary (0/1), Column 1 = probability

#### iscell_alt.npy (ADDITIONAL)
- **Shape**: (n_rois, 2)
- **Type**: float32 [binary, probability]
- **Description**: Alternative cell classification from soma classifier
- **Source**: `classification/soma/probabilities`
- **Note**: Column 0 = binary prediction, Column 1 = probability of soma class

#### stat.npy
- **Type**: List of dictionaries (numpy object array)
- **Length**: n_rois
- **Description**: Per-ROI statistics
- **Fields** (see mapping below)

#### ops.npy
- **Type**: Dictionary (numpy object)
- **Description**: Pipeline parameters and intermediate outputs
- **Fields** (see mapping below)

## Field Mapping: AIND → Suite2p

### stat.npy Field Mapping

| Suite2p Field | AIND Source | Notes |
|---------------|-------------|-------|
| `ypix` | `rois/coords[1]` for each ROI | Extract y coordinates where coords[0] == roi_id |
| `xpix` | `rois/coords[2]` for each ROI | Extract x coordinates where coords[0] == roi_id |
| `lam` | `rois/data` for each ROI | Extract weights where coords[0] == roi_id |
| `med` | `rois/med` | Already [y, x] format |
| `npix` | `rois/npix` | Direct copy |
| `npix_norm` | `rois/npix_norm` | Direct copy |
| `npix_soma` | `rois/npix_soma` | Direct copy |
| `radius` | `rois/radius` | Direct copy |
| `aspect_ratio` | `rois/aspect_ratio` | Direct copy |
| `compact` | `rois/compact` | Direct copy |
| `footprint` | `rois/footprint` | Direct copy |
| `skew` | `traces/skew` | Direct copy |
| `std` | `traces/std` | Direct copy |
| `overlap` | `rois/overlap` for each ROI | Extract boolean mask where coords[0] == roi_id |
| `soma_crop` | `rois/soma_crop` for each ROI | Extract boolean mask where coords[0] == roi_id |
| `neuropil_mask` | `rois/neuropil_coords` for each ROI | **OPTIONAL** - can skip if not needed |
| `solidity` | `rois/solidity` | Direct copy |

**Missing/Skippable Fields:**
- `chan2_prob` - Skip if no channel 2
- `inmerge` - Skip (for merged ROIs)
- `iplane` - Add as plane index for multiplane data

### ops.npy Field Mapping

| Suite2p Field | AIND Source | Notes |
|---------------|-------------|-------|
| `Ly` | From movie shape | 512 (height) |
| `Lx` | From movie shape | 512 (width) |
| `nframes` | From movie shape | e.g., 8918 |
| `meanImg` | `extraction/meanImg` | Direct copy |
| `max_proj` | `extraction/maxImg` | Direct copy |
| `refImg` | `motion_correction/ref_image` | Reference image for registration |
| `yrange` | Compute from image | [0, Ly] or compute valid region |
| `xrange` | Compute from image | [0, Lx] or compute valid region |
| `Vcorr` | **MISSING** | Skip - correlation map not in AIND format |
| `meanImgE` | **MISSING** | Skip - can compute if needed |
| `xoff` | `motion_correction/reg_metrics/regDX[:,1]` | X-shifts per chunk, may need interpolation |
| `yoff` | `motion_correction/reg_metrics/regDX[:,0]` | Y-shifts per chunk, may need interpolation |
| `corrXY` | `motion_correction/reg_metrics/regDX[:,2]` | Correlation per chunk |

**Additional ops fields to add (for GUI compatibility):**
- `nplanes` = number of plane folders found
- `nchannels` = 1 (or 2 if channel 2 data exists)
- `tau` = mean of `events/tau_hat` (or default 1.0)
- `fs` = sampling rate from `processing.json` → `movie_frame_rate_hz` field
- `data_path` = list of source directories
- `save_path` = output directory path
- `date_proc` = current timestamp

**Skippable ops fields:**
- `reg_file` - Skip (binary file path)
- `filelist` - Skip (original tiff files)
- `badframes` - Skip if not available
- Registration parameters (already done) - use defaults

## Conversion Algorithm

### Step 1: Discovery
1. Find all plane folders (VISp_0, VISp_1, etc.)
2. Verify required H5 files exist in each plane
3. Load processing.json for each plane to extract metadata (sampling rate, etc.)

### Step 2: Per-Plane Conversion
For each plane folder:

1. **Load AIND H5 files**
   - extraction/VISp_X_extraction.h5
   - classification/VISp_X_classification.h5 (optional)
   - motion_correction/VISp_X_registered.h5
   - events/VISp_X_events_oasis.h5
   - dff/VISp_X_dff.h5 (optional, for reference)

2. **Convert to numpy arrays**
   - F.npy: `extraction/traces/corrected`
   - Fneu.npy: `extraction/traces/neuropil`
   - spks.npy: `events/events`
   - iscell.npy: `extraction/iscell`
   - iscell_alt.npy: `classification/soma/probabilities` (convert predictions to binary for col 0)

3. **Build stat.npy**
   - Create list of dictionaries
   - For each ROI, extract coordinates and properties from sparse format
   - Reconstruct ypix, xpix, lam from coords/data arrays

4. **Build ops.npy**
   - Create dictionary with required fields
   - Extract images (meanImg, max_proj, refImg)
   - Extract motion correction shifts
   - Add metadata and parameters

5. **Save to suite2p folder structure**VISp_<i>/`
   - Save all .npy files (F, Fneu, spks, stat, ops, iscell, iscell_alt)h>/<dataset_name>/plane<i>/`
   - Save all .npy files

### Step 3: Validation
- Check file shapes are consistent
- Verify n_rois matches across all files
- Verify n_frames matches across F, Fneu, spks
- Print summary statistics

## Implementation Decisions

### D1: Sampling rate (fs)
**Source:** Extract from each plane's `processing.json` file
- Look for `processing_pipeline.data_processes[*].parameters.movie_frame_rate_hz`
- Example: 9.48 Hz for the test dataset
- This is the per-plane frame rate

### D2: Cell classification
**Primary:** Use `extraction/iscell` (shape: n_rois, 2)
- Column 0: binary is_cell flag (0 or 1)
- Column 1: probability

**Alternative:** Also save `classification/soma/probabilities` as `iscell_alt.npy`
- Column 0: binary prediction (soma class = 1)
- Column 1: probability of soma class
- This provides an alternative classifier result for comparison

### D3: Output structure
**Format:** `<output_path>/<dataset_name>/VISp_<i>/`
- How to handle missing fields?
**Decision:** 
- **Required fields**: Error if missing (F, Fneu, stat basics)
- **Optional fields**: Set to None or skip (Vcorr, meanImgE)
- **Computable fields**: Compute if easy (yrange, xrange)

### Q3: Multi-plane indexing
**Decision:**
- Add `iplane` field to each stat entry
- Number planes by extracting index from folder name (VISp_0 → iplane=0)
- Preserve original folder names in output

### Q4i_id in range(n_rois):
    mask = coords[0] == roi_id
    ypix = coords[1][mask]
    xpix = coords[2][mask]
    lam = data[mask]
    overlap = overlap_flags[mask]
```

### Q2: Should we include optional files?
**Decision:** Include if source data exists:
- DFF traces - Store as separate file or in ops?
- Classification probabilities - Useful for GUI
- Cellpose masks - Useful for visualization

### Q3: How to handle missing fields?
**Decision:** 
- **Required fields**: Error if missing (F, Fneu, stat basics)
- **Optional fields**: Set to None or skip (Vcorr, meanImgE)
- **Computable fields**: Compute if easy (yrange, xrange)

### Q4: Multi-plane indexing
**Decision:**
- Add `iplane` field to each stat entry
- Number planes starting from 0
- Plane folder name mapping: VISp_0 → plane0, VISp_1 → plane1, etc.

### Q5: Should we support combined view?
**Decision:** 
- Phase 1: Individual planes only
- Phase 2: Consider adding combined view support

## Configuration

### Script Parameters
- `input_path`: Path to AIND dataset root
- `output_path`: Path for suite2p output
- `dataset_name`: Name of dataset (for output folder)
- `plane_pattern`: Regex pattern to find plane folders (default: "VISp_\d+")
- `overwrite`: Whether to overwrite existing output
- `validate`: Whether to run validation checks
 (contains VISp_X folders)
- `output_path`: Path for suite2p output root
- `dataset_name`: Name of dataset (for output folder structure)
- `plane_pattern`: Regex pattern to find plane folders (default: r"VISp_\d+")
- `overwrite`: Whether to overwrite existing output
- `validate`: Whether to run validation checks

### Example Usage
```bash
python convert_aind_to_suite2p.py \
    --input /s3-cache/suite2p-dev/multiplane-ophys_827543_2025-12-11_14-26-35_processed_2025-12-12_13-39-05 \
    --output /output/suite2p_converted \
    --dataset-name multiplane-ophys_827543 \
    --validate
```

**Output structure:**
```
/output/suite2p_converted/
└── multiplane-ophys_827543/
    ├── VISp_0/
    │   ├── F.npy
    │   ├── Fneu.npy
    │   ├── spks.npy
    │   ├── stat.npy
    │   ├── ops.npy
    │   ├── iscell.npy
    │   └── iscell_alt.npy
    ├── VISp_1/
    └── ...se coordinate unpacking
   - Test field mapping for single ROI
   - Test ops dictionary creation

2. **Integration test**
   - Convert single plane
   - Load in suite2p GUI
   - Verify visualization works

3. **Full dataset test**
   - Convert all 8 planes
   - Load in suite2p GUI
   - Compare statistics with original AIND outputs

## Success Criteria

1. ✅ All required .npy files are created
2. ✅ Data loads in suite2p GUI without errors
3. ✅ ROI visualizations match original segmentation
4. ✅ Fluorescence traces display correctly
5. ✅ Cell classification labels are preserved
6. ✅ Multi-plane navigation works (if implemented)

## Future Enhancements

- Support for channel 2 data (if available in AIND format)
- Combined view across planes
- Preserve more metadata from AIND processing
- Bidirectional conversion (suite2p → AIND)
- Support for crosstalk-corrected data
