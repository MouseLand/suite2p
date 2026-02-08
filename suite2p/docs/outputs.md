# Outputs

The pipeline saves its outputs as `.npy` files in each plane folder (`plane0/`, `plane1/`, etc.).

## Main output files

`F.npy`: array of fluorescence traces (n_rois by n_frames)

`Fneu.npy`: array of neuropil fluorescence traces (n_rois by n_frames)

`spks.npy`: array of deconvolved traces (n_rois by n_frames)

`stat.npy`: list of statistics computed for each ROI (length n_rois)

`iscell.npy`: specifies whether an ROI is a cell, first column is 0/1,
and second column is probability that the ROI is a cell based on the
default classifier (n_rois by 2)

`reg_outputs.npy`: registration outputs including shifts, reference image, and mean images (dictionary)

`detect_outputs.npy`: detection outputs including correlation and projection maps (dictionary)

`redcell.npy`: red channel cell detection scores, saved if a second anatomical channel is present (n_rois by 2)

`zcorr.npy`: correlations of registered frames with Z-stack, saved if a Z-stack is provided (n_frames by n_z)

All can be loaded in python with numpy:

```python
import numpy as np

F = np.load('suite2p/plane0/F.npy', allow_pickle=True)
Fneu = np.load('suite2p/plane0/Fneu.npy', allow_pickle=True)
spks = np.load('suite2p/plane0/spks.npy', allow_pickle=True)
stat = np.load('suite2p/plane0/stat.npy', allow_pickle=True)
reg_outputs = np.load('suite2p/plane0/reg_outputs.npy', allow_pickle=True).item()
detect_outputs = np.load('suite2p/plane0/detect_outputs.npy', allow_pickle=True).item()
iscell = np.load('suite2p/plane0/iscell.npy', allow_pickle=True)
```

## MATLAB output

If `'save_mat'=1`, then a MATLAB file is created `Fall.mat`. This
will contain F, Fneu, stat, spks and iscell. The "iscell"
assignments are only saved ONCE when the pipeline is finished running.
If you make changes in the GUI to the cell assignments, ONLY
`iscell.npy` changes. To load a modified `iscell.npy` into MATLAB, I
recommend using this package: [npy-matlab](https://github.com/kwikteam/npy-matlab). Alternatively there is a
*new* save button in the GUI (in the file menu) that allows you to save
the iscell again to the `Fall.mat` file.

## NWB Output

If `settings['save_NWB']=1`, then an NWB file is created `ophys.nwb`. This
will contain the fields from stat required to load back into the GUI, along
with F, Fneu, spks and iscell. If
the recording has multiple planes, then they are all saved together like in
combined view. See fields below:

`stat`: `stat['ypix'], stat['xpix']` (if multiplane `stat['iplane']`) are saved in
`'pixel_mask'` (called `'voxel_mask'` in multiplane).

`reg_outputs`: `'meanImg', 'max_proj', 'Vcorr'` are saved in Images `'Backgrounds_k'` where k is the plane
number, and have the same names. Optionally if two channels, `'meanImg_chan2'` is saved.

`iscell`: saved as an array `'iscell'`

`F,Fneu,spks` are saved as roi_response_series `'Fluorescence', 'Neuropil', and 'Deconvolved'`.

## Multichannel recordings

Cells are detected on the `settings['functional_chan']` and the fluorescence
signals are extracted from both channels. The functional channel signals
are saved to `F.npy` and `Fneu.npy`, and non-functional channel
signals are saved to `F_chan2.npy` and `Fneu_chan2.npy`.

## stat.npy fields

Each element of the `stat` array is a dictionary with the following keys:

### ROI pixels (set during detection)

- **ypix**: y-pixel coordinates of the ROI
- **xpix**: x-pixel coordinates of the ROI
- **lam**: pixel weights for fluorescence extraction (`sum(lam * frames[ypix,xpix,:])` = fluorescence trace)
- **med**: (y, x) center of the ROI
- **npix**: total number of pixels in the ROI
- **footprint**: initial template map ID from the Sparsery algorithm (0 for Sourcery/Cellpose)

### ROI shape statistics (computed after detection)

- **soma_crop**: boolean array indicating which pixels are in the somatic compartment, estimated from the radial profile of pixel weights
- **npix_soma**: number of pixels in the somatic compartment
- **compact**: compactness of the ROI (1.0 = disk, >1 = less compact than a disk)
- **mrs**: mean Euclidean distance of pixels from the ROI center
- **mrs0**: mean Euclidean distance expected for a disk with the same number of pixels
- **radius**: estimated radius from a 2D Gaussian fit to the pixel weights (2x the major axis radius)
- **aspect_ratio**: ratio between 2x the major axis radius and the sum of major and minor axis radii (ranges from 1.0 to 2.0; larger = more elongated)
- **npix_norm**: number of soma pixels normalized by the median size of the first 100 ROIs (or all ROIs for Cellpose)
- **npix_norm_no_crop**: same as `npix_norm` but using total `npix` instead of `npix_soma`
- **overlap**: boolean array indicating which pixels overlap with other ROIs (these are excluded from fluorescence extraction by default)

### Activity statistics (computed after extraction)

- **snr**: signal-to-noise ratio of the neuropil-corrected trace, estimated as `1 - var(diff(x)) / (2 * var(x))`
- **skew**: skewness of the neuropil-corrected fluorescence trace
- **std**: standard deviation of the neuropil-corrected fluorescence trace

Here is example code to make an image where each cell (without its
overlapping pixels) is a different "number":

```python
stat = np.load('stat.npy', allow_pickle=True)
reg_outputs = np.load('reg_outputs.npy', allow_pickle=True).item()

Ly, Lx = reg_outputs['meanImg'].shape
im = np.zeros((Ly, Lx))

for n in range(len(stat)):
    ypix = stat[n]['ypix'][~stat[n]['overlap']]
    xpix = stat[n]['xpix'][~stat[n]['overlap']]
    im[ypix,xpix] = n+1

plt.imshow(im)
plt.show()
```

## reg_outputs.npy fields

The registration outputs dictionary contains the following keys:

### Reference and mean images

- **refImg**: reference image used for registration (Ly x Lx)
- **meanImg**: mean of all registered frames (Ly x Lx)
- **meanImgE**: enhanced mean image (high-pass filtered version of meanImg)
- **meanImg_chan2**: mean image of the second channel, if present (Ly x Lx)

### Registration shifts

- **yoff**: rigid y-shifts at each timepoint (n_frames,)
- **xoff**: rigid x-shifts at each timepoint (n_frames,)
- **corrXY**: peak phase-correlation between each frame and the reference image (n_frames,)
- **yoff1**: non-rigid y-shifts per block at each timepoint (only if `nonrigid=True`)
- **xoff1**: non-rigid x-shifts per block at each timepoint (only if `nonrigid=True`)
- **corrXY1**: non-rigid phase-correlation per block (only if `nonrigid=True`)

### Frame quality

- **badframes**: boolean array of frames excluded from detection (n_frames,)
- **badframes0**: boolean array of bad frames detected before registration (n_frames,)
- **yrange**: valid y-range [ymin, ymax] for detection (excludes edges shifted out of FOV)
- **xrange**: valid x-range [xmin, xmax] for detection (excludes edges shifted out of FOV)

### Other

- **rmin**: lower intensity clip bound used for frame normalization
- **rmax**: upper intensity clip bound used for frame normalization
- **bidiphase**: estimated bidirectional phase offset in pixels
- **zpos_registration**: z-position estimates per frame (only for multi-plane with z-correction)
- **cmax_registration**: correlation max for z-position estimates (only for multi-plane with z-correction)

### Registration metrics (optional, computed if `do_regmetrics=True` and n_frames >= 1500)

- **tPC**: time courses of the principal components of the registered movie (n_subsample x n_PCs)
- **regPC**: spatial principal components, top and bottom averages (2 x n_PCs x Ly x Lx)
- **regDX**: registration shifts between top and bottom PC averages, quantifying residual motion

## detect_outputs.npy fields

The detection outputs dictionary contains the following keys:

- **max_proj**: maximum projection image across time (Ly_crop x Lx_crop)
- **meanImg_crop**: mean image of the cropped valid region
- **Vcorr**: correlation map showing regions of correlated pixels (used in GUI as the "correlation map")
- **diameter**: cell diameter used for detection [dy, dx]
- **Vmax**: maximum variance map (Sparsery only)
- **Vmap**: variance maps at multiple spatial scales (Sparsery only)
- **spatscale_pix**: estimated spatial scale in pixels (Sparsery only)
- **chan2_masks**: red channel masks (only if a second anatomical channel is present)