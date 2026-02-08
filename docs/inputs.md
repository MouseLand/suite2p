# Inputs

See all possible input options for `db.npy` [here](parameters.md#dbnpy)

## Input format

This applies to all file types!

### Directory structure

suite2p looks for all tiffs/hdf5 in the folders listed in
`db['data_path']`. If you want suite2p to look in those folders AND
all their children folders, set `db['look_one_level_down']=True`. If
you want suite2p to only look at some of the folder’s children, then set
`db['subfolders']` to those folder names.

If you want suite2p to only use specific tiffs in ONE folder, then set
the data path to only have one folder
(`db['data_path']=['my_folder_path']`), and name the tiffs you want
processed in `db['file_list']`.

See examples in this [notebook](https://github.com/MouseLand/suite2p/blob/master/jupyter/run_pipeline_tiffs_or_batch.ipynb).

### Frame ordering

If you have data with multiple planes and/or multiple channels, suite2p
expects the frames to be interleaved, e.g.

- frame0 = time0_plane0_channel1
- frame1 = time0_plane0_channel2
- frame2 = time0_plane1_channel1
- frame3 = time0_plane1_channel2
- frame4 = time1_plane0_channel1
- frame5 = time1_plane0_channel2
- …

channels are ones-based (channel 1 and 2 NOT 0 and 1).

<a id="recordings-with-photostim-other-artifacts"></a>

### Recordings with photostim / other artifacts

Photostim and other artifacts require you to exclude these frames during
ROI detection. Otherwise there will be “ROIs” that are related to the
stimulation, not actually cells. To exclude them, make an array of
integers corresponding to the frame times of the photostimulation. Save
this array into a numpy array called `bad_frames.npy`:

```default
import numpy as np

bad_frames = np.array([20,30,40])
np.save('bad_frames.npy', bad_frames)
```

Put this file into the first folder in your db[‘data_path’] (the first
folder you choose in the GUI).

<a id="inputs-diff-file-types"></a>

## Different file types

### Tiffs

Most tiffs should work out of the box. suite2p relies on two external
tiff readers: [scanimage-tiff-reader](http://scanimage.gitlab.io/ScanImageTiffReaderDocs/) and [sklearn.external.tifffile](http://scikit-image.org/docs/dev/api/skimage.external.tifffile.html).
The default is the scanimage one, but it will use the other one if it
errors.

You can use single-page tiffs. These will work out of the box if they
end in \*.tif or \*.tiff. NOTE that these will be slower to
load in and create the binary, so if you’re planning on using the
pipeline extensively you may want to change your acquisition output.

If you save a stack of tiffs using ImageJ, and it’s larger than 4GB,
then it may not run through suite2p. A work-around is to save as
an OME-TIFF in FIJI: “File->save as->OME-TIFF->compression type
uncompressed” in FIJI (thanks @kylemxm! see issue [here](https://github.com/MouseLand/suite2p/issues/149#issuecomment-473862374)).

If you have old Scanimage tiffs (version <5) that are larger than 2GB,
then most tiff readers will not work. @elhananby has recommended this [repository](https://github.com/dgreenberg/read_patterned_tifdata) for reading the data into matlab.
After reading it into matlab, you can re-save the tiff in a format that
imageJ and suite2p can recognize (see matlab tiff writing
[here](https://www.mathworks.com/help/matlab/ref/tiff.write.html)).

### Bruker

**Single Page Tifs**:
Using Bruker Prairie View system, .RAW files are batch converted to single .ome.tifs.
Now, you can load the resulting multiple tif files (i.e. one per frame per channel) to suite2p to be converted to binary.
This looks for files containing ‘Ch1’, and will assume all additional files are ‘Ch2’.
Select “input_format” as “bruker” in the drop down menu in the GUI or set `db['input_format'] = "bruker"`.

**Multi Page Tifs**:
To speed up the processing of input from bruker scopes, we recommend you save your .RAW files as multipage tifs.  This can be done using the Bruker Prairie View system.

In the PrairieView software, set your preferences to convert your raw files to multipage TIFFs.

* Preferences > Save Multipage TIFFs
* Preferences > Automatically Convert Raw Files > After Acquisition

This will cause the GUI to be unresponsive for some time after each acquisition. This should work for both single-channel and 2-channel recordings.

### Mesoscope tiffs

We have a matlab script
[here](https://github.com/MouseLand/suite2p/blob/master/helpers/mesoscope_json_from_scanimage.m)
for extracting the parameters from scanimage tiffs collected from the
Thorlabs mesoscope. The script creates an `ops.json` file that you can
then load into the run GUI using the button “load db/settings file”. This should
populate the run GUI with the appropriate parameters. Behind the scenes
there are `db['lines']` loaded and `db['dy'],db['dx']` that
specify which lines in the tiff correspond to each ROI and where in
space each ROI is respectively. `db['nplanes']` will only be greater
than 1 if you collected in multi-plane mode. Please open issues if
you’re using this and having trouble because it’s not straightforward.

<a id="hdf5-files-and-sbx"></a>

### HDF5 files (and \*.sbx)

These should work out of the box, but are less well-tested. Dario
Ringach has a utility to convert neurolabware \*.sbx files to \*.h5
files (see blog post
[here](https://scanbox.org/2018/08/29/using-suite2p-with-scanbox/)).

The H5 loading from the GUI now works the same as it always has for tiffs. Select
“h5” from the drop-down menu and input the h5 KEY for the data as a string. Now
choose the folder with your \*.h5 or \*.hdf5 files and the pipeline will use all
h5 files in that folder. You can use db[‘look_one_level_down’] to process all
subfolders of the data_path.

### sbx binary files

Scanbox binary files (*.sbx) work out of the box if you set `db['input_format'] = "sbx"`.

When recording in bidirectional mode some columns might have every other line saturated; to trim these during loading set `db['sbx_ndeadcols']`. Set this option to `-1` to let suite2p compute the number of columns automatically, a positive integer to specify the number of columns to trim. Joao Couto (@jcouto) wrote the binary sbx parser.

### Nikon nd2 files

Suite2p reads nd2 files using the nd2 package and returns a numpy array representing the data with a minimum of two dimensions (Height, Width). The data can also have additional dimensions for Time, Depth, and Channel. If any dimensions are missing, Suite2p adds them in the order of Time, Depth, Channel, Height, and Width, resulting in a 5-dimensional array. To use Suite2p with nd2 files, simply set `db['input_format'] = "nd2".`

## BinaryFile

The `BinaryFile` is a special class in suite2p that is used to read/write imaging data and acts like a Numpy Array. Inputs of any format listed above will be converted into a `BinaryFile` before being passed in through the suite2p pipeline. An input file can easily be changed to a `BinaryFile` in the following way:

```default
import suite2p

fname = "gt1.tif" # Let's say input is of shape (4200, 325, 556)
Lx, Ly = 556, 326 # Lx and Ly are the x and y dimensions of the imaging input
# Read in our input tif and convert it to a BinaryFile
f_input = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
```

`BinaryFile` can work with any of the input formats above. For instance, if you’d like to convert an input binary file, you can do the following:

```default
# Read in an input binary file and convert it to a BinaryRWFile
f_input2 = suite2p.io.BinaryRWFile(Ly=Ly, Lx=Lx, filename='gt1.bin')
```

Elements of these `BinaryFile` instances can be accessed similar to how one would access a Numpy Array.

```default
f_input.shape # returns shape of your input (num_frames, Ly, Lx)
f_input[0] # returns the first frame with shape (Ly, Lx)
```

Also, `BinaryFile` instances can be directly passed to the several wrapper functions `suite2p` offers (e.g., `suite2p.detection_wrapper`, `suite2p.extraction_wrapper`, etc.). These wrapper functions can  also directly work with Numpy arrays so feel free to pass them as inputs. If you’d like to run only specific modules, you will have to use the `BinaryFile` class. For example, this is how you can run the detection module on an input file that has already been registered.

```default
f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='data.bin')
detect_outputs, stat = suite2p.detection_wrapper(f_reg=f_reg, settings=settings)[:2]
```
