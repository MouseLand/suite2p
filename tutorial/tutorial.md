This tutorial will take you through running suite2p and exploring the results in the GUI.

### 0. Download our example data, or use your own.

A short recording is available [here](https://drive.google.com/file/d/1Q8OT7mxn9_5jUg1vl48ZQZpw7OYMirrt/view?usp=sharing). It's a subset of frames from one plane in a 3-plane recording.

### 1. Install suite2p

There are more details on the readme, but in brief:

1. Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3.9** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Create a new environment with `conda create --name suite2p python=3.9`.
4. To activate this new environment, run `conda activate suite2p`
5. Install the GUI version with `python -m pip install suite2p[gui]`. If you're on a zsh server, you may need to use `' '` around the suite2p[gui] call: `python -m pip install 'suite2p[gui]'`.
7. Now run `python -m suite2p` and you're all set.
8. Running the command `suite2p --version` in the terminal will print the install version of suite2p.

For additional dependencies, like h5py, NWB, Scanbox, and server job support, use the command `python -m pip install suite2p[io]`. If using the zsh shell, make sure to use `' '` around the suite2p[io].

### 2. Run suite2p on the dataset

Click `File > Run suite2p`. This will open up a menu with options for running suite2p. Provide suite2p with the folder with the tiffs using `Add directory to data_path`. You can also change the input format with the drop-down menu. If you have an SSD on your computer you can change the `fast_disk` to a folder on the SSD -- this will speed up processing. See details about all parameters [here](https://suite2p.readthedocs.io/en/latest/settings.html).

There are a few parameters that are important to set: 
~~~~
    nplanes, nchannels, tau, fs
~~~~

For the tiff provided, this is `nplanes`=1, `nchannels`=1, `tau`=1.25, and `fs`=13. To be able to view the registered and unregistered data after running, turn on `keep_movie_raw` by setting it to 1 (this is recommended the first few times you're running new data through suite2p to help examine the registration quality).

Otherwise, we recommend using the default settings in most cases. For more zoomed in recordings, you may want to increase `spatial_hp_detect` to 40 or more. For datasets with a lot of nonrigid motion, you may want to decrease the `block_size` to 64, 64. 

You can enable PCA denoising of the data for detection with `denoise` = 1.

The `threshold_scaling` parameter can be reduced to find more cells, or increased to find fewer cells. Also, the number of iterations can be increased to find more cells -- the maximum number of cells found is 250 * `max_iterations`.

Click `RUN SUITE2P` to start the processing.

### 3. Explore the output

Once suite2p finishes running, you will see the output in the GUI, and you can close the run window. You can see more info [here](https://suite2p.readthedocs.io/en/latest/gui.html) about how to explore your data in the GUI. The main key commands are:

1. Pan  = Left-Click  + drag
2. Zoom = (Scroll wheel) OR (Right-Click + drag)
3. Full view = Double left-click OR escape key
4. Swap ROI label = Right-click on the ROI to changes its label (ie, cell to non-cell).
5. Select multiple cells = (Ctrl + left-click) OR (SHIFT + left-click) AND/OR ("Draw selection" button)

You will see ROIs classified as CELLS on the left, and ROIs classified as NOT CELLS on the right, classified using suite2p's default classifier. You can click on different cells with left-click to see their activity over time

### 4. Registration quality

Let's first look at the registration. Click on the menu option `Registration >> View registered binary`. A window will pop up with the binary file loaded (first row) along with the registration shifts (second row), and the fluorescence of a selected ROI (third row). The fourth row can be used for z-registration (not demo'ed here). Since we set `keep_movie_raw`=1, we can click the checkbox `view raw binary` and see the raw movie on the right side. You can select an ROI by typing in the ROI number in the upper right.

When not playing the movie, you can click on the shift plot and the fluorescence plot to go to a specific point in time in the movie. You can also seek through the movie by clicking the slide bar. The space bar will pause and play the movie. When paused the left and right arrow keys will move the slide bar incrementally. This can allow you to see if the registration looks good or bad.

Now let's quantify the quality of the registration. Click on the menu option `Registration >> View registration metrics`. A window will pop up with ops[‘regDX’] and ops[‘regPC’] plotted. The ops[‘regPC’]’s are computed by taking the principal components of the registered movie. ops['regPC'][0,0] is the average of the top 500 frames of the 1st PC, ops['regPC'][1,0] is the average of the bottom 500 frames of the 1st PC -- these are what are plotted in the 3 image plots. The first image is the “difference” between the top and the bottom of the PC. The second image is the “merged” image of the top and bottom of the PC. The third image allows you to flip between the top and bottom PCs using the “play” button. The left and right arrow keys will change the PC number (or you can type in a number). The space bar will pause and play the movie.

If you "play" this movie, ideally you will see different cells lighting up -- this means the PC is activity-based, that's good! (that's what it looks like in the demo tiff). If it looks instead like there are movements of cells in and out of the field of view or translating in the field of view, then this PC corresponds to motion -- this is bad. If the movement is in-plane (cells translating), then the registration could work better with better parameters potentially (maybe decreasing the `block_size` or increasing `maxregshiftNR`). But if the movement is out-of-plane, then no algorithm can fix your data. What you should hope then is that most cells' activity traces are not correlated with this PC over time, and also that any behavioral/other variables you are tracking are not related to this PC. You can see the PC over time in the upper right corner of the plot. You can see some examples of movements [here](https://twitter.com/marius10p/status/1051494533786193920).

More info about registration is available [here](https://suite2p.readthedocs.io/en/latest/registration.html#).

### 5. Cell detection

You can see all the ROIs detected if you go under the Colors bar and set `J: classifier, cell prob`= 0.0 and click enter -- this sets the cell probability threshold to 0.0. Now all ROIs will flip to the left side. Not all of these ROIs will be somatic. For example, you can see that some of them look more like dendrites (elongated), you can color the ROIs by that statistic by clicking `G: aspect_ratio` or typing the letter `g`, these you likely want classified as "NOT CELLS". You will also see very small ROIs, these are likely dendrites passing through the plane, or tips of cells. These we also probably don't want to use. You will also see some big frilly looking cells, these might be part of the neuropil (sums of dendrites) that we don't want to use as a cell either. These will often be classified as "NOT CELLS" because their traces will not be skewed -- you can color all the ROIs by skewness with `S: skew` or letter `s`.

We can build our own classifier but for now we'll be using the built-in classifier or default classifier that was used when we ran suite2p. This was trained using our own manual curation of GCaMP6s imaging of cells in cortex. Let's set the cell probability threshold to 0.25 and click enter. Now most of the elongated, smaller and/or frilly ROIs are on the right side. You can further classify ROIs yourself by right-clicking to flip the ROI to the other side. The assignment of the ROIs is updated each time you click / change the cell probability, and is available in the output file `iscell.npy`. 

The ROI statistics are available in `stat.npy`. You can see more info about this [here](https://suite2p.readthedocs.io/en/latest/outputs.html#stat-npy-fields). To revisit a past run of suite2p, click `File > Load processed data`. 

### 6. Signal extraction

From each ROI, we extract the mean activity in the ROI from each timepoint (weighted by the pixel mask in `stat['lam']`), this is the `F` fluorescence matrix saved in `F.npy`. We also compute the mean activity of the pixels surrounding the ROI -- the `Fneu` neuropil matrix saved in `Fneu.npy`. This neuropil activity contributes to the ROI itself so we correct the fluorescence trace of the ROI using the equation `F - 0.7*Fnew`. This corrected trace is then baselined over time and deconvolved to get an estimated spike rate at each timepoint for the ROI. Note that the scaling of this spike rate is arbitrary. Some discussion about it [here](https://suite2p.readthedocs.io/en/latest/FAQ.html#deconvolution-means-what).

When one cell is selected, the fluorescence, neuropil and deconvolved traces are shown for the chosen cell in the bottom row of the GUI. When multiple cells are selected, you can choose what type of traces to view with the "Activity mode" drop-down menu in the lower left: F: fluorescence; Fneu: neuropil fluorescence; F - 0.7*Fneu: corrected fluorescence; deconvolved: deconvolution of corrected and baselined fluorescence

You can resize the trace view with the triangle buttons (bigger = ▲, smaller = ▼). If multiple cells are selected, you can vary how much the traces overlap with the +/- buttons. You can select as many cells as you want, but by default only 40 of those will be plotted. You can increase or decrease this number by changing the number in the box below max # plotted.

The "Activity mode" is also used for the [Rastermap](https://github.com/mouseland/rastermap) visualization to explore patterns in the data -- choosing "deconvolved" is recommended. Click on the menu option `Visualizations >> Visualize selected cells`. This will either show selected cells (if you have selected more than one cell), or all cells on the side of the GUI on which you are clicked (e.g. select an ROI on the CELLS side to show all CELLS). This will open up a window to view all the traces. Click `compute rastermap + PCs` and then you'll see in the terminal that Rastermap is running. Once it runs, you'll see groups of neurons that are active together. You can then move the red box and click `show selected cells in GUI` to see which cells are active together. For more options when running Rastermap, run in a terminal with your `suite2p` environment `python -m rastermap` and then drag and drop your `spks.npy` file. See the Rastermap [github](https://github.com/mouseland/rastermap) for more details.
