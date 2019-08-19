Multiday recordings
---------------------------------

In the matlab version of suite2p, Henry Dalgleish wrote the utility "registers2p" for multiday alignment, but it has not been ported to python.

I recommend trying to run all your recordings together (add all the separate folders to data_path). This has worked well for people who have automated online registration on their microscope to register day by day (scanimage 2018b (free) offers this capability). I highly recommend checking this out - we have contributed to a module in that software for online Z-correction that has greatly improved our recording quality.

However, if there are significant non-rigid shifts between days (angle changes etc) and low SNR then concatenating recordings and running them together will not work so well.

In this case, (if you have a matlab license) here is a package written by Adam Ranson which is based on similar concepts as 'registers2p' by Henry Dalgleish that takes the output of suite2p-python directly: https://github.com/ransona/ROIMatchPub.
