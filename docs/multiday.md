# Multiday recordings

In the matlab version of suite2p, Henry Dalgleish wrote the utility “registers2p” for multiday alignment, but it has not been ported to python.

We recommend trying to run all your recordings together (add all the separate folders to data_path). This has worked well for people who have automated online registration on their microscope to register day by day (scanimage 2018b (free) offers this capability). We highly recommend checking this out - we have contributed to a module in that software for online Z-correction that has greatly improved our recording quality.

However, if there are significant non-rigid shifts between days (angle changes etc) and low SNR then concatenating recordings and running them together will not work so well.

If this is the case, in python, we recommend the package [ROICaT](https://github.com/RichieHakim/ROICaT) by Richard Hakim, or the package [Track2p](https://track2p.github.io/home.html) from Majnik et al 2025. In matlab, we recommend [ROIMatchPub](https://github.com/ransona/ROIMatchPub) by Adam Ranson (which is based on similar concepts as 'registers2p' by Henry Dalgleish). All these algorithms take the outputs of Suite2p directly.
