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
