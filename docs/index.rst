.. suite2p documentation master file, created by
   sphinx-quickstart on Sun Aug 18 15:27:04 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to suite2p's documentation!
===================================

suite2p is an imaging processing pipeline written in Python 3 which
includes the following modules:

-  Registration
-  Cell detection
-  Spike detection
-  Visualization GUI

For examples of how the output looks and how the GUI works, check out
this twitter `thread`_.

This code was written by Carsen Stringer and Marius Pachitariu. For
support, please open an `issue`_.

The reference paper is `here`_. The deconvolution algorithm is based on
`this paper`_, with settings based on `this
paper <http://www.jneurosci.org/content/early/2018/08/06/JNEUROSCI.3339-17.2018>`__.

We make pip installable releases of suite2p, here is the `pypi`_. You
can install it as ``pip install suite2p``

.. _thread: https://twitter.com/marius10p/status/1032804776633880583
.. _issue: https://github.com/MouseLand/suite2p/issues
.. _here: https://www.biorxiv.org/content/early/2017/07/20/061507
.. _this paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423
.. _pypi: https://pypi.org/project/suite2p/

* :ref:`modindex`
* :ref:`search`
* :ref:`genindex`

.. toctree::
   :maxdepth: 3
   :caption: Basics:

   installation
   inputs
   settings
   gui
   outputs
   multiday
   developer_doc
   FAQ

.. toctree::
   :maxdepth: 3
   :caption: How it works:

   registration
   celldetection
   roiextraction
   deconvolution

.. toctree::
   :maxdepth: 3
   :caption: API:

   api/suite2p.io
   api/suite2p.registration
   api/suite2p.detection
   api/suite2p.extraction
   api/suite2p.classification
   api/suite2p.gui

   
