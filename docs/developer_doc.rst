Developer Documentation
---------------------------

Testing
~~~~~~~~~~~~~~~~~~~~~

Before contributing to Suite2P, please make sure your changes pass all our tests. To run the tests (located in
the ``tests`` subdirectory of your working ``suite2p`` directory) , you'll first need to download our test data.
Suite2p depends on `dvc`_ to download the test data. Before testing, make sure you have dvc installed:

.. prompt:: bash

    pip install dvc

Use to following command to download the test data into the ``data`` subdirectory of your working ``suite2p`` directory.

.. prompt:: bash

    dvc pull

Tests can then be easily run with the following command:

.. prompt:: bash

    python setup.py test

If all the tests pass, you're good to go!

.. _`dvc`: https://dvc.org/
