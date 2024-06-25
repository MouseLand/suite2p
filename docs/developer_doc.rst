Developer Documentation
---------------------------

Versioning
~~~~~~~~~~~~~~~~~~~~~
There's a rare issue that developers may face when calling `suite2p --version` on their command line. You
may get an incorrect version number. To fix this issue, one should use the following command:

.. prompt:: bash

        git fetch --prune --unshallow


Testing
~~~~~~~~~~~~~~~~~~~~~

Before contributing to Suite2P, please make sure your changes pass all our tests.

Downloading Test Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To run the tests (located in
the ``tests`` subdirectory of your working ``suite2p`` directory) , you'll first need to download our test data.
Suite2p depends on `dvc`_ to download the test data.

.. note::

    Before testing, make sure you have dvc and pydrive2 installed. Navigate to the suite2p
    directory and use the following command to install both dvc and pydrive2.

    .. prompt:: bash

        pip install -e .[data]

    zsh users should use the following:

    .. prompt:: bash

        pip install -e .\[docs\]


Use to following command to download the test data into the ``data`` subdirectory of your working ``suite2p`` directory.

.. prompt:: bash

    dvc pull

Running the tests
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Tests can then be easily run with the following command:

.. prompt:: bash

    python setup.py test

If all the tests pass, you're good to go!

.. _`dvc`: https://dvc.org/
