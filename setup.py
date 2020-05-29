import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="suite2p",
    version="0.7.5",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="marius10p@gmail.com",
    description="Pipeline for calcium imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/suite2p",
    packages=setuptools.find_packages(),
    setup_requires=[
      'pytest-runner',
    ],
    install_requires=[
      'numpy>=1.16',
      'numba>=0.43.1',
      'tbb',  # Parallel processing library used by numba.  Needed when installing numba from pip  https://github.com/numba/numba/issues/4068
      'matplotlib',
      'scipy',
      'h5py',
      'scikit-learn',
      'natsort',
      'rastermap>0.1.0',
      'tifffile',
      'scanimage-tiff-reader!=1.4.1',
    ],
    tests_require=[
      'pytest',
      'tqdm',
    ],
    extras_require={
      "docs": [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
      ],
      # Note: Available in pypi, but cleaner to install as pyqt from conda.
      "gui": [
        "pyqt5",
        "pyqt5-tools",
        "pyqt5.sip",
      ],
      # Note: Not currently available in pip: use conda to install.
      "mkl": [
        "mkl_fft>=1.0.10",
        "mkl>=2019.3",
      ],
      "data": [
        "dvc",
        "pydrive2",
      ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
      entry_points = {
        'console_scripts': [
          'suite2p = suite2p.__main__:parse_arguments',
        ]
        },
)
