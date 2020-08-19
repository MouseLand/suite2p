import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="suite2p",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="marius10p@gmail.com",
    description="Pipeline for calcium imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/suite2p",
    packages=setuptools.find_packages(),
    setup_requires=[
      'pytest-runner',
      'setuptools_scm',
    ],
    use_scm_version=True,
    install_requires=[
      'natsort',
      'rastermap>0.1.0',
      'tifffile',
      'scanimage-tiff-reader>=1.4.1',
      'pyqtgraph',
      'importlib-metadata',
      'paramiko'
    ],
    tests_require=[
      'pytest',
      'pytest-qt',
    ],
    extras_require={
      "docs": [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
        'sphinx-prompt',
        'sphinx-autodoc-typehints',
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
        "dvc>=1.1",
        "pydrive2",
      ],
      "nwb": [
        "pynwb",
      ]
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
      entry_points = {
        'console_scripts': [
          'suite2p = suite2p.__main__:main',
          'reg_metrics = benchmarks.registration_metrics:main',
          'tiff2scanimage = scripts.make_tiff_scanimage_compatible:main',
        ]
        },
)
