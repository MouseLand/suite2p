import setuptools

install_deps = ['importlib-metadata',
        'natsort',
        'rastermap>0.1.0',
        'tifffile',
        'scanimage-tiff-reader>=1.4.1',
        'torch>=1.7.1',
        'paramiko',
        'numpy>=1.16',
        'numba>=0.43.1',
        'matplotlib',
        'scipy>=1.4.0',
        'h5py',
        'sbxreader',
        'scikit-learn',]

gui_deps = [
        "pyqt5",
        "pyqt5-tools",
        "pyqt5.sip",
        'pyqtgraph',
        'rastermap>0.1.0',
      ]
data_deps = [
        "dvc==1.11.0",
        "pydrive2",
      ]
nwb_deps = [
        "pynwb",
      ]
test_deps = [
      'pytest',
      'pytest-qt==3.3.0',
    ]

all_deps = gui_deps + data_deps + nwb_deps

try:
    import torch
    a = torch.ones(2, 3)
    version = int(torch.__version__[2])
    if version >= 6:
        install_deps.remove('torch')
except:
    pass

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
    install_requires=install_deps,
    tests_require=test_deps,
    extras_require={
      "docs": [
        'sphinx>=3.0',
        'sphinxcontrib-apidoc',
        'sphinx_rtd_theme',
        'sphinx-prompt',
        'sphinx-autodoc-typehints',
      ],
      "gui": gui_deps,
      "data": data_deps,
      "nwb": nwb_deps,
      "all": all_deps
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
