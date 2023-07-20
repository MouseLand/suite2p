import setuptools

install_deps = ["importlib-metadata",
        "natsort",
        "rastermap>0.1.0",
        "tifffile",
        "torch>=1.13.1",
        "numpy>=1.24.3",
        "numba>=0.57.0",
        "matplotlib",
        "scipy>=1.9.0",
        "scikit-learn",
        "cellpose",
        ]

gui_deps = [
        "pyqt5",
        "pyqt5-tools",
        "pyqt5.sip",
        "pyqtgraph",
      ]

io_deps = [
    "scanimage-tiff-reader>=1.4.1",
    "paramiko",
    "nd2",
    "sbxreader",
    "h5py"
]

nwb_deps = [
        "pynwb>=2.3.2",
      ]

test_deps = [
      "pytest",
      "tenacity",
      "tqdm",
      "pynwb>=2.3.2", #this is needed as test_io contains a test with nwb
      "pytest-qt>3.3.0",
]

all_deps = gui_deps + nwb_deps + test_deps + io_deps

try:
    import torch
    a = torch.ones(2, 3)
    major_version, minor_version, _ = torch.__version__.split(".")
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
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
      "pytest-runner",
      "setuptools_scm",
    ],
    use_scm_version=True,
    install_requires=install_deps,
    tests_require=test_deps,
    extras_require={
      "docs": [
        "sphinx>=3.0",
        "sphinxcontrib-apidoc",
        "sphinx_rtd_theme",
        "sphinx-prompt",
        "sphinx-autodoc-typehints",
      ],
      "gui": gui_deps,
      "nwb": nwb_deps,
      "io": io_deps,
      "tests": test_deps,
      "all": all_deps,
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
      entry_points = {
        "console_scripts": [
          "suite2p = suite2p.__main__:main",
          "reg_metrics = benchmarks.registration_metrics:main",
          "tiff2scanimage = scripts.make_tiff_scanimage_compatible:main",
        ]
        },
)
