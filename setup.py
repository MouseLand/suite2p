import setuptools

# ------------------------------------------------------------------------------------
# NOTE: This is a custom fork of Suite2p, originally by Marius Pachitariu & Carsen Stringer.
# We preserve their license (GPL v3) and give credit for the original code. 
# This file has been adapted to clarify it is a modified version. 
# ------------------------------------------------------------------------------------

install_deps = [
    "importlib-metadata",
    "natsort",
    "rastermap>=0.9.0",
    "tifffile",
    "torch>=1.13.1",
    "numpy>=1.24.3",
    "numba>=0.57.0",
    "matplotlib",
    "scipy>=1.9.0",
    "scikit-learn",
    "cellpose",
    "scanimage-tiff-reader>=1.4.1"
]

gui_deps = [
    "qtpy",
    "pyqt6",
    "pyqt6.sip",
    "pyqtgraph",
]

io_deps = [
    "paramiko",
    "nd2",
    "sbxreader",
    "h5py",
    "opencv-python-headless",
    "xmltodict",
    "dcimg"
]

nwb_deps = [
    "pynwb>=2.3.2",
]

test_deps = [
    "pytest",
    "tenacity",
    "tqdm",
    "pynwb>=2.3.2",  # needed since test_io contains an NWB test
    "pytest-qt>3.3.0",
]

# Check if PyQt/PySide is installed to avoid conflicts
try:
    import PyQt5
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except ImportError:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except ImportError:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except ImportError:
    pass

all_deps = gui_deps + nwb_deps + test_deps + io_deps

# Torch version check
try:
    import torch
    major_version, minor_version, _ = torch.__version__.split(".")
    # Optionally remove older constraints if a newer Torch is present
    if major_version == "2" or int(minor_version) >= 6:
        install_deps.remove("torch>=1.6")
except ImportError:
    pass

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    # Name changed to avoid conflicts with the official suite2p package on PyPI.
    name="suite2p-custom",  
    # Include both original and your info. 
    author="Marius Pachitariu, Carsen Stringer (Original Authors); Fork Maintainer: Ahmed Jamali",
    author_email="Ahmedj@ntnu.no",  # Replace with your contact if desired
    description="Pipeline for calcium imaging (Custom Fork of Suite2p)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AhmedJamali/suite2pR",
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
    entry_points={
        "console_scripts": [
            "suite2p = suite2p.__main__:main",
            "reg_metrics = benchmarks.registration_metrics:main",
            "tiff2scanimage = scripts.make_tiff_scanimage_compatible:main",
        ]
    },
)
