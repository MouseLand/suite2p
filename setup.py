import setuptools

install_deps = ["importlib-metadata",
        "natsort",
        "tifffile",
        "torch>=1.13.1",
        "numpy>=1.24.3",
        "numba>=0.57.0",
        "matplotlib",
        "scipy>=1.9.0",
        "scikit-learn",
        "cellpose>=4.0.1",
        "scanimage-tiff-reader>=1.4.1"
        ]

gui_deps = [
        "qtpy",
        "superqt",
        "pyqt6",
        "pyqt6.sip",
        "pyqtgraph",
        "rastermap>=0.9.0",
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
      "pynwb>=2.3.2", #this is needed as test_io contains a test with nwb
      "pytest-qt>3.3.0",
]

# check if pyqt/pyside already installed
try:
    import PyQt5
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide2
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

try:
    import PySide6
    gui_deps.remove("pyqt6")
    gui_deps.remove("pyqt6.sip")
except:
    pass

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
    extras_require={
      "docs": [
        "sphinx>=3.0",
        "mkdocs",
        "mkdocs-material",
        "mkdocs-git-revision-date-localized-plugin",
        "mkdocstrings",
        "mkdocstrings-python"
      ],
      "gui": gui_deps,
      "nwb": nwb_deps,
      "io": io_deps,
      "test": test_deps,
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
        ]
        },
)
