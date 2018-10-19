import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="suite2p",
    version="0.4.4",
    author="Marius Pachitariu and Carsen Stringer",
    author_email="marius10p@gmail.com",
    description="Pipeline for calcium imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MouseLand/suite2p",
    packages=setuptools.find_packages(),
    install_requires = ['pyqtgraph', 'PyQt5', 'numpy>=1.13.0', 'scipy', 'h5py', 'scikit-image', 'scikit-learn', 'matplotlib','rastermap'],
    include_package_data=True,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ),
)
