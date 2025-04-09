# sl-suite2p

A modified suite2p library implementation refactored to work with Sun (NeuroAI) lab’s data processing pipeline.

![PyPI - Version](https://img.shields.io/pypi/v/sl-experiment)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sl-experiment)
![PyPI - License](https://img.shields.io/pypi/l/sl-experiment)
![PyPI - Status](https://img.shields.io/pypi/status/sl-experiment)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sl-experiment)
___

## Detailed Description

This service library contains a partially refactored and repackaged [suite2p](https://github.com/MouseLand/suite2p) 
source code. Overall, since source code refactoring was intentionally minimal, this implementation should behave very 
similar to the official suite2p implementation. 

Primarily, the refactoring efforts were aimed at making suite2p work for the specific data format and processing server 
configuration used in the Sun lab. As part of this process, we had to repackage the library to use modern Python tools, 
such as pyproject.toml, and introduced tighter dependency control and necessary code changes to make the library work 
with newer Python versions (up to 3.13).

## Disclaimer

This library is distributed under the GPL 3 license, inheriting it from the original library. All source code rights 
belong to the original authors. This release is viewed as temporary and will not be maintained after the original 
suite2p implements the necessary changes to make it compatible with newer Python versions and server configurations
different from Hami's. This implementation is intended to be used solely by other Sun lab pipelines and will likely not
work in other contexts.

For API documentation, see the original API documentation available 
[here](https://suite2p.readthedocs.io/en/latest/settings.html).