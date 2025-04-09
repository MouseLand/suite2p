# sl-suite2p

A modified suite2p library implementation refactored to work with Sun (NeuroAI) lab’s data processing pipeline.

![PyPI - Version](https://img.shields.io/pypi/v/sl-experiment)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sl-experiment)
![PyPI - License](https://img.shields.io/pypi/l/sl-experiment)
![PyPI - Status](https://img.shields.io/pypi/status/sl-experiment)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/sl-experiment)
___

## Detailed Description

This service library contains minorly refactored and repackaged [suite2p](https://github.com/MouseLand/suite2p) source 
code and should behave very similar to the original suite2p library. Primarily, this library aims at refactoring the 
suite2p source code to make it work for the specific data format and processing server configuration used in the Sun 
lab. Moreover, it repackages the library to use modern Python tools, such as pyproject.toml, and introduced tighter 
dependency control and necessary code changes to make the library work with newer Python versions (up to 3.13).

## Notes

This library is distributed under the GPL 3 license, inheriting it from the original library. All source code rights 
belong to the original authors. 