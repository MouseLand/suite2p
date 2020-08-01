from importlib_metadata import metadata as _metadata

from .run_s2p import run_s2p, default_ops, builtin_classfile, user_classfile
from .gui import run as run_gui

name = "suite2p"
version = _metadata('suite2p')['version']

