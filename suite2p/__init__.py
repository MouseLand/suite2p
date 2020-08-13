from importlib_metadata import metadata as _metadata

from .run_s2p import run_s2p, default_ops
from .gui import run as run_gui
from .detection import ROI

name = "suite2p"
version = _metadata('suite2p')['version']

