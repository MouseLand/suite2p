"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from .version import version
from .parameters import default_ops, user_ops, OPS_FOLDER, default_db
from .pipeline_s2p import pipeline
from .run_s2p import run_s2p, run_plane
from .detection import detection_wrapper
from .classification import classify
from .extraction import extraction_wrapper
from .registration import registration_wrapper

name = "suite2p"
