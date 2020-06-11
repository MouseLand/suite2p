"""
Tests for the Suite2p Extraction module.
"""
import numpy as np
from suite2p import extraction
import utils


def prepare_for_extraction():
    pass


# def test_extraction_output_1plane1chan(default_ops):
#     ops = prepare_for_detection(
#         default_ops,
#         [[default_ops['data_path'][0].joinpath('detection', 'pre_registered.npy')]],
#         (404, 360)
#     )
#     detect_and_extract_wrapper(ops)
#     utils.check_output(
#         default_ops['save_path0'],
#         ['F', 'Fneu', 'iscell', 'stat', 'spks'],
#         default_ops['data_path'][0],
#         default_ops['nplanes'],
#         default_ops['nchannels'],
#     )