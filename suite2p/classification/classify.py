"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from pathlib import Path
from .classifier import Classifier
import logging 
logger = logging.getLogger(__name__)


builtin_classfile = Path(__file__).joinpath(
    "../../classifiers/classifier.npy").resolve()
user_classfile = Path.home().joinpath(".suite2p/classifiers/classifier_user.npy")


def classify(stat, classfile, keys = ("skew", "npix_norm", "compact")):
    """
    Classify ROIs as cells or not cells using a saved classifier.

    Parameters
    ----------
    stat : numpy.ndarray
        Array of dictionaries, each containing ROI statistics, including the `keys`
    classfile : str or pathlib.Path
        Path to saved classifier.
    keys : sequence of str, optional (default ("skew", "npix_norm", "compact"))
        Keys of ROI stat to use to classify.

    Returns
    -------
    iscell : numpy.ndarray
        Array of shape (n_rois, 2) where column 0 is the binary classification
        and column 1 is the probability.
    """
    keys = list(set(keys).intersection(set(stat[0])))
    logger.info(f"classifying with stats: {keys}")
    iscell = Classifier(classfile, keys=keys).run(stat)
    return iscell
