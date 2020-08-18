import numpy as np
from pathlib import Path
from typing import Union, Sequence
from .classifier import Classifier

builtin_classfile = Path(__file__).joinpath('../../classifiers/classifier.npy').resolve()
user_classfile = Path.home().joinpath('.suite2p/classifiers/classifier_user.npy')


def classify(stat: np.ndarray,
             classfile: Union[str, Path],
             keys: Sequence[str] = ('npix_norm', 'compact', 'skew'),
             ):
    """Returns array of classifier output from classification process."""
    keys = list(set(keys).intersection(set(stat[0])))
    return Classifier(classfile, keys=keys).run(stat)
