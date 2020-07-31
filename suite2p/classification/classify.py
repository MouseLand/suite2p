import numpy as np
from pathlib import Path
from typing import Union, Sequence
from .classifier import Classifier


def classify(stat: np.ndarray,
             classfile: Union[str, Path] = None, keys: Sequence[str] = ('npix_norm', 'compact', 'skew'),
             ):
    """Returns array of classifier output from classification process."""

    for key in keys:
        if key not in stat[0]:
            raise KeyError("Key '{}' not found in stats dictionary.".format(key))

    return Classifier(classfile, keys=keys).run(stat)
