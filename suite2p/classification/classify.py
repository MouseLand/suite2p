import numpy as np
from pathlib import Path
from typing import Union, Sequence
from .classifier import Classifier


def classify(stat: np.ndarray, use_builtin_classifier: bool = False,
             classfile: Union[str, Path] = None, keys: Sequence[str] = ('npix_norm', 'compact', 'skew'),
             ):
    """Returns array of classifier output from classification process."""
    builtin_classfile = Path(__file__).joinpath('../../classifiers/classifier.npy').resolve()
    user_classfile = Path.home().joinpath('.suite2p', 'classifiers', 'classifier_user.npy')
    if use_builtin_classifier:
        print(f'NOTE: Applying builtin classifier at {str(builtin_classfile)}')
        classfile = builtin_classfile
    elif not user_classfile.is_file():
        print(f'NOTE: no user default classifier exists.  applying builtin classifier at {str(builtin_classfile)}')
        classfile = builtin_classfile
    elif classfile is None or not Path(classfile).is_file():
        print(f'NOTE: applying default {str(user_classfile)}')
        classfile = user_classfile
    else:
        print(f'NOTE: applying classifier {str(classfile)}')
        classfile = classfile
    keys = list(set(keys).intersection(set(stat[0])))
    return Classifier(classfile, keys=keys).run(stat)
