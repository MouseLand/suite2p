import numpy as np
from pathlib import Path
from typing import Union, Sequence
from .classifier import Classifier


def get_built_in_classifier_path() -> Path:
    print('NOTE: applying built-in classifier.npy')
    return Path(__file__).joinpath('../../classifiers/classifier.npy').resolve()


def classify(stat: np.ndarray, use_builtin_classifier: bool = False,
             classfile: Union[str, Path] = None, keys: Sequence[str] = ('npix_norm', 'compact', 'skew'),
             ):
    """Returns array of classifier output from classification process."""
    if len(stat) == 0:
        return np.zeros((0, 2))

    if len(stat) > 0:
        if use_builtin_classifier:
            classfile = get_built_in_classifier_path()
        elif classfile is None or not Path(classfile).is_file():
            print('NOTE: applying default $HOME/.suite2p/classifiers/classifier_user.npy')
            classfile = Path.home().joinpath('.suite2p', 'classifiers', 'classifier_user.npy')
            if not Path(classfile).is_file():
                print('(no user default classifier exists)')
                classfile = get_built_in_classifier_path()
        else:
            print('NOTE: applying classifier %s' % classfile)
        keys = list(set(keys).intersection(set(stat[0])))
        return Classifier(classfile, keys=keys).run(stat)
