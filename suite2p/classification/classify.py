import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from .classifier import Classifier


def get_built_in_classifier_path() -> Path:
    print('NOTE: applying built-in classifier.npy')
    return Path(__file__).joinpath('../../classifiers/classifier.npy')


def classify(save_path: Union[str, Path], stat: np.ndarray, use_builtin_classifier: bool = False,
             classfile: Union[str, Path] = None, keys: Optional[List[str]] = None,
             ):
    """
    Applies classifier and saves output to iscell.npy.

    Parameters
    ----------
    save_path : string / Pathlike object
        destination of output files

    stat : array of dicts
        each dict contains statistics for an ROI

    classfile: string (optional)    
        path to classifier

    use_builtin_classifier: bool
        whether or not classify should use built-in classifier

    keys: List[str] (optional)
        features

    Returns
    -------

    iscell : array of classifier output

    """
    if keys is None:
        keys = ['npix_norm', 'compact', 'skew']
    # apply default classifier
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
        iscell = Classifier(classfile, keys=keys).run(stat)
    else:
        iscell = np.zeros((0,2))
    np.save(Path(save_path).joinpath('iscell.npy'), iscell)
    return iscell
