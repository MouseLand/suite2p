import numpy as np
from pathlib import Path
from typing import Union, List, Optional
from .classifier import Classifier


def classify(save_path: Union[str, Path], stat: np.ndarray, use_builtin_classifier: bool,
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
        if classfile is None:
            print('NOTE: applying user default $HOME/.suite2p/classifiers/classifier_user.npy')
            classfile = Path.home().joinpath('.suite2p', 'classifiers', 'classifier_user.npy')
        if (not Path(classfile).is_file()) or use_builtin_classifier:
            print('No user default was found or use_builtin_classifier set to True.')
            print('NOTE: applying built-in classifier.npy')
            classfile = Path(__file__).parent.parent.joinpath('classifiers', 'classifier.npy')
        else:
            print('NOTE: applying classifier %s' % classfile)
        for k in keys:
            if k not in stat[0]:
                keys.remove(k)
        iscell = Classifier(classfile, keys=keys).run(stat)
    else:
        iscell = np.zeros((0,2))
    np.save(Path(save_path).joinpath('iscell.npy'), iscell)
    return iscell
