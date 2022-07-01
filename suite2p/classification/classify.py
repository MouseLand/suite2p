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
    """ 
    Main classification function 
    
    Returns array of classifier output from classification process

    Parameters
    ----------------

    stat: dictionary 'ypix', 'xpix', 'lam'
        Dictionary containing statistics for ROIs

    classfile: string (optional, default None)
        path to saved classifier

    keys: list of str (optional, default None)
        keys of ROI stat to use to classify

    Returns
    ----------------

    iscell : np.ndarray
        Array in which each i-th element specifies whether i-th ROI is a cell or not.

    """
    keys = list(set(keys).intersection(set(stat[0])))
    print(keys)
    iscell = Classifier(classfile, keys=keys).run(stat)
    return iscell
