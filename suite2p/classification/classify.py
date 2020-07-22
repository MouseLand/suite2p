import os
import numpy as np
from pathlib import Path
from .classifier import Classifier


def classify(ops, stat, classfile=None, keys=['npix_norm', 'compact', 'skew']):
    """
    Applies classifier and saves output to iscell.npy. Also, saves stat.npy.

    Parameters
    ----------
    ops : dictionary
        'save_path'
        (optional 'preclassify')

    stat : array of dicts

    classfile: string (optional)    
        path to classifier

    Returns
    -------

    ops : dictionary

    stat : array of dicts

    iscell : array of classifier output

    """
    # apply default classifier
    if len(stat) > 0:
        if classfile is None or not Path(classfile).is_file():
            print('NOTE: applying default $HOME/.suite2p/classifiers/classifier_user.npy')
            user_dir = Path.home().joinpath('.suite2p')
            classfile = user_dir.joinpath('classifiers', 'classifier_user.npy')
            if not Path(classfile).is_file():
                print('(no user default classifier exists)')
                print('NOTE: applying built in classifier.npy')
                s2p_dir = Path(__file__).parent.parent
                classfile = os.fspath(s2p_dir.joinpath('classifiers', 'classifier.npy'))
        else:
            print('NOTE: applying classifier %s'%classfile)
        for k in keys:
            if k not in stat[0]:
                keys.remove(k)
        iscell = Classifier(classfile, keys=keys).run(stat)
    else:
        iscell = np.zeros((0,2))
    fpath = Path(ops['save_path'])
    np.save(fpath.joinpath(('iscell.npy')), iscell)
    return iscell