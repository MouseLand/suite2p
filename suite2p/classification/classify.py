import numpy as np
from pathlib import Path
from .classifier import Classifier


def classify(save_path, stat, classfile=None, keys=None):
    """
    Applies classifier and saves output to iscell.npy. Also, saves stat.npy.

    Parameters
    ----------
    save_path : destination of output files

    stat : array of dicts

    classfile: string (optional)    
        path to classifier

    keys: features (optional)
        features

    Returns
    -------

    iscell : array of classifier output

    """
    if keys is None:
        keys = []
    keys = keys + ['npix_norm', 'compact', 'skew']
    # apply default classifier
    if len(stat) > 0:
        if classfile is None or not Path(classfile).is_file():
            print('NOTE: applying default $HOME/.suite2p/classifiers/classifier_user.npy')
            classfile = Path.home().joinpath('.suite2p', 'classifiers', 'classifier_user.npy')
            if not Path(classfile).is_file():
                print('(no user default classifier exists)')
                print('NOTE: applying built in classifier.npy')
                classfile = Path(__file__).parent.parent.joinpath('classifiers', 'classifier.npy')
        else:
            print('NOTE: applying classifier %s' % classfile)
        for k in keys:
            if k not in stat[0]:
                keys.remove(k)
        iscell = Classifier(classfile, keys=keys).run(stat)
    else:
        iscell = np.zeros((0,2))
    fpath = Path(save_path)
    np.save(fpath.joinpath(('iscell.npy')), iscell)
    return iscell