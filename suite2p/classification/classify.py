import os
import numpy as np
from pathlib import Path
from . import Classifier


def classify(ops, stat):
    """
    Applies classifier and saves output to iscell.npy. Also, saves stat.npy.

    Parameters
    ----------
    ops : dictionary
        'save_path'
        (optional 'preclassify')

    stat : array of dicts

    Returns
    -------

    ops : dictionary

    stat : array of dicts

    iscell : array of classifier output

    """
    # apply default classifier
    if len(stat) > 0:
        user_dir = Path.home().joinpath('.suite2p')
        classfile = user_dir.joinpath('classifiers', 'classifier_user.npy')
        if not Path(classfile).is_file():
            s2p_dir = Path(__file__).parent.parent
            classfile = os.fspath(s2p_dir.joinpath('classifiers', 'classifier.npy'))
        print('NOTE: applying classifier %s'%classfile)
        iscell = Classifier(classfile, keys=['npix_norm', 'compact', 'skew']).run(stat)
        # Code Below does not work. Setting ops['preclassify'] gives you typeError.
        # if 'preclassify' in ops and ops['preclassify'] > 0.0:
        #     ic = (iscell[:,0]>ops['preclassify']).flatten().astype(np.bool)
        #     stat = stat[ic]
        #     iscell = iscell[ic]
        #     print('After classification with threshold %0.2f, %d ROIs remain'%(ops['preclassify'], len(stat)))
        #     np.save(fpath.joinpath(('stat.npy')), stat)
        # else:
    else:
        iscell = np.zeros((0,2))
    fpath = Path(ops['save_path'])
    np.save(fpath.joinpath(('iscell.npy')), iscell)
    return ops, iscell, stat