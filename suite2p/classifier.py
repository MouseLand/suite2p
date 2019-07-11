import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression

class Classifier:
    def __init__(self, classfile=None, keys=None):
        # stat are cell stats from currently loaded recording
        # classfile is a previously saved classifier file
        if classfile is not None:
            self.classfile = classfile
            self.load(keys=keys)
        else:
            self.loaded = False

    def load(self, keys=None):
        try:
            model = np.load(self.classfile, allow_pickle=True).item()
            if keys is None:
                self.keys = model['keys']
                self.stats = model['stats']
            else:
                model['keys'] = np.array(model['keys'])
                ikey = np.isin(model['keys'], keys)
                self.keys = model['keys'][ikey].tolist()
                self.stats = model['stats'][:,ikey]
            self.iscell = model['iscell']
            self.loaded = True
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print('ERROR: incorrect classifier file')
            self.loaded = False

    def apply(self, stat):
        y_pred     = probability(stat,self.stats,self.iscell,self.keys)
        return y_pred

def get_logp(test_stats, grid, p):
    nroi, nstats = test_stats.shape
    logp = np.zeros((nroi,nstats))
    for n in range(nstats):
        x = test_stats[:,n]
        x[x<grid[0,n]]   = grid[0,n]
        x[x>grid[-1,n]]  = grid[-1,n]
        ibin = np.digitize(x, grid[:,n], right=True) - 1
        logp[:,n] = np.log(p[ibin,n] + 1e-6) - np.log(1-p[ibin,n] + 1e-6)
    return logp

def probability(stat, train_stats, train_iscell, keys):
    nodes = 100
    nroi, nstats = train_stats.shape
    ssort= np.sort(train_stats, axis=0)
    isort= np.argsort(train_stats, axis=0)
    ix = np.linspace(0, nroi-1, nodes).astype('int32')
    grid = ssort[ix, :]
    p = np.zeros((nodes-1,nstats))
    for j in range(nodes-1):
        for k in range(nstats):
            p[j, k] = np.mean(train_iscell[isort[ix[j]:ix[j+1], k]])
    p = gaussian_filter(p, (2., 0))
    logp = get_logp(train_stats, grid, p)
    logisticRegr = LogisticRegression(C = 100.)
    logisticRegr.fit(logp, train_iscell)
    # now get logP from the test data
    test_stats = get_stat_keys(stat, keys)
    logp = get_logp(test_stats, grid, p)
    y_pred = logisticRegr.predict_proba(logp)
    y_pred = y_pred[:,1]
    return y_pred

def get_stat_keys(stat, keys):
    test_stats = np.zeros((len(stat), len(keys)))
    for j in range(len(stat)):
        for k in range(len(keys)):
            test_stats[j,k] = stat[j][keys[k]]
    return test_stats

def run(classfile, stat, keys=None):
    model = Classifier(classfile=classfile, keys=keys)

    flag = np.zeros(len(model.keys), 'bool')
    new_keys = []
    for j in range(len(model.keys)):
        key = model.keys[j]
        if key not in stat[0]:
            flag[j] = True
        else:
            new_keys.append(key)
    model.keys = new_keys
    model.stats = np.delete(model.stats, np.nonzero(flag)[0], axis=1)

    # compute cell probability
    probcell = model.apply(stat)
    iscell = probcell > 0.5
    iscell = np.concatenate((np.expand_dims(iscell,axis=1),np.expand_dims(probcell,axis=1)),axis=1)
    return iscell
