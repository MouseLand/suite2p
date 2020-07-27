import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model  import LogisticRegression


class Classifier:
    """ ROI classifier model that uses logistic regression
    
    Parameters
    ----------

    classfile: string (optional, default None)
        path to saved classifier

    keys: list of str (optional, default None)
        keys of ROI stat to use to classify

    """
    def __init__(self, classfile=None, keys=None):
        # stat are cell stats from currently loaded recording
        # classfile is a previously saved classifier file
        if classfile is not None:
            self.load(classfile, keys=keys)
        else:
            self.loaded = False

    def load(self, classfile, keys=None):
        """ data loader

        saved classifier contains stat with classification labels 

        Parameters
        ----------
        
        classfile: string 
            path to saved classifier

        keys: list of str (optional, default None)
            keys of ROI stat to use to classify
         
        """
        try:
            model = np.load(classfile, allow_pickle=True).item()
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
            self.classfile = classfile
            self._fit()
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print('ERROR: incorrect classifier file')
            self.loaded = False

    def run(self, stat, p_threshold: float = 0.5) -> np.ndarray:
        """Returns cell classification thresholded with 'p_threshold' and its probability."""
        probcell = self.predict_proba(stat)
        is_cell = probcell > p_threshold
        return np.stack([is_cell, probcell]).T

    def predict_proba(self, stat):
        """ apply logistic regression model and predict probabilities

        model contains stat with classification labels 

        Parameters
        ----------
        
        stat : list of dicts
            needs self.keys keys

        """
        test_stats = np.array([stat[j][k] for j in range(len(stat)) for k in self.keys]).reshape(len(stat), -1)
        logp = self._get_logp(test_stats)
        y_pred = self.model.predict_proba(logp)[:, 1]
        return y_pred

    def save(self, filename: str) -> None:
        """ save classifier to filename """
        np.save(filename, {'stats': self.stats, 'iscell': self.iscell, 'keys': self.keys})

    def _get_logp(self, stats):
        """ compute log probability of set of stats
        
        Parameters
        --------------

        stats : 2D array
            size [ncells, nkeys]
        
        """
        logp = np.zeros(stats.shape)
        for n in range(stats.shape[1]):
            x = stats[:,n]
            x[x<self.grid[0,n]]   = self.grid[0,n]
            x[x>self.grid[-1,n]]  = self.grid[-1,n]
            ibin = np.digitize(x, self.grid[:,n], right=True) - 1
            logp[:,n] = np.log(self.p[ibin,n] + 1e-6) - np.log(1-self.p[ibin,n] + 1e-6)
        return logp

    def _fit(self):
        """ fit logistic regression model using stats, keys and iscell """
        nodes = 100
        ncells, nstats = self.stats.shape
        ssort= np.sort(self.stats, axis=0)
        isort= np.argsort(self.stats, axis=0)
        ix = np.linspace(0, ncells-1, nodes).astype('int32')
        grid = ssort[ix, :]
        p = np.zeros((nodes-1,nstats))
        for j in range(nodes-1):
            for k in range(nstats):
                p[j, k] = np.mean(self.iscell[isort[ix[j]:ix[j+1], k]])
        p = gaussian_filter(p, (2., 0))
        self.grid = grid 
        self.p = p
        logp = self._get_logp(self.stats)
        self.model = LogisticRegression(C = 100., solver='liblinear')
        self.model.fit(logp, self.iscell)

    


