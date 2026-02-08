"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
import logging
logger = logging.getLogger(__name__)

class Classifier:
    """
    ROI classifier model that uses a weighted, non-parametric, naive Bayes classifier.

    Parameters
    ----------
    classfile : str, optional (default None)
        Path to saved classifier.
    keys : list of str, optional (default None)
        Keys of ROI stat to use to classify.
    """

    def __init__(self, classfile=None, keys=None):
        # stat are cell stats from currently loaded recording
        # classfile is a previously saved classifier file
        if classfile is not None:
            logger.info(f"classifier file: {classfile}")
            self.load(classfile, keys=keys)
        else:
            self.loaded = False

    def load(self, classfile, keys=None):
        """
        Load a saved classifier containing stats with classification labels.

        Parameters
        ----------
        classfile : str
            Path to saved classifier.
        keys : list of str, optional (default None)
            Keys of ROI stat to use to classify.
        """
        try:
            model = np.load(classfile, allow_pickle=True).item()
            if keys is None:
                self.keys = model["keys"]
                self.stats = model["stats"]
            else:
                model["keys"] = np.array(model["keys"])
                ikey = np.isin(model["keys"], keys)
                self.keys = model["keys"][ikey].tolist()
                self.stats = model["stats"][:, ikey]
            self.iscell = model["iscell"]
            self.loaded = True
            self.classfile = classfile
            self._fit()
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: incorrect classifier file")
            self.loaded = False

    def run(self, stat, p_threshold = 0.5):
        """
        Return cell classification thresholded with p_threshold and its probability.

        Parameters
        ----------
        stat : list of dict
            List of ROI statistics dictionaries, each containing the keys in self.keys.
        p_threshold : float, optional (default 0.5)
            Probability threshold for classifying an ROI as a cell.

        Returns
        -------
        iscell : numpy.ndarray
            Array of shape (n_rois, 2) where column 0 is the binary classification
            and column 1 is the probability.
        """
        probcell = self.predict_proba(stat)
        is_cell = probcell > p_threshold
        return np.stack([is_cell, probcell]).T

    def predict_proba(self, stat):
        """
        Apply classifier and predict probabilities.

        Parameters
        ----------
        stat : list of dict
            List of ROI statistics dictionaries, each containing the keys in self.keys.

        Returns
        -------
        y_pred : numpy.ndarray
            Predicted probability of each ROI being a cell, shape (n_rois,).
        """
        test_stats = np.array([stat[j][k] for j in range(len(stat)) for k in self.keys
                              ]).reshape(len(stat), -1)
        logp = self._get_logp(test_stats)
        y_pred = self.model.predict_proba(logp)[:, 1]
        return y_pred

    def save(self, filename):
        """
        Save classifier to an .npy file.

        Parameters
        ----------
        filename : str
            Path to save the classifier file.
        """
        np.save(filename, {
            "stats": self.stats,
            "iscell": self.iscell,
            "keys": self.keys
        })

    def _get_logp(self, stats):
        """
        Compute log probability of a set of stats.

        Parameters
        ----------
        stats : numpy.ndarray
            Statistics array of shape (n_cells, n_keys).

        Returns
        -------
        logp : numpy.ndarray
            Log probability array of shape (n_cells, n_keys).
        """
        logp = np.zeros(stats.shape)
        for n in range(stats.shape[1]):
            x = stats[:, n]
            x[x < self.grid[0, n]] = self.grid[0, n]
            x[x > self.grid[-1, n]] = self.grid[-1, n]
            x[np.isnan(x)] = self.grid[0, n]
            ibin = np.digitize(x, self.grid[:, n], right=True) - 1
            logp[:, n] = np.log(self.p[ibin, n] + 1e-6) - np.log(1 - self.p[ibin, n] +
                                                                 1e-6)
        return logp

    def _fit(self):
        """
        Fit weighted, non-parametric naive Bayes classifier using self.stats, self.keys, and self.iscell.

        Bins the stats into a non-parametric probability grid, smooths with a
        Gaussian, and fits a logistic regression on the log probabilities on the grid.
        """
        nodes = 100
        ncells, nstats = self.stats.shape
        ssort = np.sort(self.stats, axis=0)
        isort = np.argsort(self.stats, axis=0)
        ix = np.linspace(0, ncells - 1, nodes).astype("int32")
        grid = ssort[ix, :]
        p = np.zeros((nodes - 1, nstats))
        for j in range(nodes - 1):
            for k in range(nstats):
                p[j, k] = np.mean(self.iscell[isort[ix[j]:ix[j + 1], k]])
        p = gaussian_filter(p, (2., 0))
        self.grid = grid
        self.p = p
        logp = self._get_logp(self.stats)
        self.model = LogisticRegression(C=100., solver="liblinear")
        self.model.fit(logp, self.iscell)
