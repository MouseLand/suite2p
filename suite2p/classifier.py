import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import os

class Classifier:
    def __init__(self, classfile=None, trainfiles=None, statclass=None):
        # load previously trained classifier
        if classfile is not None:
            self.classfile = classfile
            self.load_classifier()

        elif trainfiles is not None and statclass is not None:
            self.trainfiles = trainfiles
            self.statclass = statclass
            self.load_data()
            if self.traindata.shape[0]==0:
                self.loaded = False
            else:
                self.loaded = True
                self.train()

    def train(self):
        '''input: matrix ncells x cols, where first column are labels, and the other
                    columns are the statistics to use for classification
            output: distribution of cell likelihood for each statistic
        '''
        iscell = self.traindata[:,0].astype(bool)
        notcell = ~iscell
        stats = self.traindata[:,1:]
        # make grid of statistics values
        ncells,nstats = stats.shape
        grid = np.zeros((100, stats.shape[1]), np.float32)

        for n in range(nstats):
            grid[:,n] = np.linspace(np.percentile(stats[:,n], 2),
                                    np.percentile(stats[:,n], 98),
                                    100)
        hists = np.zeros((99,nstats,2), np.float32)
        for k in range(2):
            if k==1:
                ix = iscell
            else:
                ix = notcell
            for n in range(nstats):
                hists[:,n,k] = smooth_distribution(stats[ix,n], grid[:,n])

        self.hists = hists
        self.grid = grid

    def apply(self, stats, classval):
        '''inputs: model (from train), statistics of cells to classify, and
                    classval (probability of cell cutoff)
            output: iscell labels
        '''
        ncells, nstats = stats.shape
        grid = self.grid
        hists = self.hists
        logp = np.zeros((ncells,2), np.float32)
        for n in range(nstats):
            x = stats[:,n]
            x[x<grid[0,n]]   = grid[0,n]
            x[x>grid[-1,n]]  = grid[-1,n]
            ibin = np.digitize(x, grid[:,n], right=True) - 1
            logp = logp + np.log(np.squeeze(hists[ibin,n,:])+1e-5)
        p = np.ones((1,2),np.float32)
        p = p / p.sum()
        for n in range(10):
            L = logp + np.log(p)
            L = L - np.expand_dims(L.max(axis=1), axis=1)
            rs = np.exp(L) + 1e-5
            rs = rs / np.expand_dims(rs.sum(axis=1), axis=1)
            p = rs.mean(axis=0)

        probcell = rs[:,0]
        iscell = probcell > classval
        return iscell, probcell


    def load_classifier(self):
        try:
            model = np.load(self.classfile)
            model = model.item()
            self.grid = model['grid']
            self.hists = model['hists']
            self.trainfiles = model['trainfiles']
            self.statclass = model['statclass']
            self.loaded = True
        except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
            print('ERROR: incorrect classifier file')
            self.loaded = False


    def save_classifier(self, fname):
        model = {}
        model['grid'] = self.grid
        model['hists'] = self.hists
        model['trainfiles'] = self.trainfiles
        model['statclass'] = self.statclass
        print('saving classifier in ' + fname)
        np.save(fname, model)

    def load_data(self):
        statclass = self.statclass
        trainfiles = self.trainfiles
        traindata = np.zeros((0,len(statclass)+1),np.float32)
        trainfiles_good = []
        if trainfiles is not None:
            for fname in trainfiles:
                badfile = False
                basename, bname = os.path.split(fname)
                try:
                    iscell = np.load(fname)
                    ncells = iscell.shape[0]
                except (ValueError, OSError, RuntimeError, TypeError, NameError):
                    print('\t'+fname+': not a numpy array of booleans')
                    badfile = True
                if not badfile:
                    basename, bname = os.path.split(fname)
                    lstat = 0
                    try:
                        stat = np.load(basename+'/stat.npy')
                        stat = stat.item()
                        ypix = stat[0]['ypix']
                        lstat = len(stat)
                    except (KeyError, OSError, RuntimeError, TypeError, NameError):
                        print('\t'+basename+': incorrect or missing stat.npy file :(')
                    if lstat != ncells:
                        print('\t'+basename+': stat.npy is not the same length as iscell.npy')
                    else:
                        # add iscell and stat to classifier
                        print('\t'+fname+' was added to classifier')
                        iscell = iscell.astype(np.float32)
                        nall = np.zeros((ncells, len(statclass)+1),np.float32)
                        nall[:,0] = iscell
                        k=0
                        for key in statclass:
                            k+=1
                            for n in range(0,ncells):
                                nall[n,k] = stat[n][key]
                        traindata = np.concatenate((traindata,nall),axis=0)
                        trainfiles_good.append(fname)
        self.traindata = traindata
        self.trainfiles = trainfiles

def smooth_distribution(x, grid):
    xbin = x
    sig = 10.0
    xbin[xbin<grid[0]] = grid[0]#*np.ones(((xbin<grid[0]).sum(),))
    xbin[xbin>grid[-1]] = grid[-1]#*np.ones(((xbin>grid[-1]).sum(),))
    hist0 = np.histogram(xbin, grid)
    hist0 = hist0[0]
    hist = hist0#%gaussian_filter(hist0, sig)
    print(hist)
    hist = hist / hist.sum()
    return hist0
