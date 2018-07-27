import sys
import numpy as np
import os

class Classifier:
    def __init__(self, classfile=None, trainfiles=None, statclass=None):
        # load previously trained classifier
        if classfile is not None:
            self.classfile = classfile
            self.load_classifier()
            if not self.loaded:
                raise ValueError('ERROR: bad classifier')
        elif trainfiles is not None and statclass is not None:
            self.trainfiles = trainfiles
            self.statclass = statclass
            self.load_data()
            if self.traindata.shape[0]==0:
                raise ValueError('ERROR: no valid files added to classifier')
            else:
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
            grid[:,n] = np.linspace(np.percentile(stats[:,n], .02),
                                    np.percentile(stats[:,n], .98),
                                    100)
        for n in range(ncells):
            hists = 0
        self.hists = hists
        self.grid = grid

    def apply(self, stats, classval):
        '''inputs: model (from train), statistics of cells to classify, and
                    classval (probability of cell cutoff)
            output: iscell labels
        '''
        iscell = 0
        return iscell

    def load_classifier(self):
        try:
            model = np.load(self.classfile)
            model = model.item()
            self.grid = model['grid']
            self.hists = model['hists']
            self.trainfiles = model['trainfiles']
            self.statclass = model['statclass']
            self.loaded = True
        except (KeyError, OSError, RuntimeError, TypeError, NameError):
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
                except (OSError, RuntimeError, TypeError, NameError):
                    print('\t'+fname+': not a numpy array of booleans')
                    badfile = True
                if not badfile:
                    basename, bname = os.path.split(fname)
                    lstat = 0
                    try:
                        stat = np.load(basename+'/stat.npy')
                        stat = stat.item()
                        ypix = stat[0]['ypix']
                        lstat = len(stat) - 1
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
