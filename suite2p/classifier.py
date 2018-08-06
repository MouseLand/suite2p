from PyQt5 import QtGui, QtCore
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import os
from suite2p import fig, gui
import time

class Classifier:
    def __init__(self, classfile=None, trainfiles=None, statclass=None):
        # load previously trained classifier
        if classfile is not None:
            self.classfile = classfile
            self.load()

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

    def load(self):
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


    def save(self, fname):
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
    hist = gaussian_filter(hist0, sig)
    hist = hist / hist.sum()
    return hist0

def load(parent, name):
    parent.model = Classifier(classfile=name,
                               trainfiles=None,
                               statclass=None)
    if parent.model.loaded:
        # statistics from current dataset for Classifier
        parent.statclass = parent.model.statclass
        # fill up with current dataset stats
        get_stats(parent)
        parent.trainfiles = parent.model.trainfiles
        activate(parent)

def get_stats(parent):
    ncells = parent.Fcell.shape[0]
    parent.statistics = np.zeros((ncells, len(parent.statclass)),np.float32)
    k=0
    for key in parent.statclass:
        for n in range(0,ncells):
            parent.statistics[n,k] = parent.stat[n][key]
        k+=1

def load_data(parent):
    # will return
    LC = gui.ListChooser('classifier training files', parent)
    result = LC.exec_()
    if result:
        print('Populating classifier:')
        parent.model = Classifier(classfile=None,
                                           trainfiles=parent.trainfiles,
                                           statclass=parent.statclass)
        if parent.trainfiles is not None:
            get_stats(parent)
            activate(parent)

def apply(parent):
    classval = parent.probedit.value()
    iscell, parent.probcell = parent.model.apply(parent.statistics, classval)
    fig.flip_for_class(parent, iscell)
    M = fig.draw_masks(parent)
    fig.plot_masks(parent,M)
    parent.lcell0.setText('%d ROIs'%parent.iscell.sum())
    parent.lcell1.setText('%d ROIs'%(parent.iscell.size-parent.iscell.sum()))

def save(parent):
    name = QtGui.QFileDialog.getSaveFileName(parent,'Save classifier')
    if name:
        try:
            parent.model.save(name[0])
        except (OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
            print('ERROR: incorrect filename for saving')

def save_list(parent):
    name = QtGui.QFileDialog.getSaveFileName(parent,'Save list of iscell.npy')
    if name:
        try:
            with open(name[0],'w') as fid:
                for f in parent.trainfiles:
                    fid.write(f)
                    fid.write('\n')
        except (ValueError, OSError, RuntimeError, TypeError, NameError,FileNotFoundError):
            print('ERROR: incorrect filename for saving')

def activate(parent):
    iscell, parent.probcell = parent.model.apply(parent.statistics, 0.5)
    istat = parent.probcell
    if len(parent.clabels) < parent.ncolors:
        parent.clabels.append([istat.min(), (istat.max()-istat.min())/2, istat.max()])
    else:
        parent.clabels[-1] = [istat.min(), (istat.max()-istat.min())/2, istat.max()]
    istat = istat - istat.min()
    istat = istat / istat.max()
    istat = istat / 1.3
    istat = istat + 0.1
    icols = 1 - istat
    if parent.ops_plot[3].shape[1] < parent.ncolors:
        parent.ops_plot[3] = np.concatenate((parent.ops_plot[3],
                                            np.expand_dims(icols,axis=1)), axis=1)
    else:
        parent.ops_plot[3][:,-1] = icols
    fig.init_masks(parent)
    parent.classbtn.setEnabled(True)
    parent.saveClass.setEnabled(True)
    parent.saveTrain.setEnabled(True)
    for btns in parent.classbtns.buttons():
        btns.setEnabled(True)

def disable(parent):
    parent.classbtn.setEnabled(False)
    parent.saveClass.setEnabled(False)
    parent.saveTrain.setEnabled(False)
    for btns in parent.classbtns.buttons():
        btns.setEnabled(False)


def add_to(parent):
    fname = parent.basename+'/iscell.npy'
    ftrue =  [f for f in parent.trainfiles if fname in f]
    if len(ftrue)==0:
        parent.trainfiles.append(parent.basename+'/iscell.npy')
    print('Repopulating classifier including current dataset:')
    parent.model = Classifier(classfile=None,
                                       trainfiles=parent.trainfiles,
                                       statclass=parent.statclass)
