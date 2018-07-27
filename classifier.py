import sys
import numpy as np
import os

def train(traindata):
    '''input: matrix ncells x cols, where first column are labels, and the other
                columns are the statistics to use for classification
        output: distribution of cell likelihood for each statistic
    '''
    iscell = traindata[:,0]
    stats = traindata[:,1:]
    hists = 0
    return hists

def apply(hists, statistics, classval):
    '''inputs: hists (from train), statistics of cells to classify, and
                classval (probability of cell cutoff)
        output: iscell labels
    '''
    iscell = 0
    return iscell
