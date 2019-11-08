import numpy as np
from scipy.stats import zscore


class C_Cache():
    '''
    Class for holding the C matrix X@X.T and updating it when cells are flipped
    to non-cells or non-cells are flipped to cells.

    '''
    def __init__(self):
        self.first = True
        self.C = None
        self.prev = None

    def first_computation(self,X):
        self.C = X@X.T

    def delete(self,del_cells):
        for cell in del_cells:
            index = list(self.prev).index(cell)
            self.C = np.delete(self.C,index,axis=0)
            self.C = np.delete(self.C,index,axis=1)
            self.prev = self.prev[self.prev!=cell]

    def update(self,X,new_inds):
        new_columns = zscore(X[list(self.prev)+list(new_inds),:],axis=1)@zscore(X[new_inds,:].T,axis=0)
        self.C = np.append(self.C,new_columns[:len(list(self.prev)),:],axis=1)
        self.C = np.append(self.C,new_columns.T,axis=0)
        sorted_inds = list(np.argsort(list(self.prev)+list(new_inds)))
        self.C = self.C[:,sorted_inds]
        self.C = self.C[sorted_inds,:]
        #np.testing.assert_array_equal(self.C,self.C.T)
        print('debug',np.nonzero(np.abs(self.C-self.C.T)>1e-3))
