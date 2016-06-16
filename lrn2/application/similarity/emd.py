'''
Created on Jun 15, 2016

@author: Stefan Lattner
'''
import numpy as np
from scipy.optimize._linprog import linprog
from scipy.spatial.distance import pdist, squareform
from itertools import product

def emd(x, y, xw=None, yw=None, metric='euclidean', distances=None):
    """
    Calculates the earth mover's distance between two sets of points.
    Points can be weighted and weights do not have to sum to 1 or to a common
    value (returns residual).
    
    Parameters
    ----------
    
    x : list of tuples
        a list of coordinates of a set of points
        
    y : list of tuples
        a list of coordinates of a set of points
        
    xw : array-like, optional
        a list of weights for each point in x
    
    yw : array-like, optional
        a list of weights for each point in y
        
    metric : string, optional, default = 'euclidean'
        A valid distance type accepted by the scipy.spatial.cdist function.
        Alternatively, if distance='precomputed', a precomputed distance matrix is
        expected to be supplied to the optional argument 'distances'.
        
    Returns
    -------
    
    triple:
        (distance,
        amount earth moved (order e.g. [(x_1,y_1),(x_1,y_2),(x_2,y_1),(x_2,y_2)]),
        amount earth leftover)
    
    """
    assert len(x) == len(xw), "number of points and number of weights have to match (x)"
    assert len(y) == len(yw), "number of points and number of weights have to match (y)"
    assert xw == None or yw != None, "assign weights for either both point sets or none"
    assert metric == 'precomputed' or distances == None, \
                "set metric='precomputed' when using a custom distance matrix"
    
    if metric is not 'precomputed':             
        distances = squareform(pdist(np.vstack((x,y)), metric='euclidean'))
    else:
        assert distances != None, "pass a distance matrix when metric='precomputed'"
        
    # e.g. [d(x_1,y_1),d(x_1,y_2),d(x_2,y_1),d(x_2,y_2)]
    distances = distances[:len(x),-len(y):].reshape((-1))
    

    if xw == None:
        xw = np.ones((len(x))) / len(x)
        yw = np.ones((len(y))) / len(y)

    weights = np.concatenate((xw, yw))
    
    prod = [i for i in product(range(len(x)),len(x) + np.arange(len(y)))]
    A_ub = np.cast['int8']([[c in tup for tup in prod] for c in range(len(x)+len(y))])
    
    min_mass = min(np.sum(xw),np.sum(yw))
    A_eq = np.ones((1,len(distances)))
    
    res = linprog(c = distances, A_ub = A_ub, b_ub = weights, 
                  A_eq = A_eq, b_eq = min_mass)
    
    cost = res.fun    
    return cost / min_mass, res.x, res.slack
    
if __name__ == '__main__':
    x = [[3,4],[2,3],[10,8]]
    y = [[3,4],[7,9]]
    wx = [2,1,1]
    wy = [1,2]
    print emd(x,y,wx,wy)
