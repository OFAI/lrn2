'''
Created on May 13, 2015

@author: Stefan Lattner
'''

import theano
import numpy as np
import theano.tensor as T

from lrn2.util.utils import ensure_gpu_batch, ensure_ndarray


def similarity_matrix(instances_a, instances_b):
    """
    Returns a similarity matrix based on two 1D vectors (array-likes)
    """
    return [[np.linalg.norm(instances_a[i] - instances_b[j])
             for i in range(instances_a.shape[0])]
            for j in range(instances_b.shape[0])]



def get_similarity_matrix(rbm, train_set):
    """
    Projects a (ordered, sequential) train set into the feature space and
    calculate a similarity matrix on the resulting hidden unit activations.
    """
    X = T.matrix()
    project = theano.function([X], rbm.activation_h(X))

    gpu_batch = ensure_gpu_batch(train_set)

    h_act = None
    for curr_batch in gpu_batch:
        if h_act == None:
            h_act = project(ensure_ndarray(curr_batch))
        else:
            h_act = np.hstack((h_act, project(ensure_ndarray(curr_batch))))

    sim_matrix = similarity_matrix(h_act, h_act)

    return np.array(sim_matrix) * -1

def get_diff(rbm, train_set):
    """
    Calculates the difference of consecutive hidden unit activations based
    on an (ordered) train set which represents sequential data.
    """
    X = T.matrix()
    project = theano.function([X], rbm.activation_h(X))

    gpu_batch = ensure_gpu_batch(train_set)

    h_act = None
    for curr_batch in gpu_batch:
        if h_act == None:
            h_act = project(ensure_ndarray(curr_batch))
        else:
            h_act = np.hstack((h_act, project(ensure_ndarray(curr_batch))))

    diff = [np.linalg.norm(h_act[i] - h_act[i+1]) for i in range(h_act.shape[0]-1)]

    return diff