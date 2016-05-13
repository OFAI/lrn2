'''
Created on Feb 18, 2015

@author: Stefan Lattner
'''
import os
import bz2
import theano
import logging
import cPickle
import numpy as np
import theano.tensor as T
from theano.scan_module.scan_utils import equal_computations
from theano.tensor.shared_randomstreams import RandomStreams

from cPickle import UnpicklingError
from lrn2.util.gpu_batch import GPU_Batch
from lrn2.util.utils import ensure_ndarray

fx = T.config.floatX

LOGGER = logging.getLogger(__name__)

def load_pyc_bz(fn):
    return cPickle.load(bz2.BZ2File(fn, 'r'))

def save_pyc_bz(d, fn):
    cPickle.dump(d, bz2.BZ2File(fn, 'w'), cPickle.HIGHEST_PROTOCOL)
    
def create_dir(*args):
    out_dir = os.path.join(*args)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir

def batch_by_batch(function, data, batch_size, out_size):
    """
    If batch size is fixed (e.g. for some convolutional methods), to feed
    a severe amount of data, it has to be fed batch by batch).
    """
    result = np.zeros([len(data)] + out_size, dtype=fx)
    LOGGER.debug("Feeding data batch by batch ...")
    for idx in range(0,len(data)-batch_size,batch_size):
        result[idx:idx+batch_size] = function(data[idx:idx+batch_size])

    return result

def find_equi_branch(graph, branch):
    """
    Recursively searches for a branch in the graph which is equivalent
    to the given branch. Helps to clone with replace if the pointer
    to the node to be replaced is not available (this function is less
    picky than 'replace' in theano.clone).
    """

    if equal_computations([graph], [branch]):
        return graph

    if graph.owner is None:
        return None

    for curr_input in graph.owner.inputs:
        result = find_equi_branch(curr_input, branch)
        if result != None:
            return result

def create_rbm_graph(rbms, x, noise_level=0.):
    """
    Creates a theano graph out of a list of (stacked) RBMs

    Parameters
    ----------

    rbms : array-like
        a list of lrn2.models.RBM which should get stacked in a theano graph

    x : theano tensor
        input to the graph

    noise_level : float, optional
        noise level added to the input (e.g. for training auto-encoders with
        backpropagation)

    Returns
    -------

    a theano graph representation of the list of rbms

    """
    outputs = {}

    # first layer
    outputs[rbms[0]] = rbms[0].activation_h(get_corrupted_input(x, noise_level))

    # remaining layers
    for x in range(len(rbms)-1):
        outputs[rbms[x+1]] = rbms[x+1].activation_h(outputs[rbms[x]])

    return outputs[rbms[-1]]

def get_corrupted_input(x, corruption_level):
    """ This function keeps ``1-corruption_level`` entries of the inputs the same
    and zero-out randomly selected subset of size ``corruption_level``
    """
    theano_rng = RandomStreams(int(np.random.uniform(0,100000)))
    return T.cast(theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level)
                   * x, fx)

    def ngram_padding(self, ngram_size):
        '''
        Creates a matrix with [-1] padding for an ngram
        e.g. 3-gram_
        [[-1, -1, 0],
         [-1,  0, 1],
         [ 0,  1, 2],
        '''
        assert ngram_size >= 1
        l = range(ngram_size)

        lpadded = (ngram_size - 1) * [-1] + l
        out = [lpadded[i:(i + ngram_size)] for i in range(len(l))]

        assert len(out) == len(l)
        return out

def project_to_feature_space(rbm, train_set):

    if not hasattr(rbm, '__iter__'):
        rbm = (rbm,)

    curr_train_set = train_set

    for curr_rbm in rbm:
        X = T.matrix()
        func = theano.function([X], curr_rbm.activation_h(X))

        if isinstance(curr_train_set, GPU_Batch):
            gpu_batch = curr_train_set
        else:
            gpu_batch = GPU_Batch(curr_train_set, 1)


        h_act_app = None
        for curr_batch in gpu_batch:
            curr_h_act = func(ensure_ndarray(curr_batch))
            if h_act_app == None:
                h_act_app = curr_h_act
            else:
                h_act_app = np.vstack((h_act_app, curr_h_act))

        curr_train_set = h_act_app

    return curr_train_set

def get_from_cache_or_compute(cache_fn, func, args=(), kwargs={}, refresh_cache=False):
    """
    If `cache_fn` exists, return the unpickled contents of that file
    (the cache file is treated as a bzipped pickle file). If this
    fails, compute `func`(*`args`), pickle the result to `cache_fn`,
    and return the result.

    Parameters
    ----------

    func : function
        function to compute

    args : tuple
        argument for which to evaluate `func`

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    refresh_cache : boolean
        if True, ignore the cache file, compute function, and store the result in the cache file

    Returns
    -------

    object

        the result of `func`(*`args`)

    """
    result = None
    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                result = load_pyc_bz(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(('The file {0} exists, but cannot be unpickled. Is it readable? Is this a pickle file?'
                              '').format(cache_fn))
                raise e

    if result is None:
        result = func(*args, **kwargs)
        if cache_fn is not None:
            save_pyc_bz(result, cache_fn)
    return result

def compute_w_backup(obj, cache_fn, func, args = (), refresh_cache = False, 
                     load_only = False, load_existing = False,
                     continue_existing = False, **kwargs):
    """
    1.) Calls obj.load(cache_fn) if cache_fn exists and returns obj.
    2.) Executes a function with parameters args.
    3.) Calls obj.save(cache_fn) after execution.

    Parameters
    ----------

    obj : object
        an object providing a save() and load() function

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    func : function
        function to compute

    refresh_cache : boolean, optional
        if True, ignore the cache file, compute function, and store the result in the cache file

    load_only : boolean, optional
        if True, just call obj.load() and return the result (does not call func)
        
    args : tuple
        arguments for which to evaluate `func`

    kwargs : dict
        keyword arguments for which to evaluate `func`

    Returns
    -------

    object
        obj after executing func

    """
    assert not (load_existing and continue_existing), "Both cannot be true."

    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                obj.load(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(('The file {0} exists, but cannot be unpickled. Is it readable? Is this a pickle file?'
                              '').format(cache_fn))
                raise e

    if not load_only:
        overwrite = True
        if os.path.isfile(cache_fn):
            if load_existing:
                overwrite = False
            elif continue_existing:
                overwrite = True
            else:
                overwrite = not raw_input("Do you want to overwrite file {0} (choosing 'no' only loads the file) (Y/n)? ".format(cache_fn)).lower() == 'n'

        if overwrite:
            obj = func(*args, **kwargs)
            if cache_fn is not None:
                obj.save(cache_fn)

    return obj
