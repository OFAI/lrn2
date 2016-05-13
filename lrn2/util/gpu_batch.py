# -*- coding: utf-8 -*-
import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

class GPU_Batch:
    """
    Splits a given dataset in several batches and provides
    simple access via an iterator.

    Usage example:
    gpu_batch = GPU_Batch(_train_set, div=2)
    
    for curr_batch in gpu_batch:
    ># training_data is a theano shared variable
    >rbm.training_data.set_value(curr_batch.astype(T.config.floatX))
    >rbm.train()
    """

    def __init__(self, train_set, n_batches = 1):
        """
        Parameters
        ----------
        train_set : array-like
            A 2-D set of instances where the rows correspond to instances,
            the colums to the values of instances.

            The iterator accesses the set in the following way:
            train_set[instance_from:instance_to,:]
            
        n_batches: int
            The desired number of gpu batches. If there is a residual
            after division, a small residual batch is created as the
            last batch of the batch set.
        """
        self._train_set = train_set
        self.div = n_batches
        self.count = 0

        # nr of instances
        n_instances = train_set.shape[0]

        # GPU sub-patches
        sub_batch_size = n_instances / n_batches

        # Divide data into smaller sets that are copied to GPU memory separately
        batch_indices = range(0, n_instances, sub_batch_size)

        # Very small batch to account for division residue
        if batch_indices[-1] < n_instances:
            batch_indices.extend([n_instances])

        self.batch_indices = batch_indices

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_indices)

    def seek(self, index):
        self.count = index

    def next(self):
        self.count += 1
        if self.count >= len(self.batch_indices):
            self.count = 0
            raise StopIteration
        else:
            return self._train_set[self.batch_indices[self.count-1]:
                             self.batch_indices[self.count],:]

def GPU_safe_process(function, data, converter = lambda x : x):
    """
    Tries to process data with a given function and increases the nr. of
    GPU batches until every batch fits into memory. Returns results of all
    single batches as a list.
    """
    success = False
    n_batches = 1
    result_all = []
    while not success:
        try:
            batches = GPU_Batch(data, n_batches)
            for batch in batches:
                curr_res = function(converter(batch))
                success = True
                result_all.append(curr_res)
        except (MemoryError, RuntimeError):
            n_batches += 1
            LOGGER.debug("Data didn't fit in memory, try with {0} batches...".format(n_batches))

    return np.vstack(result_all)

def GPU_safe_process_reduce(function, data, converter = lambda x : x):
    """
    Tries to process data with a given function and decreases the nr. of
    instances, until everything fits in memory.
    """
    success = False
    step = 1
    result_all = []
    while not success:
        try:
            data_reduced = data[::step]
            curr_res = function(converter(data_reduced))
            success = True
            result_all.append(curr_res)
        except MemoryError:
            step += 1
            LOGGER.debug("Data didn't fit in memory, try with {0} batches...".format(step))

    return result_all