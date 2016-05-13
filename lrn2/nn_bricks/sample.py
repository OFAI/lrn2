'''
Created on Mar 11, 2015

@author: Stefan Lattner
'''

import theano
import theano.tensor as T
import numpy as np
from lrn2.nn_bricks.utils import fx
import logging
from theano.compile.function_module import UnusedInputError

LOGGER = logging.getLogger(__name__)

class Sampler(object):
    """
    Enables an RBM to sample from it. Possibility to clamp some units.

    Parameters
    ----------

    n_samples : int, optional
        number of samples to be generated at a time

    clamp_mask : array-like, optional
        binary mask in the size of an input, telling which units should
        be clamped (1 for clamped units, 0 for unclamped)

    clamp_v_config : array-like, optional
        any input activation / configuration (input size), defining the
        state of clamped units.

    initial_samples : array-like, optional
        initial state of sampling chains
    """
    def __init__(self, n_samples = 1, clamp_mask = None, clamp_v_config = None,
                 initial_samples = None):

        if self.convolutional:
            LOGGER.warning("In convolutional nets the number of samples is "
                           "automatically set to the mini batch size. "
                           "The parameter n_samples = {0} ".format(n_samples) +
                           "will be ignored.")

        self.n_samples = n_samples

        self.init_samples(initial_samples)

        if clamp_mask == None:
            self.clamp_mask_ = theano.shared(np.zeros((self.input_shape), dtype=fx))
        else:
            clamp_mask = np.asarray(clamp_mask, dtype=fx)
            self.clamp_mask_ = theano.shared(clamp_mask)

        if clamp_v_config == None:
            self.clamp_v_config_ = theano.shared(np.zeros((self.input_shape), dtype=fx))
        else:
            clamp_v_config = np.asarray(clamp_v_config, dtype=fx)
            self.clamp_v_config_ = theano.shared(clamp_v_config)

        self.compile_sample_fun()

    def init_samples(self, init = None):
        """ Initialize Fantasy particles """
        if init is not None:
            self.v_samples = theano.shared(init, borrow=True)
        else:
            pps_shape = [self.n_samples] + list(self.input_shape)
            self.v_samples = theano.shared(np.random.uniform(0, 1, pps_shape)
                                              .astype(fx), borrow=True)

    def reset_samples(self, init = None):
        if init is not None:
            self.v_samples.set_value(init)
        else:
            pps_shape = [self.n_samples] + list(self.input_shape)
            self.v_samples.set_value(np.random.uniform(0, 1, pps_shape)
                                              .astype(fx), borrow=True)

    def compile_sample_fun(self):
        """ Compile Gibbs sampling of phantasy particles as function using scan """
        k = T.iscalar('k')

        result, updates = theano.scan(fn=lambda x :
                                      self.sample_step(x),
                                      outputs_info = self.v_samples,
                                      n_steps = k)
        final_result = result[-1]
        try:
            self.sample = theano.function([k, self.variables['input']], 
                                          outputs=final_result,
                                          updates=updates,
                                          allow_input_downcast=True)
        except UnusedInputError:
            self.sample = theano.function([k], outputs=final_result,
                                          updates=updates,
                                          allow_input_downcast=True)

    def sample_step(self, v_in):
        v_act = self.gibbs_step_(v_in)
        v_act_cl = (self.clamp_v_config_.dimshuffle('x', 0, 1, 2) * \
                    self.clamp_mask_.dimshuffle('x', 0, 1, 2)) + \
                    (1 - self.clamp_mask_.dimshuffle('x', 0, 1, 2)) * v_act
        return v_act_cl

    @property
    def clamp_mask(self):
        return self.clamp_mask_.get_value()

    @clamp_mask.setter
    def clamp_mask(self, value):
        self.clamp_mask_.set_value(value)

    @property
    def clamp_v_config(self):
        return self.clamp_v_config_.get_value()

    @clamp_v_config.setter
    def clamp_v_config(self, value):
        self.clamp_v_config_.set_value(value)



