'''
Created on Mar 11, 2015

@author: Stefan Lattner
'''

import os
import theano
import logging
import numpy as np
import theano.tensor as T

from lrn2.nn_bricks.sample import Sampler
from lrn2.nn_bricks.plot import dummy_tiler, make_tiles

LOGGER = logging.getLogger(__name__)

class Approximator(Sampler):
    """
    Approximates the conditional (log-)probability of a subset of visible units,
    given the remaining visible units via sampling (training with
    e.g. Persistent Contrastive Divergence recommended).
    This is an implementation of Lattner et. al. 2015.

    Parameters
    ----------

    gibbs_chains : integer, optional
        number of samples to approximate the actual probability

    gibbs_steps : integer, optional
        number of gibbs steps for each gibbs chain

    clamp_mask : array-like, optional
        binary mask (dims of input) defining the subset of units which should
        be clamped (marked with ones). The probability of non-clamped units 
        given the clamped units will be estimated.

    eval_mask : array-like, optional
        if the probability of only a subset of unclamped units should be
        estimated, mark those units with ones in an additional mask.
    """

    def __init__(self, gibbs_chains = 200, gibbs_steps = 200, clamp_mask = None,
                 eval_mask = None, **kwargs):

        super(Approximator, self).__init__(gibbs_chains, clamp_mask, **kwargs)

        self.eval_mask = eval_mask
        self.gibbs_steps = gibbs_steps

    def generate_samples(self, v_in = None, init=None):
        """ Generates samples which are then used by get_probability() and
        get_entropy() to approximate the respective values. Call this for each
        new visible unit configuration (when visible units masked with 0's were
        updated) """

        self.reset_samples()
        
        if v_in is not None:
            # Property of Sampler (super class)
            self.clamp_v_config = v_in

        # Gibbs sampling
        self.v_samples.set_value(self.sample(self.gibbs_steps))

        if self.eval_mask is not None:
            self.v_samples.set_value(self.clamp_v_config * (1 - self.eval_mask) +
                                     self.v_samples.get_value() * self.eval_mask)

    def plot_samples(self, out_dir = ".", tiler = dummy_tiler, postfix=""):
        if self.convolutional:
            make_tiles(tiler(np.reshape(self.clamp_mask, (1,-1))),
                       os.path.join(out_dir, "mask{0}.png".format(postfix)))
        else:
            make_tiles(tiler(self.clamp_mask),
                       os.path.join(out_dir, "mask{0}.png".format(postfix)))
        make_tiles(tiler(self.v_samples.get_value()),
                   os.path.join(out_dir, "samples{0}.png".format(postfix)))
        if self.eval_mask is not None:
            make_tiles(tiler(np.asarray((self.eval_mask,))),
                       os.path.join(out_dir, "eval_mask{0}.png".format(postfix)))


    def get_probability(self, v_in):
        """ Estimate the (log-)probability of a visible configuration, given the
        activations of the phantasy particles. Call generate_samples() before,
        if visible units changed since the last call of generate_samples(). """
        if not hasattr(self, 'prob_fun'):
            self.compile_prob_fun()
        return self.prob_fun(v_in)

    def check_overlap(self):
        if np.sum(self.clamp_mask * self.eval_mask) != 0:
            LOGGER.warning("Clamp mask and eval mask overlap! This may cause "
                           "problems in entropy estimation. Define "
                           "eval mask only over unclampled units.")
        
    def get_entropy(self):
        """ Estimate the entropy of a visible configuration, given the
        activations of the phantasy particles. Note that visible units clamped
        with 1's do not influence the result. Call generate_samples() before,
        if visible units changed since the last call of generate_samples(). """
        self.check_overlap()
        if not hasattr(self, 'entropy_fun'):
            self.compile_entropy_fun()
        return self.entropy_fun()

    def binomial_elemwise(self, y, t):
        # specify broadcasting dimensions (multiple inputs to multiple
        # density estimations)
        if self.convolutional:
            est_shuf = y.dimshuffle('x', 0, 1, 2, 3)
            v_shuf = t.dimshuffle(0, 'x', 1, 2, 3)
        else:
            est_shuf = y.dimshuffle('x', 0, 1)
            v_shuf = t.dimshuffle(0, 'x', 1)

        # Calculate probabilities of current v's to be sampled given estimations
        # Real-valued observation -> Binomial distribution
        # Binomial coefficient (factorial(x) == gamma(x+1))
        bin_coeff = 1 / (T.gamma(1 + v_shuf) * T.gamma(2 - v_shuf))
        pw_probs = bin_coeff * T.pow(est_shuf, v_shuf) * T.pow(1. - est_shuf,
                                                                 1. - v_shuf)
        return pw_probs
        
    def log_likelihood(self, y, t):
        pw_probs = self.binomial_elemwise(y, t)
        # Multiply and average probs
        if self.convolutional:
#             return T.prod(pw_probs, axis=(2,3,4))
            return T.mean(T.sum(T.log(pw_probs), axis=(2,3,4)), axis = 0)
        else:
            return T.mean(T.sum(T.log(pw_probs), axis=2), axis=0)
        
    def compile_prob_fun(self):
        v = T.tensor4("v_in") if self.convolutional else T.matrix("v_in")
        self.prob_fun = theano.function([v], 
                                        self.log_likelihood(self.v_samples, v),
                                        on_unused_input='warn')

    def compile_entropy_fun(self):
        p = self.v_samples

        h = -p*T.log2(p)-(1-p)*T.log2(1-p)
        h = T.switch(T.isnan(h), 0., h)
        
        if self.eval_mask is not None:
            eval_units = np.sum(self.eval_mask)
        else:
            eval_units = np.prod(self.clamp_mask.shape) - np.sum(self.clamp_mask)
            
        entropy = T.sum(h) / (self.n_samples * eval_units) 

        self.entropy_fun = theano.function([], entropy)
