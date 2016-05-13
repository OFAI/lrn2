'''
Created on Mar 11, 2015

@author: Stefan Lattner
'''

import os
import theano
import theano.tensor as T
import numpy as np

from lrn2.nn_bricks.sample import Sampler
from lrn2.nn_bricks.plot import dummy_tiler, make_tiles


class Approximator(Sampler):
    """
    Approximates the conditional probability of a subset of visible units,
    given the remaining visible units via sampling (training with
    e.g. Persistent Contrastive Divergence recommended).
    This is an implementation of Lattner et. al. 2015.

    Parameters
    ----------

    gibbs_chains : integer, optional
        number of samples to approximate the actual probability

    gibbs_steps : integer, optional
        number of gibbs steps for each gibbs chain

    mask : array-like, optional
        binary mask defining the subset of units which should be clamped (marked with zero)
        (dims of input). The probability of non-clamped units given the clamped
        units is estimated.

    eval_mask : array-like, optional
        if the probability of only a subset of unclamped units should be
        estimated, this additional mask can be defined.
    """

    def __init__(self, gibbs_chains = 200, gibbs_steps = 200, mask = None,
                 eval_mask = None):

        super(Approximator, self).__init__(gibbs_chains, mask)

        self.eval_mask = eval_mask
        self.gibbs_steps = gibbs_steps
        self.compile_entropy_fun()
        self.compile_prob_fun()


    def generate_samples(self, v_in, init=None):
        """ Generates samples which are then used by get_probability() and
        get_entropy() to approximate the respective values. Call this for each
        new visible unit configuration (when visible units masked with 0's were
        updated) """

        self.reset_samples()
        
        # Property of Sampler (super class)
        self.clamp_v_config = v_in

        # Gibbs sampling
        self.v_samples.set_value(self.sample(self.v_samples.get_value(), 
                                                self.gibbs_steps))

        if self.eval_mask is not None:
            self.v_samples.set_value(v_in * self.eval_mask +
                                        self.v_samples.get_value() *
                                        (1 - self.eval_mask))

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
        """ Estimate the probability of a visible configuration, given the
        activations of the phantasy particles. Call generate_samples() before,
        if visible units changed since the last call of generate_samples(). """

        return self.prob_fun(v_in)

    def get_entropy(self):
        """ Estimate the entropy of a visible configuration, given the
        activations of the phantasy particles. Note that visible units masked
        with 1's do not influence the result. Call generate_samples() before,
        if visible units changed since the last call of generate_samples(). """

        return self.entropy_fun()

    def build_prob_graph(self, v):
        # specify broadcasting dimensions (multiple inputs to multiple
        #phantasy particles)
        if self.convolutional:
            phant_shuf = self.v_samples.dimshuffle('x', 0, 1, 2, 3)
            v_shuf = v.dimshuffle(0, 'x', 1, 2, 3)
        else:
            phant_shuf = self.v_samples.dimshuffle('x', 0, 1)
            v_shuf = v.dimshuffle(0, 'x', 1)

        # Calculate probabilities of current v's to be sampled given phant parts
        # Real-valued observation -> Binomial distribution
        # Binomial coefficient (factorial(x) == gamma(x+1))
        bin_coeff = 1 / (T.gamma(1 + v_shuf) * T.gamma(2 - v_shuf))
        pw_probs = bin_coeff * T.pow(phant_shuf, v_shuf) * T.pow(1. - phant_shuf,
                                                                 1. - v_shuf)

        # Multiply and average probs
        if self.convolutional:
            return T.mean(T.prod(pw_probs, axis=(2,3,4)), axis=0)
        else:
            return T.mean(T.prod(pw_probs, axis=2), axis=0) 
        
    def compile_prob_fun(self):
        v = T.tensor4("v_in") if self.convolutional else T.matrix("v_in")
        self.prob_fun = theano.function([v], self.build_prob_graph(v),
                                        on_unused_input='warn')

    def compile_entropy_fun(self):
        prob_samples = self.build_prob_graph(self.v_samples)

        # average the sum of ic over all phantasy samples
        entropy = T.sum(T.log2(1/prob_samples)) / self.n_samples

        self.entropy_fun = theano.function([], entropy)

    @property
    def mask(self):
        return 1 - self.clamp_mask

    @mask.setter
    def mask(self, value):
        self.clamp_mask = 1 - value