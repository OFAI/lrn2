'''
Created on Dec 8, 2014

@author: Stefan Lattner
'''

import abc
import theano
import theano.tensor as T
import numpy as np
from lrn2.nn_bricks.cost_functions import cross_entropy, squared_error,\
    categorical_crossentropy, kl
from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.notifier import Notifier
from _functools import partial
import logging

LOGGER = logging.getLogger(__name__)

class Cost(object):
    """
    Base class for costs without target (e.g. RBM, reconstruction, rnn prediction)
    """
    def __init__(self, **kwargs):
#         if grad_clip is not None:
#             self.ins_outs = dict([[key, theano.gradient.grad_clip(value, *grad_clip)]
#                                    for key, value in self.ins_outs.items()])
        compile_own_f = partial(Cost.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)

    def compile_functions(self):
        try:
            self.validate = theano.function(self.variables.values(),
                                            self.validate_(),
                                            allow_input_downcast = True,
                                            on_unused_input = 'ignore')
        except AttributeError:
            pass

    def validate_(self):
        return self.cost()

class CostRNNPred(Cost):
    """
    Simple RNN predictor cost (cross_entropy cost)
    """
    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def cost(self):
        return cross_entropy(self.output(self.input),
                             T.roll(self.input, -1, axis = 0))

class CostCD(Cost):
    """
    Basic Contrastive Divergence (CD1) cost.
    Works for single RBM layers only.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_cd = 1, **kwargs):
        if n_cd > 10:
            LOGGER.warning("You are initializing Contrastive Divergence with "
                           "{0} Gibbs sampling steps. ".format(n_cd)
                           + "This could cause problems with maximum recursion "
                           "depth. Consider using Persistent Contrastive "
                           "Divergence for Gibbs chains > 10 steps.")
        Cost.__init__(self, **kwargs)
        self.gibbs_steps = n_cd

    @abc.abstractmethod
    def free_energy_(self, v_in):
        raise NotImplementedError("Method is to be implemented in a UNIT class.")

    @abc.abstractmethod
    def gibbs_step_(self, v_in):
        raise NotImplementedError("Method is to be implemented in a UNIT class.")

    def cost_up(self, v_in):
        return T.mean(self.free_energy_(v_in))

    def cost_down(self, v_in):
        for _ in range(self.gibbs_steps):
            v_in = self.gibbs_step_(v_in)

        return T.mean(self.free_energy_(v_in))

    def cost(self):
        return self.cost_up(self.input) - self.cost_down(self.input)

    def validate_(self):
        return self.recon_err_(self.input)

class CostCDNegData(Cost):
    """
    Contrastive Divergence with explicit data for negative phase.
    Works for single RBM layers only.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_cd = 1, **kwargs):
        if n_cd > 10:
            LOGGER.warning("You are initializing Contrastive Divergence with "
                           "{0} Gibbs sampling steps. ".format(n_cd)
                           + "This could cause problems with maximum recursion "
                           "depth. Consider using Persistent Contrastive "
                           "Divergence for Gibbs chains > 10 steps.")
        Cost.__init__(self, **kwargs)
        self.gibbs_steps = n_cd

    @abc.abstractmethod
    def free_energy_(self, v_in):
        raise NotImplementedError("Method is to be implemented in a UNIT class.")

    @abc.abstractmethod
    def gibbs_step_(self, v_in):
        raise NotImplementedError("Method is to be implemented in a UNIT class.")

    def cost_up(self, v_in):
        return T.mean(self.free_energy_(v_in))

    def cost_down(self, v_in):
        try:
            return T.mean(self.free_energy_(self.variables['neg_phase']))
        except KeyError:
            raise ValueError("Please define an entry with key 'neg_phase' in 'variables'.")

    def cost(self):
        return self.cost_up(self.input) - self.cost_down(self.input)

    def validate_(self):
        return self.recon_err_(self.input)

class Persistent(object):
    __metaclass__ = abc.ABCMeta
    """
    Base class for persistent types (PCD, PCDr)
    """
    def __init__(self, reset_pps_int = -1, **kwargs):
        Persistent.register_plotting(self)
        if reset_pps_int > 0:
            reset_pps_f = lambda **kwargs : self.reset_pps() if kwargs['curr_epoch'] % reset_pps_int == 0 \
                                else lambda *args : args[0]
            self.callback_add(reset_pps_f, Notifier.EPOCH_FINISHED)

    def register_plotting(self):
        self.register_plot(lambda *args : self.pps.get_value(), label = "pps",
                           name_net = self.name)

    @abc.abstractmethod
    def reset_pps(self, dist = 'uniform'):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "actual cost class derived from 'Persistent'.")

class CostPCD(CostCD, Persistent):
    """
    Persistent Constrastive Divergence (PCD) cost. Training with this cost
    enables RBMs to sample properly. Works for single RBM layers only.

    Parameters
    ----------

    input_shape : array-like, optional
        input dimensions (if available)

    fantasy_particles: int, optional
        number of fantasy particles. If None, the batch size is used

    n_cd: int, optional
        gibbs steps for each fantasy particle update (after each batch).
        Should be higher for convolutional nets in order to sample properly.
    """
    def __init__(self, input_shape, fantasy_particles = 1, n_cd = 1,
                 reset_pps_int = -1, **kwargs):
        CostCD.__init__(self, **kwargs)
        Persistent.__init__(self, reset_pps_int)

        self.pps_shape = [fantasy_particles] + list(input_shape)

        """Initialize Fantasy particles """
        if len(self.pps_shape) > 2 and (self.pps_shape[3] is None or self.pps_shape[2] is None):
            raise NotImplementedError("PCD cannot yet deal with dynamic "
                                      "dimension lengths. Hint: Use fixed "
                                      "length training and dynamic length "
                                      "sampling.")
        else:
            self.pps = theano.shared(np.cast[fx](np.random.uniform(0, 1, self.pps_shape)),
                                     borrow=True)

        if self.pps.broadcastable != self.gibbs_step_(self.pps).broadcastable:
            rebroadcast = T.patternbroadcast(self.gibbs_step_(self.pps),
                                             self.pps.broadcastable)
        else:
            rebroadcast = self.gibbs_step_(self.pps)

        self.pps_gibbs_step = self.pps_gibbs_step_fun(rebroadcast)

#         print shape(self.W.get_value())
#         print shape(self.bh.get_value())
#         print self.pps_gibbs_step(self.pps.get_value())
        pps_input_step = partial(self.pps_gibbs_step, self.pps.get_value())
        for _ in range(n_cd):
            self.callback_add(pps_input_step, Notifier.BATCH_FINISHED)

#         self.callback_add(lambda **kwargs: LOGGER.debug("fe fantasy particles = {0}".format(\
#                                                         np.mean(self.free_energy(self.pps.get_value())))),
#                           Notifier.EPOCH_FINISHED)

    def pps_gibbs_step_fun(self, gibbs_graph):
#         return theano.function([self.variables['input']], gibbs_graph,
#                                name = "pps_gibbs_step",
#                                on_unused_input='ignore')
        return theano.function([self.variables['input']], None,
                               updates={self.pps: gibbs_graph},
                               name = "pps_gibbs_step",
                               on_unused_input='ignore')
    def cost_down(self, v_in):
        """ Calculate cost for down-phase """
        cost_fe_down = T.mean(self.free_energy_(self.pps))
        return cost_fe_down

    def reset_pps(self, dist = 'uniform'):

        assert dist in ('uniform', 'binomial') # Only those distributions are implemented

        if dist == 'binomial':
            self.pps.set_value(np.cast[fx](np.random.binomial(1, 0.5, self.pps_shape)))
        elif dist == 'uniform':
            self.pps.set_value(np.cast[fx](np.random.uniform(0, 1, self.pps_shape)))

    def validate_(self):
        # This is the cost which is shown during training
        return self.recon_err_(self.input)

class CostPCDFW(CostPCD):
    """
    Persistent Constrastive Divergence (PCD) with fast weights cost.
    Works for single RBM layers only.

    Parameters
    ----------

    input_shape : array-like, optional
        input dimensions (if available)

    fantasy_particles: int, optional
        number of fantasy particles. If None, the batch size is used

    n_cd: int, optional
        gibbs steps for each fantasy particle update (after each batch).
        Should be higher for convolutional nets in order to sample properly.
    """
    def __init__(self, input_shape, fantasy_particles = None, n_cd = 1,
                 reset_pps_int = -1, max_lr = None, fw_decay = 19./20,
                 **kwargs):
        # Initialize fast weights
        self.Wf = theano.shared(self.get_W_init()*0, name='Wf', borrow=True)
        CostPCD.__init__(self, input_shape, fantasy_particles = fantasy_particles,
                         n_cd = n_cd,
                         reset_pps_int = reset_pps_int, **kwargs)
        # Initialize learning rate
        self.lrfw_ = theano.shared(np.cast[fx](0.), name = 'lrfw')

        def update_Wf():
            W_grad = None
            for p in self.optimizer.gparams:
                if p.name == "W_grad":
                    W_grad = p

            self.Wf.set_value(self.Wf.get_value() * fw_decay - \
                              W_grad.get_value() * self.lrfw)

        self.callback_add(lambda *args : update_Wf(),
                          Notifier.BATCH_FINISHED)

        if max_lr:
            # Register to increase learning rate from 1/3*max_lr to max_lr over training
            self.callback_add(lambda **kwargs: self.set_lr(max_lr,
                                            kwargs['curr_epoch'], kwargs['n_epochs']),
                                       Notifier.EPOCH_FINISHED)
        # Add fast weights to params (to get stored and updated)
        #self.params += [self.Wf]

        self.register_plot(lambda *args : self.Wf.get_value(), label = "fweights",
                           name_net = self.name)
        self.register_plot(lambda *args : self.Wf.get_value(), label = "fweights",
                           ptype = 'hist', name_net = self.name)

    def set_lr(self, max_lr, curr_epoch, n_epochs):
        self.lrfw = 1./3 * max_lr + 2./3 * max_lr * (1. * curr_epoch / n_epochs)


    @property
    def lrfw(self):
        return self.lrfw_.get_value()

    @lrfw.setter
    def lrfw(self, val):
        self.lrfw_.set_value(val)

    # Gibbs step for fantasy particles with fast weights
    def pps_gibbs_step_fun(self, gibbs_graph):
        return theano.function([self.variables['input']], None,
                               updates={self.pps: gibbs_graph},
                               givens={self.W : self.W + self.Wf},
                               name = "pps_gibbs_step",
                               on_unused_input='ignore')


class CostReconErr(Cost):
    """
    Reconstruction Error cost (e.g. for Auto-Encoders).
    Works for single layers and stacks.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def recon_err_(self, v_in):
        return T.sum((self.recon_(v_in) - v_in) ** 2) / T.cast(v_in.shape[0], fx)

    def recon_(self, v_in):
        return self.activation_v(self.activation_h(v_in))

    def cost(self):
        return self.recon_err_(self.input)

    def validate_(self):
        return CostReconErr.cost(self)


class CostReconErrDenoise(Cost):
    """
    Reconstruction Error cost for denoising Auto-Encoders.
    Works for single layers and stacks.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, noise_level = 0.5, **kwargs):
        Cost.__init__(self, **kwargs)
        self.level_corrupt = noise_level

    def cost(self):
        return CostReconErrDenoise.recon_err_(self, self.input)

    def recon_err_(self, v_in):
        return T.sum((self.recon_(self.corrupt(v_in, self.level_corrupt)) \
                      - v_in) ** 2) / v_in.shape[0]

    def recon_(self, v_in):
        return self.activation_v(self.activation_h(v_in))

    def corrupt(self, x, level):
        """ This function keeps '1-level' entries of the inputs the same
        and zero-out randomly selected subset of size 'level'
        """
        return self.t_rng.binomial(size=x.shape, p=1. - level,
                                   dtype=fx) * x

class CostSquaredError(Cost):
    """
    Squared error cost (best used with linear units)
    Works for single layers and stacks.
    """
    def __init__(self, var1 = None, var2 = None, **kwargs):
        Cost.__init__(self, **kwargs)
        if var1 == None:
            var1 = self.output(self.input)
        if var2 == None:
            var2 = self.target
        self.var1 = var1
        self.var2 = var2

    def cost(self):
        return squared_error(self.var1, self.var2)

class CostSquaredErrorGen(Cost):
    """
    Squared error cost for generative targets.
    Works for single layers and stacks.
    """
    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def cost(self):
        return squared_error(self.output(self.input), self.target)

#     def validate_(self):
#         # This is the cost shown during training
#         return squared_error(self.output(self.input), self.target)

class CostVariationalAutoencoder(Cost):
    """
    Cost for Variational Autoencoder.
    """
    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def cost(self):
        mu = self.mu_out(self.input)
        logsig = self.logsigma_out(self.input)
        kl_div = 0.5 * T.sum(1 + 2*logsig - T.sqr(mu) - T.exp(2 * logsig))
        return kl_div * -1


class CostKL(Cost):
    """
    Kullback-Leibler divergence cost (best used with softmax units).
    Works for single layers and stacks.
    """
    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def cost(self):
        return kl(self.target, self.output(self.input))

class CostLogLikelihoodBinomial(Cost):
    """
    Log-lokelihood cost.
    Built for single layers and stacks.

    Parameters
    ----------

    var1 : theano.tensor, optional
        symbolic variable for actual output. If None, the output of the net
        is used.

    var2 : theano.tensor
        symbolic variable for target. If None, the default self.target is used.

    """
    def __init__(self, var1 = None, var2 = None, **kwargs):
        Cost.__init__(self, **kwargs)

        if var1 == None:
            var1 = self.output(self.input)
        if var2 == None:
            var2 = self.target
        self.var1 = var1
        self.var2 = var2

        self.log_like = theano.function([self.input, self.target],
                                        T.sum(T.log(self.binomial_elemwise(self.output(self.input), self.target)), axis=1),
                                        allow_input_downcast = True)

    def binomial_elemwise(self, y, t):
        # specify broadcasting dimensions (multiple inputs to multiple
        # density estimations)
        if self.convolutional:
            est_shuf = y.dimshuffle(0, 1, 2, 3)
            v_shuf = t.dimshuffle(0, 1, 2, 3)
        else:
            est_shuf = y.dimshuffle(0, 1)
            v_shuf = t.dimshuffle(0, 1)

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
            return T.mean(T.sum(T.log(pw_probs), axis=(1,2,3)), axis = 0)
        else:
            return T.mean(T.sum(T.log(pw_probs), axis=1), axis=0)

    def cost(self):
        return -T.mean(self.log_likelihood(self.var1, self.var2))

class CostCrossEntropy(Cost):
    """
    Cross entropy. Targets have to be of set {0,1}, predictions of range (0,1).
    (works fine with sigmoid units).
    Works for single layers and stacks.

    Parameters
    ----------

    var1 : theano.tensor, optional
        symbolic variable for actual output. If None, the output of the net
        is used.

    var2 : theano.tensor
        symbolic variable for target. If None, the default self.target is used.

    scale : boolean
        tells if output should be scaled, so that it never reaches 0 or 1
        (to prevent NaNs)

    """
    def __init__(self, var1 = None, var2 = None, scale = True, **kwargs):
        Cost.__init__(self, **kwargs)

        if var1 == None:
            var1 = self.output(self.input)
        if var2 == None:
            var2 = self.target
        self.var1 = var1
        self.var2 = var2
        self.scale = scale

    def cost(self):
        eps = 1e-8
        if self.scale:
            return cross_entropy(eps + self.var1 * (1 - 2*eps), self.var2)
        return cross_entropy(self.var1, self.var2)

    def validate_(self):
        eps = 1e-8
        if self.scale:
            return cross_entropy(eps + self.var1 * (1 - 2*eps), self.var2)
        return cross_entropy(self.var1, self.var2)

class CostMaxLikelihood(Cost):
    def __init__(self, **kwargs):
        Cost.__init__(self, **kwargs)

    def cost(self):
        return -T.log(T.mean(self.output(self.input)))

class CostCategoricCrossEntropy(Cost):
    """
    Useful for multi-class targets (best used with softmax units)
    Works for single layers and stacks.
    """
    def __init__(self, var1 = None, var2 = None, **kwargs):
        Cost.__init__(self, **kwargs)
        if var1 == None:
            var1 = self.output(self.input)
        if var2 == None:
            var2 = self.target
        self.var1 = var1
        self.var2 = var2

    def cost(self):
        return categorical_crossentropy(self.var1, self.var2)

class CostCrossEntropyAuto(Cost):
    """
    Cross entropy. Targets have to be of set {0,1}, predictions of range (0,1).
    (works fine with sigmoid units).
    Works for single layers and stacks.
    """
    def __init__(self, var1 = None, var2 = None, **kwargs):
        Cost.__init__(self, **kwargs)
        if var1 == None:
            var1 = self.recon_(self.input)
        if var2 == None:
            var2 = self.input
        self.var1 = var1
        self.var2 = var2

    def cost(self):
        return cross_entropy(self.var1, self.var2)

class CostCategoricCrossEntropyAuto(Cost):
    """
    Useful for multi-class targets (best used with softmax units)
    Works for single layers and stacks.
    """
    def __init__(self, var1 = None, var2 = None, **kwargs):
        Cost.__init__(self, **kwargs)
        if var1 == None:
            var1 = self.recon_(self.input)
        if var2 == None:
            var2 = self.input
        self.var1 = var1
        self.var2 = var2

    def cost(self):
        eps = 1e-8
        return categorical_crossentropy(eps + self.var1 * (1-2*eps), self.var2)

class ParallelTempering(object):
    """ *Under construction* Not yet well tested!! """
    def __init__(self, **kwargs):
        n_part = self.pps_shape[0]
        max_temp = 3
        step = 1.0 * (max_temp - 1) / n_part
        self.temperatures = T.arange(1, max_temp - 0.000001, step = step, dtype = fx)
        gibbs_step = self.gibbs_step_
        self.gibbs_step_ = partial(gibbs_step, tempering = self.temperatures)

        self.callback_add(lambda **kwargs: theano.function([kwargs["curr_epoch"]], None,
                                          updates={self.pps: self.swap_particles()},
                                          on_unused_input = 'ignore'),
                          Notifier.EPOCH_FINISHED)

    def swap_particles(self):
        t = self.temperatures
        t_term = (1./t - T.roll(1./t, shift = -1))
        t_term = T.set_subtensor(t_term[-1], 0)

        e_term = self.energy_(self.pps) - T.roll(self.energy_(self.pps),
                                                 shift = -1)
        e_term = T.set_subtensor(e_term[-1], 0.)
        probs = T.exp(t_term * e_term)

        actions = T.cast(T.gt(probs, self.t_rng.uniform((probs.shape))), fx)

        add = T.concatenate([[np.cast[fx](0.)],actions])

        add = T.roll(add, shift = -1) - add
        add = add[:-1]
        add = T.switch(T.gt(add, 0), 1., 0.)
        add = T.set_subtensor(add[-1], 0.)
        add = add - T.roll(add, shift = 1)
        idx = T.arange(actions.shape[0], dtype = fx)
        idx = idx + add

        return self.pps[T.cast(idx, 'int32')]

    def get_probs(self):
        t = self.temperatures
        t_term = (1./t - T.roll(1./t, shift = -1))
        t_term = T.set_subtensor(t_term[-1], 0)

        e_term = self.energy_(self.pps) - T.roll(self.energy_(self.pps),
                                                 shift = -1)
        e_term = T.set_subtensor(e_term[-1], 0.)
        probs = T.exp(t_term * e_term)


        actions = T.cast(T.gt(probs, self.t_rng.uniform((probs.shape))), fx)

        add = T.concatenate([[np.cast[fx](0.)],actions])

        add = T.roll(add, shift = -1) - add
        add = add[:-1]
        add = T.switch(T.gt(add, 0), 1., 0.)
        add = T.set_subtensor(add[-1], 0.)
        add = add - T.roll(add, shift = 1)
        idx = T.arange(actions.shape[0], dtype = fx)
        idx = idx + add

        return self.energy_(self.pps)
