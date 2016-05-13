'''
Created on Dec 8, 2014

@author: Stefan Lattner
'''

import theano
import theano.tensor as T
import numpy as np
import logging
from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.notifier import Notifier
from _functools import partial
from theano.tensor.nnet import nnet

if theano.config.device.startswith('gpu'):
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
else:
    from theano.tensor.shared_randomstreams import RandomStreams

LOGGER = logging.getLogger(__name__)

"""
Units define the activation functions, weight initialization and
(in case of RBMs) - the energy function.
"""

class UnitsNN(object):
    """
    Base class for dense units
    """
    def __init__(self):
        self.act_fun_h = lambda x : x
        self.act_fun_v = lambda x : x
        self.gain = 1.
        
    def get_W_init(self, shape = None):
        if shape == None:
            shape = (self.hidden_shape[0], self.input_shape[0])
        rng = np.random.RandomState()
        W_init = np.asarray(rng.uniform(
                        low = -self.gain * np.sqrt(6. / (shape[0] + shape[1])),
                        high=  self.gain * np.sqrt(6. / (shape[0] + shape[1])),
                        size=shape),
                            dtype=fx)

        return W_init
    
class UnitsCNN(object):
    """
    Base class for convolutional units
    """
    def __init__(self):
        self.act_fun_h = lambda x : x
        self.act_fun_v = lambda x : x
        self.gain = 1.
        
    def get_W_init(self):
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        filter_shape = self.filter_shape
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) / np.prod(self.downsample_out)
        # initialize weights with random weights
        W_bound = self.gain * np.sqrt(6. / (fan_in + fan_out))
        rng = np.random.RandomState()
        W_init = np.asarray(
                            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                            dtype=fx
                            )
        return W_init
    
class UnitsCNNLinear(UnitsCNN):
    """
    Linear convolutional units
    """
    def __init__(self, downsample_out = [1,1]):
        UnitsCNN.__init__(self)
        self.downsample_out = downsample_out
    
class UnitsNNSigmoid(UnitsNN):
    """
    Base class for sigmoid units
    """
    def __init__(self):
        UnitsNN.__init__(self)
        self.act_fun_h = lambda x : T.nnet.sigmoid(x)
        self.act_fun_v = lambda x : T.nnet.sigmoid(x)
        self.gain = 4.

class UnitsNNTanh(UnitsNN):
    """
    Base class for sigmoid units
    """
    def __init__(self):
        UnitsNN.__init__(self)
        self.act_fun_h = lambda x : T.tanh(x)
        self.act_fun_v = lambda x : T.tanh(x)
        self.gain = 1.
        
class UnitsNNLinear(UnitsNN):
    """
    Binary units for Neural Networks with linear hidden activation
    """
    def __init__(self):
        UnitsNN.__init__(self)
        
class UnitsNNLinearNonNeg(UnitsNN):
    """
    Binary units for Neural Networks with linear hidden activation
    with non-negativity regularizer
    """
    def __init__(self):
        UnitsNN.__init__(self)
        self.act_fun_h = lambda x : x
        self.act_fun_v = lambda x : x
        
    def get_W_init(self, shape = None):
        if shape == None:
            shape = (self.hidden_shape[0], self.input_shape[0])
        rng = np.random.RandomState()
        W_init = np.abs(np.asarray(rng.uniform(
                        low = -self.gain * np.sqrt(6. / (shape[0] + shape[1])),
                        high=  self.gain * np.sqrt(6. / (shape[0] + shape[1])),
                        size=shape),
                            dtype=fx))

        return W_init

class UnitsNNSoftmax(UnitsNN):
    """
    Binary units for Neural Networks with softmax hidden activation
    """
    def __init__(self):
        UnitsNN.__init__(self)
        self.act_fun_h = lambda x : T.nnet.softmax(x)

class UnitsNNReLU(UnitsNN):
    """
    Rectified linear units for standard NNs
    """
    def __init__(self):
        UnitsNN.__init__(self)
        self.act_fun_h = lambda x : T.maximum(0.0001*x, x)
        self.act_fun_v = lambda x : x
        self.gain = np.sqrt(2)


class UnitsRBMReLU(UnitsNNReLU):
    """
    Rectified linear units for standard RBMs
    """
    def __init__(self):
        UnitsNNReLU.__init__(self)

    def gibbs_step_(self, v_in):
        h_in = self.activation_h(v_in)
        h_bin = (h_in + self.t_rng.normal(h_in.shape, 0.0, nnet.sigmoid(h_in)))
        h_bin = T.maximum(0, h_bin)
        return self.activation_v(T.cast(h_bin, fx))

    def free_energy_(self, v_in):
        vbias_term = T.sum(((v_in - self.bv)**2)/2, axis=1)
        fe = vbias_term - T.sum(T.log(1 + T.exp(self.input_h(v_in))), axis=1)
        return fe

class UnitsRBMSigmoid(UnitsNNSigmoid):
    """
    Sigmoid units for RBMs
    """
    def __init__(self):
        UnitsNNSigmoid.__init__(self)
        
    def free_energy_(self, v_in):
        return - T.dot(v_in, self.bv) \
                    - T.sum(T.nnet.softplus(self.input_h(v_in)), axis=1)

    def gibbs_step_(self, v_in):
        h_in = self.activation_h(v_in)
        h_bin = T.gt(h_in, self.t_rng.uniform((h_in.shape)))
        return self.activation_v(T.cast(h_bin, fx))

class UnitsCNNSigmoid(UnitsNNSigmoid, UnitsCNN):
    """
    Sigmoid units for convolutional NN
    """
    def __init__(self, downsample_out = [1,1]):
        UnitsCNN.__init__(self)
        UnitsNNSigmoid.__init__(self)
        self.downsample_out = downsample_out
        
    def get_W_init(self):
        return UnitsCNN.get_W_init(self)
        
class UnitsCNNReLU(UnitsNNReLU, UnitsCNN):
    """
    ReLU units for convolutional NNs
    """
    def __init__(self, downsample_out = [1,1]):
        UnitsCNN.__init__(self)
        UnitsNNReLU.__init__(self)
        self.downsample_out = downsample_out
        
    def get_W_init(self):
        return UnitsCNN.get_W_init(self)
    
    
class UnitsCRBMReLU(UnitsCNN, UnitsNNReLU):
    """
    Rectified linear units for convolutional RBMs
    """
    def __init__(self, gauss = False, downsample_out = (1,1)):
        UnitsCNN.__init__(self)
        UnitsNNReLU.__init__(self)
        self.downsample_out = downsample_out
        if not gauss:
            self.act_fun_v = lambda x : T.maximum(0.0001*x, x)
#            self.act_fun_v = lambda x : nnet.sigmoid(x)
            
    def gibbs_step_(self, v_in):
        h_in = self.activation_h(v_in)
        h_bin = (h_in + self.t_rng.normal(h_in.shape, 0.0, nnet.sigmoid(h_in)))
        h_bin = T.maximum(0., h_bin)
        return self.activation_v(T.cast(h_bin, fx))

    def free_energy_(self, v_in):
        if self.bv.ndim == 2:
            bias_v = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias_v = self.bv.dimshuffle('x', 0, 'x', 'x')
            
        vbias_term = T.sum(((v_in - bias_v)**2)/2, axis=(1,2,3))
        fe = vbias_term - T.sum(T.nnet.softplus(self.input_h(v_in)), axis=(1,2,3))
        return fe
    
class UnitsCRBMSigmoid(UnitsCNN, UnitsNNSigmoid):
    """
    Binary units for convolutional RBMs
    """
    def __init__(self, downsample_out = (1,1), compile_functions = True):
        UnitsCNN.__init__(self)
        UnitsNNSigmoid.__init__(self)
        self.downsample_out = downsample_out
        if compile_functions:
            self.callback_add(partial(UnitsCRBMSigmoid.compile_functions, self),
                              Notifier.MAKE_FINISHED)
        
    def get_W_init(self):
        return UnitsCNN.get_W_init(self)

    def gibbs_step_(self, v_in, tempering = None):
        h_in = self.activation_h(v_in, tempering)
        h_bin = T.gt(h_in, self.t_rng.uniform((h_in.shape)))
        return self.activation_v(T.cast(h_bin, fx))

    def free_energy_(self, v_in):
        if self.bv.ndim == 2:
            bias = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias = self.bv.dimshuffle('x', 0, 'x', 'x')

        return - T.sum(v_in * bias,
                       axis=[1,2,3]) - T.sum(T.nnet.softplus(self.input_h(v_in)),
                                             axis=[1,2,3])
                       
    def free_energy_over_x(self, v_in):
        if self.bv.ndim == 2: 
            bias = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias = self.bv.dimshuffle('x', 0, 'x', 'x')

        try:
            return - T.sum(v_in * bias,
                           axis=[1,3])[:,::self.stride[0]] \
                            - T.sum(T.nnet.softplus(self.input_h(v_in, omit_bias = True)), 
                                    axis=[1,3])
        except TypeError:
            return - T.sum(v_in * bias,
                           axis=[1,3])[:,::self.stride[0]] \
                            - T.sum(T.nnet.softplus(self.input_h(v_in)), axis=[1,3])
                                      
    def compile_functions(self):
        self.fe_over_x = theano.function([self.variables['input']], 
                                         self.free_energy_over_x(self.input))

class UnitsCRBMSigmoidNonNeg(UnitsCRBMSigmoid):
    """
    Sigmoid units for non-negative convolutional RBMs
    """
    def __init__(self):
        UnitsCRBMSigmoid.__init__(self)
        
    def get_W_init(self):
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        filter_shape = self.filter_shape
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) / np.prod(self.downsample_out)
        # initialize weights with random weights
        W_bound = self.gain * np.sqrt(6. / (fan_in + fan_out))
        rng = np.random.RandomState()
        W_init = np.asarray(
                            rng.uniform(low=0, high=W_bound, size=filter_shape),
                            dtype=fx
                            )
        return W_init

class UnitsNNGauss(object):
    """
    Gaussian units for standard NNs
    """
    def __init__(self):
        self.act_fun_h = lambda x : T.nnet.sigmoid(x)
        self.act_fun_v = lambda x : x

    def get_W_init(self, shape = None):
        if shape is None:
            shape = (self.hidden_shape[0], self.input_shape[0])

        # initialize weights with random weights
        rng = np.random.RandomState()

        # weights initialized to small random values N(0,0.01)
        W_init = np.asarray(rng.normal(0., 0.05, size=shape), dtype=fx)

        return W_init

class UnitsRBMGauss(UnitsNNGauss):
    """
    Gaussian units for standard RBMs
    """
    def __init__(self):
        UnitsNNGauss.__init__(self)

    def gibbs_step_(self, v_in):
        h_in = self.activation_h(v_in)
        h_bin = T.gt(h_in, self.t_rng.uniform((h_in.shape)))
        return self.activation_v(T.cast(h_bin, fx))

    def free_energy_(self, v_in):
        vbias_term = T.sum((v_in - self.bv)**2, axis=1)
        fe = vbias_term - T.sum(T.log(1 + T.exp(self.input_h(v_in))), axis=1)
        return fe

class UnitsCRBMGauss(UnitsNNGauss):
    """
    Gaussian units for convolutional RBMs
    """
    def __init__(self):
        UnitsNNGauss.__init__(self)

    def get_W_init(self):
        # initialize weights with random weights
        fan_in = self.filter_shape[2] * self.filter_shape[3]
        filter_shape = self.filter_shape
        rng = np.random.RandomState()
        W_init = np.asarray(rng.normal(0, 0.05 / np.sqrt(fan_in),
                                       size = filter_shape),
                            dtype=fx)

        return W_init

    def gibbs_step_(self, v_in):
        h_in = self.activation_h(v_in)
        h_bin = T.gt(h_in, self.t_rng.uniform((h_in.shape)))
        return self.activation_v(T.cast(h_bin, fx))

    def free_energy_(self, v_in):
        if self.bv.ndim == 2:
            bias = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias = self.bv.dimshuffle('x', 0, 'x', 'x')
        vbias_term = T.sum(((v_in - bias)**2)/2,
                           axis=[1,2,3])
        fe = vbias_term - T.sum(T.log(1 + T.exp(self.input_h(v_in))), axis=[1,2,3])
        return fe

class UnitsDropOut(object):
    """
    Adds Dropout to any unit type.
    """
    def __init__(self, variables, dropout_h = 0., dropout_v = 0., **kwargs):
        try:
            variables['input']
        except KeyError:
            raise KeyError("Dictionary 'variables' needs an entry with key 'input'")
        
        rng = np.random.RandomState()
        self.t_rng = RandomStreams(rng.randint(2 ** 30))
        
        self.level_h_ = theano.shared(np.cast[fx](dropout_h))
        self.level_v_ = theano.shared(np.cast[fx](dropout_v))

        act_fun_h = self.act_fun_h
        self.act_fun_h = lambda x : self.dropout(act_fun_h(x), self.level_h_)

        self.input = self.dropout(variables['input'], self.level_v_)

        self.do_suspended = False
        self.callback_add(partial(self.dropout_suspend, True),
                          Notifier.MAKE_FINISHED, forward = True)
        self.callback_add(partial(self.dropout_suspend, False),
                          Notifier.TRAINING_START, forward = True)
        self.callback_add(partial(self.dropout_suspend, True),
                          Notifier.TRAINING_STOP, forward = True)

    def dropout(self, x, level):
        """ This function keeps '1-level' entries of the inputs the same
        and zero-out randomly selected subset of size 'level'
        """
        return self.t_rng.binomial(size=x.shape, p=1. - level, dtype=fx) * x

    def dropout_suspend(self, suspend=True):
        if suspend:
            if not self.do_suspended:
                self.level_v_tmp = self.level_v
                self.level_h_tmp = self.level_h
                self.W.set_value(self.W.get_value() / (1 / (1-self.level_h) * (1-self.level_v)))
                self.level_v = 0.
                self.level_h = 0.
                self.do_suspended = True
            else:
                LOGGER.warning("Dropout already suspended, nothing to do.")
        else:
            if self.do_suspended:
                self.level_v = self.level_v_tmp
                self.level_h = self.level_h_tmp
                self.W.set_value(self.W.get_value() * (1 / (1-self.level_h) * (1-self.level_v)))
                self.do_suspended = False
            else:
                LOGGER.warning("Dropout was not suspended, nothing to do.")

    @property
    def level_h(self):
        return self.level_h_.get_value()

    @level_h.setter
    def level_h(self, value):
        #assert not self.do_suspended, "Please unsuspend dropout to change its level."
        self.level_h_.set_value(value)

    @property
    def level_v(self):
        return self.level_v_.get_value()

    @level_v.setter
    def level_v(self, value):
        #assert not self.do_suspended, "Please unsuspend dropout to change its level."
        self.level_v_.set_value(value)

class UnitsDropoutColums(object):
    """
    Drops out whole columns in CNN or CRBM. Can improve results.
    """
#     def __init__(self, level_h = 0., level_v = 0.):
#         UnitsDropOut.__init__(self, level_h, level_v)

    def dropout_c(self, x, level):
        """ This function keeps '1-level' entries of the inputs the same
        and zero-out randomly selected subset of size 'level'
        """
        m = self.t_rng.binomial(size=(x.shape[2], x.shape[3]), p=1. - level, dtype=fx)
        m = m.dimshuffle('x', 'x', 0, 1)
        return m * x

    """
    Adds Column Dropout to convolutional Units (deactivates units over all maps).
    """
    def __init__(self, level_h = 0., level_v = 0.):
        self.level_h_c_ = theano.shared(np.cast[fx](level_h))
        self.level_v_c_ = theano.shared(np.cast[fx](level_v))

        act_fun_h = self.act_fun_h
        self.act_fun_h = lambda x : self.dropout_c(act_fun_h(x), self.level_h_c_)

        self.input = lambda x : self.dropout_c(self.input, self.level_v_c_)

        self.do_suspended_c = False
        self.callback_add(partial(self.dropout_suspend_c, True), Notifier.MAKE_FINISHED)

    def dropout_suspend_c(self, suspend=True):
        if suspend:
            if not self.do_suspended_c:
                self.level_v_tmp_c = self.level_v_c
                self.level_h_tmp_c = self.level_h_c
                self.W.set_value(self.W.get_value() / (1 / (1-self.level_h_c) * (1-self.level_v_c)))
                self.level_v_c = 0.
                self.level_h_c = 0.
                self.do_suspended_c = True
            else:
                LOGGER.warning("Dropout (columns) already suspended, nothing to do.")
        else:
            if self.do_suspended_c:
                self.level_v_c = self.level_v_tmp_c
                self.level_h_c = self.level_h_tmp_c
                self.W.set_value(self.W.get_value() * (1 / (1-self.level_h_c) * (1-self.level_v_c)))
                self.do_suspended_c = False
            else:
                LOGGER.warning("Dropout (columns) was not suspended, nothing to do.")

    @property
    def level_h_c(self):
        return self.level_h_c_.get_value()

    @level_h_c.setter
    def level_h_c(self, value):
        self.level_h_c_.set_value(value)

    @property
    def level_v_c(self):
        return self.level_v_c_.get_value()

    @level_v_c.setter
    def level_v_c(self, value):
        self.level_v_c_.set_value(value)

class UnitsNoisy(object):
    """
    Adds noise to units *before* the activation function
    """
    def __init__(self, level = 0.0):
        self.level_ = theano.shared(np.cast[fx](level))

        act_fun_h = self.act_fun_h
        self.act_fun_h = lambda x : act_fun_h(self.noise(x, self.level_))

    def noise(self, x, level):
        """
        Adds some gaussian noise to the unit's activation
        """
        return self.t_rng.normal(size=x.shape, avg=0.0, std=level) + x

    @property
    def level_h(self):
        return self.level_.get_value()

    @level_h.setter
    def level_h(self, value):
        self.level_.set_value(value)
