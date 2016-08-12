'''
Created on Mar 31, 2015

@author: Stefan Lattner
'''

import abc
import copy
import theano
import logging
import numpy as np
import theano.tensor as T
from _functools import partial

from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.plot import Plotter
from lrn2.nn_bricks.notifier import Notifier
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import sys

if theano.config.device.startswith('gpu'):
    from theano.sandbox.cuda.dnn import dnn_conv as conv2d
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
    contig = gpu_contiguous
else:
    from theano.tensor.nnet.conv import conv2d
    from theano.tensor.shared_randomstreams import RandomStreams
    contig = lambda x : x

LOGGER = logging.getLogger(__name__)

class NNBase(object):
    """
    Basic class for Neural Network components (layers, stacks,..).
    Empty initialization of variables, cost, params, and registration of plots
    for params and data.

    Parameters
    ----------

    name : string
        name of the net

    plot_params : boolean, optional
        switch for plotting parameters of the model

    plot_dparams : boolean, optional
        switch for plotting histograms of delta params

    tiling : string
        tiling of parameters, either 'default' or 'corpus'
    """
    plotting_registered = False
    def __init__(self, name, plot_params = True, plot_dparams = True,
                 tiling = 'corpus', **kwargs):
        rng = np.random.RandomState()
        self.t_rng = RandomStreams(rng.randint(2 ** 30))

        self.name = name
        self.variables = {}
        self.cost_ = np.cast[fx](0.)
        self.params = []

        self.plot_dparams = plot_dparams
        self.plot_params = plot_params
        self.tiling_params = tiling
        if isinstance(self, Plotter) and not self.plotting_registered:
            register_plot = partial(NNBase.register_plotting, self)
            self.callback_add(register_plot, Notifier.REGISTER_PLOTTING)
            self.plotting_registered = True

    def register_plotting(self):
        # Plot data
        def get_value(data, key):
            return data[key]
        def get_value_flat(data, key):
            return np.asarray(data[key]).flatten()

        for key in self.variables.keys():
            # Use partial, because of dynamic binding lambda notation would
            # always use the last 'key'.
            self.register_plot(partial(get_value, key=key), label = key,
                               forward = False, name_net = self.name)
            self.register_plot(partial(get_value_flat, key=key),
                               label = key, ptype = 'hist', forward = False,
                               name_net = self.name)

        if self.plot_params:
            # Plot params
            def get_val(p, *args, **kwargs):
                return p.get_value()
            def get_val_flat(p, *args, **kwargs):
                return p.get_value().flatten()

            for p in self.params:
                self.register_plot(partial(get_val, p), name_net = self.name,
                                   label = p.name,
                                   tiling = self.tiling_params)
                self.register_plot(partial(get_val_flat, p), name_net = self.name,
                                   label = p.name,
                                   ptype = 'hist')

        # Plot delta params
        def diff(curr_p, last_p, *args, **kwargs):
            return (curr_p.get_value() - last_p).flatten()

        if self.plot_dparams:
            for p in self.params:
                self.register_plot(partial(diff, p, self.last_params[p.name]),
                                   label = "d" + p.name, ptype = 'hist',
                                   name_net = self.name)

class FFBase(NNBase):
    """
    Basic class for Neural Network components (layers and stacks),
    which have one main input and one output.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like, optional
        input dimensions

    hidden_shape : array-like, optional
        shape of hidden layer (in convolutional nets, some
        dimensions might be undefined)

    plot_dparams : boolean, optional
        switch for plotting histograms of delta params

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, variables, name, input_shape = (), hidden_shape = (),
                 plot_params = True, plot_dparams = True, **kwargs):

        NNBase.__init__(self, name, plot_params, plot_dparams, **kwargs)

        for key, v in variables.items():
            try:
                assert v.owner == None, "Only basic tensor types, no graphs allowed " + \
                            "in dictionary 'variables': {0}: {1}".format(key, v)
            except AttributeError:
                raise AttributeError("Only basic tensor types allowed in " + \
                                     "dictionary 'variables': {0}".format(type(v)))
        self.variables = variables
        if not hasattr(self, 'input'):
            try:
                self.input = variables['input']
            except KeyError:
                raise ValueError("dict 'variables' has to contain key 'input'.")

        if not hasattr(self, 'target'):
            try:
                self.target = variables['target']
            except KeyError:
                self.target = None

        self.input_shape = input_shape
        self.hidden_shape = hidden_shape

        LOGGER.debug("input.shape" + str(self.input_shape))
        LOGGER.debug("hidden.shape" + str(self.hidden_shape))

        compile_own_f = partial(FFBase.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)

    @abc.abstractmethod
    def activation_v(self, h_in):
        raise NotImplementedError("Method is to be implemented in derived class.")

    @abc.abstractmethod
    def activation_h(self, v_in):
        raise NotImplementedError("Method is to be implemented in derived class.")

    @abc.abstractmethod
    def output(self, v_in):
        raise NotImplementedError("Method is to be implemented in derived class.")

    @property
    def convolutional(self):
        return self.input.ndim in [3, 4, 5]  # Conv{1,2,3}DLayer

    def compile_functions(self):
        """ Compile theano functions """
        variables = copy.copy(self.variables)
        try:
            variables.pop('target')
        except KeyError:
            pass
        self.out = theano.function(variables.values(),
                                   self.output(self.input),
                                   allow_input_downcast = True,
                                   on_unused_input = 'ignore')

class RBMBase(FFBase):
    """
    Basic class for Restricted Boltzmann Machine layers and stacks, as they
    share the same basic properties.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like
        input dimensions

    hidden_shape : array-like, optional
        shape of hidden layer, if available (in convolutional nets, some
        dimensions can remain undefined)

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, variables, name, input_shape, hidden_shape = None,
                 **kwargs):

        FFBase.__init__(self, variables, name, input_shape, hidden_shape,
                        **kwargs)

        compile_own_f = partial(RBMBase.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)

        register_plot = partial(RBMBase.register_plotting, self)
        self.callback_add(register_plot, Notifier.REGISTER_PLOTTING)

    def recon_err_(self, v_in):
        return T.sum((self.recon_(v_in) - v_in) ** 2) / T.cast(v_in.shape[0], fx)

    def recon_(self, v_in):
        return self.activation_v(self.activation_h(v_in))

    @abc.abstractmethod
    def free_energy_(self, v_in):
        raise NotImplementedError("Method is to be implemented in a unit class.")

    def compile_functions(self):
        """ Compile theano functions """
        self.free_energy = theano.function(self.variables.values(),
                                           self.free_energy_(self.input),
                                           on_unused_input = 'ignore')

        self.gibbs_step = theano.function(self.variables.values(),
                                          self.gibbs_step_(self.input),
                                          allow_input_downcast=True,
                                           on_unused_input = 'ignore')

        self.recon = theano.function(self.variables.values(),
                                     self.recon_(self.input),
                                     allow_input_downcast = True,
                                           on_unused_input = 'ignore')

        self.recon_err = theano.function(self.variables.values(),
                                         self.recon_err_(self.input),
                                         allow_input_downcast = True,
                                           on_unused_input = 'ignore')

    def register_plotting(self):
        self.register_plot(lambda data : self.recon(data['input']), "recon",
                           forward = False, name_net = self.name)
        self.register_plot(lambda data : self.recon(data['input']), "recon",
                           forward = False, name_net = self.name, ptype = 'hist')

class NN(FFBase):
    """
    Standard NN Layer

    Allocate a standard Neural Network Layer.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like
        input dimensions

    n_hidden : int
        number of hidden units

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    bias_h : float
        initial hidden bias (for cheap sparsity)

    """
    def __init__(self, variables, name, input_shape, n_hidden, params=None,
                 bias_h = 0., **kwargs):

        FFBase.__init__(self, variables, name, input_shape = input_shape,
                        hidden_shape = (n_hidden,), **kwargs)

        if params is None:
            # inputs to each hidden unit
            self.W = theano.shared(self.get_W_init([n_hidden] + list(input_shape)),
                                   name = "W")
            # create shared variable for hidden units bias
            self.bh = theano.shared(value=np.zeros(self.hidden_shape, dtype=fx) \
                                    + bias_h, name='bh')

            # create shared variable for visible units bias
            self.bv = theano.shared(value=np.zeros(input_shape, dtype=fx),
                                    name='bv')

            self.params = [self.W, self.bh, self.bv]
        else:
            self.params = params

        self.callback_add(partial(NN.register_plotting_compiled, self),
                          Notifier.REGISTER_PLOTTING)

    def activation_v(self, h_in):
        return self.act_fun_v(self.input_v(h_in))

    def input_v(self, h_in):
        return T.dot(h_in, self.W) + self.bv

    def activation_h(self, v_in):
        return self.act_fun_h(self.input_h(v_in))

    def output(self, v_in):
        return self.activation_h(v_in)

    def input_h(self, v_in):
        return T.dot(v_in, self.W.T) + self.bh

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

    def register_plotting_compiled(self):
        if self.tiling_params:
            tiling = self.tiling_params
        else:
            tiling = "corpus" if self.convolutional else "default"

        self.register_plot(lambda data : self.out(*(data.values())), 'out',
                           tiling = tiling, forward = False,
                           name_net = self.name)
        self.register_plot(lambda data :
                           np.asarray(self.out(*(data.values()))).flatten(), 'out',
                           ptype = 'hist', forward = False,
                           name_net = self.name)

class NN_BN(NN):
    """
    Standard NN Layer with batch normalization.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like
        input dimensions

    n_hidden : int
        number of hidden units

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    bias_h : float
        initial hidden bias (for cheap sparsity)

    """
    def __init__(self, variables, name, input_shape, n_hidden, params=None,
                 bias_h = 0., **kwargs):

        NN.__init__(self, variables, name, input_shape, n_hidden, params,
                    bias_h, **kwargs)
        self.gamma_bn = theano.shared(np.ones(self.hidden_shape, dtype = fx),
                                      name = "gamma_bn")
        self.beta_bn = theano.shared(np.zeros(self.hidden_shape, dtype = fx),
                                     name = "beta_bn")
        self.params = [self.W, self.beta_bn, self.gamma_bn]

    def input_h(self, v_in):
        in_ = T.dot(v_in, self.W.T)
        mean = T.mean(in_, axis = 0, dtype = fx)
        std = T.std(in_, axis = 0)
        x = (in_ - mean) / std
        return x * self.gamma_bn + self.beta_bn

    def input_v(self, h_in):
        raise NotImplementedError("Reverse of batch normalization in dense "
                                  "NNs not yet implemented.")

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

class NNAuto(NN):
    """
    Auto-Encoder Layer.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like
        input dimensions

    n_hidden : int
        number of hidden units

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    bias_h : float
        initial hidden bias (for cheap sparsity)

    """
    def __init__(self, variables, name, input_shape, n_hidden, params=None,
                 bias_h = 0., **kwargs):
        NN.__init__(self, variables, name, input_shape, n_hidden, params,
                    bias_h, **kwargs)
        self.target = self.input

    def output(self, v_in):
        return NN.recon_(self, v_in)

class CNN(FFBase):
    """
    Convolutional NN base class.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 3
        shape of input, as (visible maps, image height, image width)

    filter_shape : tuple or list of length 4
        shape of filter, as
        (hidden maps, visible maps, filter height, filter width)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    ext_bias : string in ('v', 'h', 'both'), optional
        extends bias by dimension 0 (e.g. for convolution in time) for
        hidden units ('h'), visible units ('v') or ('both')

    stride : tuple
        2D tuple of integer numbers, defining the stride of the convolution

    """

    def __init__(self, variables, name, input_shape, filter_shape, bias_h = 0,
                 params = None, ext_bias = 'v', stride = (1,1), **kwargs):

        FFBase.__init__(self, variables, name, input_shape, **kwargs)
        assert input_shape[0] == filter_shape[1], "input_shape[0] = {0}, filter_shape[1] = {1}".format(input_shape[0], filter_shape[1])

        assert ext_bias in ('v', 'h', 'both')

        self.filter_shape = filter_shape

        if input_shape[2] is not None:
            self.n_filt_x = input_shape[2] - filter_shape[3] + 1
        else:
            self.n_filt_x = None

        self.n_filt_y = input_shape[1] - filter_shape[2] + 1
        self.stride = stride

        LOGGER.debug("filter.shape" + str(self.filter_shape))
        LOGGER.debug("image.shape" + str(self.input_shape))
#         LOGGER.debug("hidden.shape" + str((self.n_filt_y, self.n_filt_x, )))

        if params is None:
            # there are "num input_sym feature maps * filter height * filter width"
            # inputs to each hidden unit
            self.W = theano.shared(self.get_W_init(),
                                   name="W", borrow=True)

            # the bias is a 1D tensor -- one bias per output feature map
            bh_values = np.zeros((filter_shape[0],), dtype=fx)
            if ext_bias in ('h', 'both'):
                bh_values = np.zeros((filter_shape[0], input_shape[2],), dtype=fx)

            bh_values = bh_values + np.cast[fx](bias_h)

            self.bh = theano.shared(value=bh_values, name = 'bh', borrow=True)

            # a constant bias bv for visible units
#             self.bv = theano.shared(np.cast[fx](0))

            bv_values = np.zeros(input_shape[0], dtype=fx)
            if ext_bias in ('v', 'both'):
                bv_values = np.zeros((input_shape[0],input_shape[2]), dtype=fx)
            self.bv = theano.shared(value=bv_values, name = 'bv', borrow = True)

            self.params = [self.W, self.bh, self.bv]
        else:
            self.params = params

        self.callback_add(partial(CNN.register_plotting_compiled, self),
                          Notifier.REGISTER_PLOTTING)


    def activation_v(self, h_in):
        return self.act_fun_v(self.input_v(h_in))

    def get_bmode(self):
        b_mode1 = self.filter_shape[2] // 2
        b_mode2 = self.filter_shape[3] // 2

        sub = np.array(self.input_shape) - np.array(self.filter_shape[1:])
        if sub[1] == 0:
            b_mode1 = 0
        if sub[2] == 0:
            b_mode2 = 0

        if theano.config.device.startswith('gpu'):
            return np.cast[fx](b_mode1), np.cast[fx](b_mode2)
        else:
            return 'valid'


    def input_h(self, v_in, tempering = None):
        # convolve hidden maps with inverted filters
        b_mode = self.get_bmode()

        conv_out = conv2d(
            v_in, self.W[:,:,::-1,::-1],
            border_mode = b_mode,
            subsample = tuple(self.stride)
        )

        conv_out = contig(conv_out)
        if tempering is None:
            if self.bh.ndim == 2:
                return conv_out + self.bh.dimshuffle('x', 0, 'x', 1)
            else:
                return conv_out + self.bh.dimshuffle('x', 0, 'x', 'x')
        else:
            if self.bh.ndim == 2:
                return conv_out + self.bh.dimshuffle('x', 0, 'x', 1) \
                                    / tempering.dimshuffle(0, 'x', 'x', 'x')
            else:
                return conv_out + self.bh.dimshuffle('x', 0, 'x', 'x') \
                                    / tempering.dimshuffle(0, 'x', 'x', 'x')

    def input_v(self, h_in):
#         v_in = self.input
        input_shape = self.input_shape
        v_in = theano.sandbox.cuda.basic_ops.gpu_alloc_empty(h_in.shape[0],
                                                             *input_shape)

        b_mode = self.get_bmode()
        fake_fwd = conv2d(
            v_in, self.W[:,:,::-1,::-1],
            border_mode = b_mode,
            subsample = tuple(self.stride)
        )

        conv_out = theano.grad(None, wrt=v_in, known_grads={fake_fwd: h_in})
        if self.bv.ndim == 2:
            bias = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias = self.bv.dimshuffle('x', 0, 'x', 'x')

        return conv_out + bias

    def activation_h(self, v_in, tempering = None):
        return self.act_fun_h(self.input_h(v_in, tempering))

    def output(self, v_in):
        return self.activation_h(v_in)

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

    def register_plotting_compiled(self):
        if self.tiling_params:
            tiling = self.tiling_params
        else:
            tiling = "corpus" if self.convolutional else "default"

        self.register_plot(lambda data : self.out(*(data.values())), 'out',
                           tiling = tiling, forward = False,
                           name_net = self.name)
        self.register_plot(lambda data : self.out(*(data.values())).flatten(),
                           'out', forward = False, ptype = 'hist',
                           name_net = self.name)


class CNN_BN(CNN):
    """
    Convolutional NN with batch normalization.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    filter_shape : tuple or list of length 4
        shape of filter, as
        (hidden maps, visible maps, filter height, filter width)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    ext_bias : string in ('v', 'h', 'both'), optional
        extends bias by dimension 0 (e.g. for convolution in time) for
        hidden units ('h'), visible units ('v') or ('both')

    stride : tuple
        2D tuple of integer numbers, defining the stride of the convolution

    """
    def __init__(self, variables, name, input_shape, filter_shape,
                 bias_h = 0, params = None, ext_bias = 'v', stride = (1,1),
                 **kwargs):
        CNN.__init__(self, variables, name, input_shape, filter_shape, bias_h,
                     params, ext_bias, stride = stride, **kwargs)
        self.gamma_bn = theano.shared(np.ones(self.filter_shape[0], dtype = fx),
                                      name = "gamma_bn")
        self.beta_bn = theano.shared(np.zeros(self.filter_shape[0], dtype = fx) + bias_h,
                                     name = "beta_bn")
        self.params = [self.W, self.beta_bn, self.gamma_bn]

    def input_h(self, v_in, tempering = None):
        # convolve hidden maps with inverted filters

        b_mode = self.get_bmode()
        conv_out = conv2d(
            v_in, self.W[:,:,::-1,::-1],
            border_mode = b_mode,
            subsample = tuple(self.stride)
        )

        # Try to remove with newer theano versions
        conv_out = contig(conv_out)

        in_ = conv_out
        mean = T.mean(in_, axis = (0,2,3), dtype = fx)
        std = T.std(in_, axis = (0,2,3))
        x = (in_ - mean.dimshuffle('x', 0, 'x', 'x')) / std.dimshuffle('x', 0, 'x', 'x')
        return x * self.gamma_bn.dimshuffle('x', 0, 'x', 'x') + self.beta_bn.dimshuffle('x', 0, 'x', 'x')

    def input_v(self, h_in):
        input_shape = self.input_shape
        v_in = theano.sandbox.cuda.basic_ops.gpu_alloc_empty(h_in.shape[0],
                                                             *input_shape)
        b_mode = self.get_bmode()

        # Go forward to derive the mean and std used to normalize
        # and invert the batch normalization with those values
        conv_out_h = conv2d(
            v_in, self.W[:,:,::-1,::-1],
            border_mode = b_mode,
            subsample = tuple(self.stride)
        )

        conv_out_h = contig(conv_out_h)

        in_ = conv_out_h
        mean = T.mean(in_, axis = (0,2,3), dtype = fx)
        std = T.std(in_, axis = (0,2,3))
        h_in_denorm = (h_in - self.beta_bn.dimshuffle('x', 0, 'x', 'x')) * self.gamma_bn.dimshuffle('x', 0, 'x', 'x')
        h_in_denorm = h_in_denorm * std.dimshuffle('x', 0, 'x', 'x') + mean.dimshuffle('x', 0, 'x', 'x')

        fake_fwd = conv2d(
            v_in, self.W[:,:,::-1,::-1],
            border_mode = b_mode,
        )

        conv_out = theano.grad(None, wrt=v_in, known_grads={fake_fwd: h_in_denorm})

        if self.bv.ndim == 2:
            bias = self.bv.dimshuffle('x', 0, 'x', 1)
        else:
            bias = self.bv.dimshuffle('x', 0, 'x', 'x')

        return conv_out + bias

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

class DCNN(CNN):
    """
    De-Convolutional NN with batch normalization.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    filter_shape : tuple or list of length 4
        shape of filter, as
        (hidden maps, visible maps, filter height, filter width)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    ext_bias : string in ('v', 'h', 'both'), optional
        extends bias by dimension 0 (e.g. for convolution in time) for
        hidden units ('h'), visible units ('v') or ('both')

    stride : tuple
        2D tuple of integer numbers, defining the stride of the convolution

    """
    def __init__(self, variables, name, input_shape, filter_shape, bias_h = 0,
                 params = None, ext_bias = 'v', stride = (1,1), **kwargs):

        CNN.__init__(self, variables, name, input_shape, filter_shape, bias_h,
                     params, ext_bias, stride = stride, **kwargs)

    def activation_h(self, v_in, tempering = None):
        return CNN.activation_v(self, v_in)

    def activation_v(self, h_in):
        return CNN.activation_h(self, h_in)


class RBM(NN, RBMBase):
    """
    Standard Restriced Boltzmann Machine

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : array-like
        input dimensions

    n_hidden : int
        number of hidden units

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0
    """
    def __init__(self, variables, name, input_shape, n_hidden, params=None,
                 bias_h = 0., **kwargs):
        RBMBase.__init__(self, variables, name, input_shape, n_hidden,
                         **kwargs)
        NN.__init__(self, variables, name, input_shape, n_hidden,
                    params = params, bias_h = bias_h, **kwargs)


class CRBM(CNN, RBMBase):
    """
    Convolutional RBM as described in "Convolutional Deep Belief Networks for
    Scalable Unsupervised Learning of Hierarchical Representations",
    Lee et al (2009).

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    filter_shape : tuple or list of length 4
        shape of filter, as
        (hidden maps, visible maps, filter height, filter width)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    ext_bias : string in ('v', 'h', 'both'), optional
        extends bias by dimension 0 (e.g. for convolution in time) for
        hidden units ('h'), visible units ('v') or ('both')

    stride : tuple
        2D tuple of integer numbers, defining the stride of the convolution

    """
    def __init__(self, variables, name, input_shape, filter_shape, bias_h = 0,
                 params = None, ext_bias = 'v', stride = (1,1), **kwargs):
        RBMBase.__init__(self, variables, name, input_shape, **kwargs)
        CNN.__init__(self, variables, name, input_shape, filter_shape, bias_h,
                     params, ext_bias, stride, **kwargs)

class CondCRBM(CNN, RBMBase):
    """
    Conditional Convolutional RBM. Adds the result of a function onto the
    output *before* the non-linearity.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    filter_shape : tuple or list of length 4
        shape of filter, as
        (hidden maps, visible maps, filter height, filter width)

    bias_h : float, optional
        initial hidden bias (for cheap sparsity), default = 0

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    ext_bias : string in ('v', 'h', 'both'), optional
        extends bias by dimension 0 (e.g. for convolution in time) for
        hidden units ('h'), visible units ('v') or ('both')

    stride : tuple
        2D tuple of integer numbers, defining the stride of the convolution

    """
    def __init__(self, variables, name, input_shape, filter_shape, bias_h = 0,
                 params = None, ext_bias = 'v', cond_in_fun = None,
                 stride = (1,1), **kwargs):
        RBMBase.__init__(self, variables, name, input_shape, **kwargs)
        CNN.__init__(self, variables, name, input_shape, filter_shape, bias_h,
                     params, ext_bias, stride = stride, **kwargs)

        self.cond_in_fun = cond_in_fun

    def input_h(self, v_in, tempering=None):
        assert self.cond_in_fun is None, "Please pass a the parameter 'cond_in_fun' " + \
                    "to the make method."
        return CNN.input_h(self, v_in, tempering=tempering) + \
                self.cond_in_fun(name_layer = self.name, type_units = "h",
                                 input_sym = self.input, cost = self.cost,
                                 self_instance = self)

class RNN(FFBase):
    """
    Standard Recurrent Neural Network class

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    n_hidden : int
        number of hidden units

    n_out : int
        length of output vector

    act_fun_h : function
        activation function for output

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    """
    def __init__(self, variables, name, input_shape, n_hidden, n_out,
                 act_fun_out = None, params=None, **kwargs):

        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape,
                        hidden_shape = (n_hidden,),
                        **kwargs)

        if params is None:
            # parameters of the model
            self.Wx = theano.shared(name='Wx',
                                    value=np.random.uniform(-0.1, 0.1,
                                    list(input_shape) + [n_hidden]).astype(fx))
            self.Wh = theano.shared(name='Wh',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.W = theano.shared(name='W',
                                   value=np.random.uniform(-0.1, 0.1,
                                   (n_hidden, n_out)).astype(fx))
            self.bh = theano.shared(name='bh',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.bo = theano.shared(name='bo',
                                   value=np.zeros(n_out, dtype=fx))
            self.h0 = theano.shared(name='h0',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.params = [self.Wx, self.Wh, self.W, self.bh, self.bo, self.h0]
        else:
            self.params = params

        if act_fun_out == None:
            self.act_fun_out = self.act_fun_h
        else:
            self.act_fun_out = act_fun_out

        compile_own_f = partial(RNN.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)
        register_plot = partial(RNN.register_plotting_compiled, self)
        self.callback_add(register_plot, Notifier.REGISTER_PLOTTING)

    def recurrence(self, v_in, h_before):
        return [self.activation_h_t(v_in, h_before),
                self.output_t(v_in, h_before)]


    def output(self, v_in):
        [_, s], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, None],
                                n_steps=v_in.shape[0])
        return s

    def activation_h(self, v_in):
        [h, _], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, None],
                                n_steps=v_in.shape[0])
        return h

    def activation_h_t(self, v_in, h_before):
        return self.act_fun_h(self.input_h(v_in, h_before))

    def output_t(self, v_in, h_before):
        return self.act_fun_out(T.dot(self.activation_h_t(v_in, h_before),
                                      self.W) + self.bo)

    def input_h(self, v_in, h_before):
        return T.dot(v_in, self.Wx) + T.dot(h_before, self.Wh) + self.bh

    def activation_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    def input_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

    def register_plotting_compiled(self):
        if self.tiling_params:
            tiling = self.tiling_params
        else:
            tiling = "corpus" if self.convolutional else "default"

        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           tiling = tiling, forward = False)
        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           ptype = 'hist', forward = False)
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           tiling = tiling)
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           ptype = 'hist')

    def compile_functions(self):
        """ Compile theano functions """
        self.hact = theano.function(self.variables.values(),
                                    self.activation_h(self.input),
                                    allow_input_downcast = True,
                                    on_unused_input = 'ignore')

class RNN_Gated(FFBase):
    """
    Gated RNN, as proposed in 'Learning Phrase Representations using RNN
    Encoder-Decoder for Statistical Machine Translation', Cho et. al. 2014

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    n_hidden : int
        number of hidden units

    n_out : int
        length of output vector

    act_fun_h : function
        activation function for output

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    """
    def __init__(self, variables, name, input_shape, n_hidden, n_out,
                 act_fun_out = None, params=None, **kwargs):

        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape,
                        hidden_shape = (n_hidden,),
                        **kwargs)

        if params is None:
            # parameters of the model
            self.Wxh = theano.shared(name='Wxh',
                                    value=np.random.uniform(-0.1, 0.1,
                                    list(input_shape) + [n_hidden]).astype(fx))
            self.Wxr = theano.shared(name='Wxr',
                                   value=np.random.uniform(-0.1, 0.1,
                                   list(input_shape) + [n_hidden]).astype(fx))
            self.Wxu = theano.shared(name='Wxu',
                                   value=np.random.uniform(-0.1, 0.1,
                                   list(input_shape) + [n_hidden]).astype(fx))
            self.Whh = theano.shared(name='Whh',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.Why = theano.shared(name='Why',
                                   value=np.random.uniform(-0.1, 0.1,
                                   (n_hidden, n_out)).astype(fx))
            self.Whr = theano.shared(name='Whr',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.Whu = theano.shared(name='Whu',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.bh = theano.shared(name='bh',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.bo = theano.shared(name='bo',
                                   value=np.zeros(n_out, dtype=fx))
            self.h0 = theano.shared(name='h0',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.params = [self.Wxh, self.Wxr, self.Wxu, self.Whh, self.Why,
                           self.Whr, self.bh, self.bo, self.h0]
        else:
            self.params = params

        if act_fun_out == None:
            self.act_fun_out = self.act_fun_h
        else:
            self.act_fun_out = act_fun_out

        compile_own_f = partial(RNN_Gated.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)
        register_plot = partial(RNN_Gated.register_plotting_compiled, self)
        self.callback_add(register_plot, Notifier.REGISTER_PLOTTING)


    def recurrence_gen(self, v_in, h_before):
        o = self.output_t(v_in, h_before)
        sample = T.cast(T.gt(o, self.t_rng.uniform((64,),0,1)),fx)
        return [sample,
                self.activation_h_t(v_in, h_before)]

    def generate_(self, v0, n_steps):
        [s, _], _ = theano.scan(fn=self.recurrence_gen,
                                   outputs_info=[v0, self.h0],
                                   n_steps=n_steps)
        return s

    def recurrence(self, v_in, h_before):
        return [self.activation_h_t(v_in, h_before),
                self.output_t(v_in, h_before)]


    def output(self, v_in):
        [_, s], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, None],
                                n_steps=v_in.shape[0])
        return s



    def activation_h(self, v_in):
        [h, _], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, None],
                                n_steps=v_in.shape[0])
        return h

    def activation_h_t(self, v_in, h_before):
        return self.activation_h_t_candidate(v_in, h_before) * \
                    self.update_t(v_in, h_before) + \
                        h_before * (1 - self.update_t(v_in, h_before))

    def activation_h_t_candidate(self, v_in, h_before):
        return self.act_fun_h(self.input_h(v_in, h_before))

    def reset_t(self, v_in, h_before):
        return T.nnet.sigmoid(T.dot(v_in, self.Wxr) + T.dot(h_before, self.Whr))

    def update_t(self, v_in, h_before):
        return T.nnet.sigmoid(T.dot(v_in, self.Wxu) + T.dot(h_before, self.Whu))

    def output_t(self, v_in, h_before):
        return self.act_fun_out(T.dot(self.activation_h_t(v_in, h_before),
                                      self.Why) + self.bo)

    def input_h(self, v_in, h_before):
        return T.dot(v_in, self.Wxh) + \
                T.dot(h_before * self.reset_t(v_in, h_before), self.Whh) + self.bh

    def activation_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    def input_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

    def register_plotting_compiled(self):
        if self.tiling_params:
            tiling = self.tiling_params
        else:
            tiling = "corpus" if self.convolutional else "default"

        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           tiling = tiling, forward = False)
        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           ptype = 'hist', forward = False)
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           tiling = tiling)
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           ptype = 'hist')

    def compile_functions(self):
        """ Compile theano functions """
        self.hact = theano.function(self.variables.values(),
                                    self.activation_h(self.input),
                                    allow_input_downcast = True,
                                    on_unused_input = 'ignore')
        v0 = T.vector('v0', dtype = fx)
        n_steps = T.scalar('n_steps', dtype='int32')
        self.generate = theano.function([v0, n_steps],
                                        self.generate_(v0, n_steps),
                                        allow_input_downcast = True)

class LSTM(FFBase):
    """
    Long-short term memory, Hochreiter et. al. 1997

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    n_hidden : int
        number of hidden units

    n_out : int
        length of output vector

    act_fun_h : function
        activation function for output

    params : tuple, optional
        intial parameters, as (weights, bias_h, bias_v)

    """
    def __init__(self, variables, name, input_shape, n_hidden, n_out,
                 act_fun_out = None, params=None, **kwargs):

        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape,
                        hidden_shape = (n_hidden,),
                        **kwargs)
        if params is None:
            # parameters of the model
            self.Wxc = theano.shared(name='Wxc',
                                    value=np.random.uniform(-0.1, 0.1,
                                    list(input_shape) + [n_hidden]).astype(fx))
            self.Wxi = theano.shared(name='Wxi',
                                    value=np.random.uniform(-0.1, 0.1,
                                    list(input_shape) + [n_hidden]).astype(fx))
            self.Wxf = theano.shared(name='Wxf',
                                   value=np.random.uniform(-0.1, 0.1,
                                   list(input_shape) + [n_hidden]).astype(fx))
            self.Why = theano.shared(name='Why',
                                   value=np.random.uniform(-0.1, 0.1,
                                   (n_hidden, n_out)).astype(fx))
            self.Whmo = theano.shared(name='Whmo',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.Who = theano.shared(name='Who',
                                   value=np.random.uniform(-0.1, 0.1,
                                   (n_hidden, n_hidden)).astype(fx))
            self.Whc = theano.shared(name='Whc',
                                   value=np.random.uniform(-0.1, 0.1,
                                   (n_hidden, n_hidden)).astype(fx))
            self.Wxo = theano.shared(name='Wxo',
                                   value=np.random.uniform(-0.1, 0.1,
                                   list(input_shape) + [n_hidden]).astype(fx))
            self.Whi = theano.shared(name='Whi',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.Whf = theano.shared(name='Whf',
                                    value=np.random.uniform(-0.1, 0.1,
                                    (n_hidden, n_hidden)).astype(fx))
            self.bf = theano.shared(name='bf',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.bi = theano.shared(name='bi',
                                   value=np.zeros(n_hidden, dtype=fx))
            self.bc = theano.shared(name='bc',
                                   value=np.zeros(n_hidden, dtype=fx))
            self.bo = theano.shared(name='bo',
                                   value=np.zeros(n_hidden, dtype=fx))
            self.h0 = theano.shared(name='h0',
                                    value=np.zeros(n_hidden, dtype=fx))
            self.params = [self.Wxc, self.Wxi, self.Wxf, self.Why, self.Whmo,
                           self.Who, self.Wxo, self.Whi, self.Whf, self.Whc,
                           self.bf, self.bo, self.bi, self.bc, self.h0]
        else:
            self.params = params

        if act_fun_out == None:
            self.act_fun_out = self.act_fun_h
        else:
            self.act_fun_out = act_fun_out

        compile_own_f = partial(LSTM.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)
        register_plot = partial(LSTM.register_plotting_compiled, self)
        self.callback_add(register_plot, Notifier.REGISTER_PLOTTING)

    def recurrence(self, v_in, h_before, state_before):
        return [self.activation_h_t(v_in, h_before, state_before),
                self.state_t(v_in, h_before, state_before),
                self.output_t(v_in, h_before)]


    def output(self, v_in):
        [_, _, s], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, self.h0, None],
                                n_steps=v_in.shape[0])
        return s

    def recurrence_gen(self, v_in, h_before, state_before):
        o = self.output_t(v_in, h_before)
        sample = T.cast(T.gt(o, self.t_rng.uniform((64,),0,1)),fx)
        return [sample,
                self.activation_h_t(v_in, h_before, state_before),
                self.state_t(v_in, h_before, state_before)]

    def generate_(self, v0, n_steps):
        [s, _, _], _ = theano.scan(fn=self.recurrence_gen,
                                   outputs_info=[v0, self.h0, self.h0],
                                   n_steps=n_steps)
        return s

    def activation_h(self, v_in):
        [h, _, _], _ = theano.scan(fn=self.recurrence,
                                sequences=v_in,
                                outputs_info=[self.h0, self.h0, None],
                                n_steps=v_in.shape[0])
        return h

    def state_t(self, v_in, h_before, state_before):
        return self.state_candidate(v_in, h_before) * \
                    self.input_gate(v_in, h_before) + \
                        state_before * self.forget_gate(v_in, h_before)

    def state_candidate(self, v_in, h_before):
        return self.act_fun_h(self.input_h(v_in, h_before))

    def forget_gate(self, v_in, h_before):
        return T.nnet.sigmoid(T.dot(v_in, self.Wxf) + T.dot(h_before, self.Whf) + self.bf)

    def input_gate(self, v_in, h_before):
        return T.nnet.sigmoid(T.dot(v_in, self.Wxi) + T.dot(h_before, self.Whi) + self.bi)

    def out_gate(self, v_in, h_before, state_before):
        return T.nnet.sigmoid(T.dot(h_before, self.Who) + \
                                T.dot(v_in, self.Wxo) + \
                                T.dot(self.state_t(v_in, h_before, state_before), self.Whmo) \
                                + self.bo)

    def activation_h_t(self, v_in, h_before, state_before):
        return self.out_gate(v_in, h_before, state_before) * \
            self.act_fun_h(self.state_t(v_in, h_before, state_before))

    def output_t(self, v_in, h_before):
        return self.act_fun_out(T.dot(h_before, self.Why))

    def input_h(self, v_in, h_before):
        return T.dot(v_in, self.Wxc) + T.dot(h_before, self.Whc) + self.bc

    def activation_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    def input_v(self, h_in):
        LOGGER.error(NotImplementedError("Reconstruction from an RNN not implemented."))
        return theano.shared(-1, 'not_implemented')

    @property
    def params_value(self):
        return [p.get_value() for p in self.params]

    @params_value.setter
    def params_value(self, value):
        for i,v in enumerate(self.params):
            v.set_value(value[i])

    def register_plotting_compiled(self):
        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           tiling = 'corpus', forward = False)
        self.register_plot(lambda data : self.out(data['input']), 'out',
                           name_net = self.name,
                           ptype = 'hist', forward = False)
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           tiling = 'default')
        self.register_plot(lambda data : self.hact(*(data.values())), 'hact',
                           name_net = self.name,
                           ptype = 'hist')

    def compile_functions(self):
        """ Compile theano functions """
        self.hact = theano.function(self.variables.values(),
                                    self.activation_h(self.input),
                                    allow_input_downcast = True,
                                    on_unused_input = 'ignore')
        v0 = T.vector('v0', dtype = fx)
        n_steps = T.scalar('n_steps', dtype='int32')
        self.generate = theano.function([v0, n_steps],
                                        self.generate_(v0, n_steps),
                                        allow_input_downcast = True)

class ToDense(FFBase):
    '''
    Converts a 4D convolutional output to 2D by concatenating
    the convolutional maps.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)
    """
    '''
    def __init__(self, variables, name, input_shape, **kwargs):
        FFBase.__init__(self, variables, name = name, **kwargs)
        self.input_shape = input_shape
        self.hidden_shape = ([np.product(input_shape)])

    def activation_v(self, h_in):
        return h_in.reshape([h_in.shape[0]] + list(self.input_shape), ndim=4)

    def activation_h(self, v_in):
        return v_in.flatten(2)

    def output(self, v_in):
        return self.activation_h(v_in)

class ConvShaping(FFBase):
    """
    Improves processing speed (and plotting is better understandable) for
    4D convolutional output with dimensions [x,y,z,1]. Reshapes such an output
    to [x,1,z,y].

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)
    """
    def __init__(self, variables, name, input_shape, **kwargs):
        FFBase.__init__(self, variables, input_shape=input_shape,
                        name = name, **kwargs)
        assert input_shape[2] == 1, "Shaper: Works only for inputs, where shape[2] = 1."
        self.input_shape = input_shape
        self.hidden_shape = ([1, self.input_shape[1],
                             self.input_shape[0]])

    def activation_v(self, h_in):
        reshaped = h_in.reshape([h_in.shape[0], self.input_shape[1],
                                 self.input_shape[0],
                                 self.input_shape[2]], ndim=4)
        return reshaped.dimshuffle(0,2,1,3)

    def activation_h(self, v_in):
        shuffled = v_in.dimshuffle(0,2,1,3)
        return shuffled.reshape([v_in.shape[0], 1,
                                 self.input_shape[1],
                                 self.input_shape[0]], ndim=4)

    def output(self, v_in):
        return self.activation_h(v_in)

class NoiseBinom(FFBase):
    """
    Adds some binomial noise to the hidden activation

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    noise_level : float
        probability of a unit to be turned off
    """
    def __init__(self, variables, name, input_shape, noise_level = 0.5, **kwargs):
        FFBase.__init__(self, variables, input_shape=input_shape,
                        name = name, **kwargs)
        self.noise_level = noise_level

    def activation_h(self, v_in):
        return self.t_rng.binomial(size = v_in.shape, n = 1, p = 1 - self.noise_level, dtype = fx) * v_in

    def activation_v(self, h_in):
        return h_in

    def output(self, v_in):
        return self.activation_h(v_in)

class TransitionFunc(FFBase):
    """
    output = trans_func(input)

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    noise_level : float
        probability of a unit to be turned off
    """
    def __init__(self, variables, name, input_shape, trans_func = lambda x:x,
                 **kwargs):
        FFBase.__init__(self, variables, input_shape=input_shape,
                        name = name, **kwargs)
        self.trans_func = trans_func

    def activation_h(self, v_in):
        return self.trans_func(v_in)

    def activation_v(self, h_in):
        return h_in

    def output(self, v_in):
        return TransitionFunc.activation_h(self, v_in)


class ConvDShaping(ConvShaping):
    """
    Inverse of ConvShaping.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    """
    def __init__(self, variables, name, input_shape, **kwargs):
        FFBase.__init__(self, variables, input_shape = input_shape,
                        name = name, **kwargs)
        assert input_shape[0] == 1, "Deshaper: Works only for inputs, where shape[0] = 1."
        self.input_shape = input_shape
        self.hidden_shape = ([self.input_shape[0], self.input_shape[1], 1])

    def activation_v(self, h_in):
        return ConvShaping.activation_h(self, h_in)

    def activation_h(self, v_in):
        return ConvShaping.activation_v(self, v_in)

    def output(self, v_in):
        return self.activation_h(v_in)

class Normalizing(FFBase):
    """
    Normalizes the input between 0 and 1

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    """
    def __init__(self, variables, name, input_shape, **kwargs):
        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape, **kwargs)

    def activation_h(self, v_in):
        return v_in / T.max(v_in)

    def activation_v(self, h_in):
        LOGGER.warning("Inverse of normalization not yet implemented. "
                       "Returning dummy.")
        return h_in

    def output(self, v_in):
        return self.activation_h(v_in)

class MaxPoolerOverlapping(FFBase):
    """
    An overlapping max pooling layer for a 4D convolutional output.
    Simple overlapping max-pooling over 3rd dimension (e.g. time dimension)
    Can be used for creating a spatio-temporal space, where points occurring
    after one another in time are also close in space.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the net

    input_shape : tuple or list of length 4
        shape of input, as
        (batch size, visible maps, image height, image width)

    """
    def __init__(self, variables, name, input_shape, **kwargs):
        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape, **kwargs)

    def activation_h(self, v_in):
        return T.maximum(v_in, T.roll(v_in, shift = 1, axis = 2))

    def activation_v(self, h_in):
        LOGGER.warning("Inverse of overlapping pooling not yet implemented. "
                       "Returning dummy.")
        return h_in

    def output(self, v_in):
        return self.activation_h(v_in)


class MaxPooler(FFBase):
    """
    A max pooling layer for a 4D convolutional output.

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the layer

    input_shape : array-like
        input dimensions

    downsample : tuple of length 2
        Factor by which to downscale (vertical ds, horizontal ds).
        (2,2) will halve the image in each dimension.

    ignore_border : bool (default None, will print a warning and set to False)
        When True, (5,5) input with ds=(2,2) will generate a (2,2) output.
        (3,3) otherwise.

    use_input_inverse : bool
        when projecting downwards through layer, should bottom up input be used
        (improves result)
    """
    def __init__(self, variables, name, input_shape, downsample = (1,1),
                 ignore_border = False, use_input_inverse = True, **kwargs):
        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape, **kwargs)
        self.downsample = downsample
        self.ignore_border = ignore_border
        self.use_input_inv = use_input_inverse

    def activation_h(self, v_in):
        return max_pool_2d(v_in, ds = self.downsample,
                           ignore_border = self.ignore_border)

    def activation_v(self, h_in):
        if self.use_input_inv:
            v_in = self.input
        else:
            input_shape = self.input_shape
            v_in = theano.sandbox.cuda.basic_ops.gpu_alloc_empty(h_in.shape[0],
                                                                 *input_shape)


        fake_fwd = max_pool_2d(v_in, ds = self.downsample,
                               ignore_border = self.ignore_border)

        return theano.grad(None, wrt=v_in, known_grads={fake_fwd: h_in})

    def output(self, v_in):
        return self.activation_h(v_in)


class UpSampling(FFBase):
    """
    UpSamples a convolutional output
    Note: Smoothing currently only works for 1D convolution

    Parameters
    ----------

    variables : dictionary
        dictionary of symbolic inputs and outputs
        (of type theano.tensor.TensorVariable)

    name : string
        name of the layer

    input_shape : array-like
        input dimensions

    upsample : tuple of length 2
        Factor by which to upsample (vertical us, horizontal us).
        (2,2) will double the image in each dimension.

    smooth : boolean
        if output should be smoothed with a smoothing kernel

    """
    def __init__(self, variables, name, input_shape, upsample = (1,1),
                 smooth = True, **kwargs):
        assert theano.config.device.startswith('gpu'), "UpSampling needs cuDNN."
        self.upsample = upsample
        self.smooth = smooth

        if smooth:
            self.smooth_kern = range(upsample[0]) + range(upsample[0] - 1)[::-1]
            self.smooth_kern = np.reshape(self.smooth_kern, (1,1,-1,1))
            self.smooth_kern = np.cast[fx](self.smooth_kern)
            self.smooth_kern = self.smooth_kern / np.sum(self.smooth_kern)

        FFBase.__init__(self, variables, name = name,
                        input_shape = input_shape, **kwargs)

    def activation_h(self, v_in):
        result = T.repeat(v_in, self.upsample[0], axis = 2)
        result = T.repeat(result, self.upsample[1], axis = 3)

        if self.smooth:
            border = self.smooth_kern.shape[2] // 2
            result = conv2d(
                    result, self.smooth_kernW[:,:,::-1,::-1],
                    border_mode = (border,0),
                )
        return result.dimshuffle(0,3,2,1)

    def activation_v(self, h_in):
        result = h_in[:,:,::self.upsample[0],:]
        result = result[:,:,:,::self.upsample[1]]
        return result

    def output(self, v_in):
        return self.activation_h(v_in)



