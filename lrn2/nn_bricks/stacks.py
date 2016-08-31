'''
Created on Dec 8, 2014

@author: Stefan Lattner
'''
import copy
import theano
import logging
import numpy as np
import theano.tensor as T

from _functools import partial
from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.plot import Plotter
from lrn2.nn_bricks.monitor import Monitor
from lrn2.nn_bricks.layers import FFBase, RBMBase
from lrn2.nn_bricks.notifier import Notifier, NotifierForwarder
from lrn2.nn_bricks.regularize import RegularizationCollector
from lrn2.nn_bricks.cost import CostSquaredError, CostKL,\
    CostCategoricCrossEntropy, CostCrossEntropy, CostReconErr,\
    CostReconErrDenoise, CostCrossEntropyAuto
from lrn2.nn_bricks.serialize import SerializeStack
from lrn2.nn_bricks.generate import NNFeatures, DeepDreamer
from collections import OrderedDict

LOGGER = logging.getLogger(__name__)

        
class ParamsBinder(object):
    """
    Enables a layer (or a stack) to use the parameters of the given 'external'
    layers in its training.
    """
    def __init__(self, layers):
        self.bind_params(layers)
        
    def bind_params(self, layers):
        if hasattr(self, 'layers_param') and self.layers_param == layers:
            LOGGER.warning("It seems that layers are param binded more than once: "
                           "{0}".format(layers))
        params_new = [p for l in layers if hasattr(l, 'params') for p in l.params]
        self.params = self.params + params_new
        self.layers_param = layers


class Binder(ParamsBinder, NotifierForwarder, RegularizationCollector):
    """
    Enables a layer (or a stack) to bind some other layers (or stacks) so
    that it incorporates the parameters and regularizations for training and
    the notification callbacks into its notifier (also used for plotting).
    Use this for sophisticated architectures
    (e.g. conditional, 'horizontal' connections to stack or back-connections,..)
    """
    def __init__(self, layers):
        ParamsBinder.__init__(self, layers)
        NotifierForwarder.__init__(self, layers)
        RegularizationCollector.__init__(self, layers)
        self.layers_bind = layers
        
    def bind(self, layers):
        self.layers = layers
        if layers == self.layers_bind:
            LOGGER.warning("It seems that layers are binded more than once: "
                           "{0}".format(layers))
        else:
            self.layers_bind = layers
        self.bind_params(layers)
        self.callback_forward(layers)
        self.collect_regulars(layers)

class LayerConnector(object):
    """
    Connects given layers so that the output of a layer equals the input
    of the next layer (in the order of the 'layers' list).
    In addition, it compiles the layer.out function for each layer in order
    to make all 'out' functions work with a single main input (instead of the
    direct input to a layer).
    
    Note: This method changes the .input variable in each layer to the output 
        of the preceding layer, and the .out function of single layers need the
        main input as the parameter.
    """
    def __init__(self, layers, input_sym):
        self.layers_connect = layers
        self.input_sym_conn = input_sym
        self.connect_sym(input_sym)
        
    def connect_layers(self, layers):
        self.layers_connect = layers
        self.connect_sym(self.input_sym_conn)
        
    def connect_sym(self, input_sym):
        for n, l in enumerate(self.layers_connect):
            l.input = input_sym
            self.callback_add(partial(LayerConnector.compile_functions, self, l, n), 
                              Notifier.COMPILE_FUNCTIONS)
            if isinstance(l, Plotter):
                self.callback_add(partial(LayerConnector.register_plotting, self, l),
                                  Notifier.REGISTER_PLOTTING)
            input_sym = l.output(input_sym)

    def compile_functions(self, layer, n):
        layer.out = self.compile_act_h_fun_range(0, n+1)
        
    def register_plotting(self, layer):
        def output_wo_target(data):
            d = copy.copy(data)
            try:
                d.pop("target")
            except KeyError:
                pass
            return layer.out(*(d.values()))
        
        layer.register_plot(lambda data : output_wo_target(data), 'out',
                            tiling = 'default', forward = True,
                            name_net = layer.name)
        layer.register_plot(lambda data : output_wo_target(data), 'out',
                            ptype = 'hist', forward = True, 
                            name_net = layer.name)   
                    
    def connect_out_h(self, v_in, layer_bottom = 0, layer_top = -1):
        layer_top = len(self.layers_connect) + layer_top + 1 if layer_top < 0 else layer_top
#         layer_top = len(self.layers_connect) if layer_top == -1 else layer_top

        if layer_bottom == layer_top:
            return v_in
        
        return self.connect_out_h(self.layers_connect[layer_bottom].activation_h(v_in),
                                 layer_bottom + 1, layer_top)

    def connect_out_v(self, h_in, layer_bottom = 0, layer_top = -1):
        layer_top = len(self.layers_connect) + layer_top + 1 if layer_top < 0 else layer_top

        if layer_bottom == layer_top:
            return h_in

        return self.connect_out_v(self.layers_connect[layer_top - 1].activation_v(h_in),
                                 layer_bottom, layer_top - 1)

    def compile_act_h_fun_range(self, layer_bottom = 0, layer_top = -1):
        variables = copy.copy(self.variables)
        try:
            variables.pop('target')
        except KeyError:
            pass

        input_bottom = self.layers_connect[layer_bottom].input
        return theano.function(variables.values(),
                               self.connect_out_h(input_bottom,
                                                  layer_bottom,
                                                  layer_top),
                               allow_input_downcast = True,
                               on_unused_input = 'ignore')

    def compile_act_v_fun_range(self, input_top, layer_bottom = 0, layer_top = -1):
        return theano.function([input_top],
                               self.connect_out_v(input_top,
                                                 layer_bottom,
                                                 layer_top),
                               allow_input_downcast = True,
                               on_unused_input = 'warn')
        
class NNStackLess(LayerConnector, FFBase):
    """
    A reduced stack for NN layers with one main input and one main output.
    If you want to train your stack, rather go with NNStack.

    Parameters
    ----------

    layers : list
        a list of NN layers from which the stack will be built.

    input_sym : theano TensorVariable
        symbolic input to the theano graph
        
    name : string
        the name of the stack

    target_sym : theano TensorVariable, optional
        symbolic output to the theano graph
        only necessary for feed-forward networks
        
    """
    def __init__(self, layers, input_sym, name, variables = None):
        
        if variables == None:
            variables = OrderedDict()
            for l in layers:
                variables.update(l.variables)
                
            variables['input'] = layers[0].variables['input']
            try:
                variables['target'] = layers[-1].variables['target']
            except KeyError:
                pass
        
        FFBase.__init__(self, variables, name,
                        input_shape = layers[0].input_shape,
                        hidden_shape = layers[-1].hidden_shape,
                        plot_dparams = False, plot_params = False)
        LayerConnector.__init__(self, layers, input_sym)
        self.layers = layers
        
    def output(self, v_in):
        return self.activation_h(v_in)
    
    def activation_h(self, v_in):
        return self.connect_out_h(v_in)
        
    def activation_v(self, h_in):
        return self.connect_out_v(h_in)
        
class NNStack(NNStackLess, Binder):
    """
    A stack of NN layers, where each layer has a main input and a main target,
    defined over layer.input and layer.target. By initializing an NNStack,
    the target of each layer gets connected to input of the next layer.

    Parameters
    ----------

    layers : list
        a list of NN layers from which the stack will be built.

    input_sym : theano TensorVariable
        symbolic input to the theano graph
        
    name : string
        the name of the stack

    """
    def __init__(self, layers, input_sym, name, variables = None, **kwargs):
        LOGGER.debug("Packing {0} layers in stack {1}.".format(len(layers), name))
        NNStackLess.__init__(self, layers, input_sym, name, variables)
        Binder.__init__(self, layers)
        
class RBMStack(NNStack, RBMBase):
    """
    A stack of RBM layers. As it derives from RBMBase, a stack can be used like
    a single RBM layer for evaluation (e.g. Gibbs sampling over all layers).

    Parameters
    ----------

    layers : list
        list of RBM layers, as returned by make.make_layer()
        
    input_sym : theano TensorVariable
        symbolic input to the stack
        
    name : string
        name of the stack 
    """
    def __init__(self, layers, input_sym, name):
        NNStack.__init__(self, layers, layers[0].input, name)
        RBMBase.__init__(self, layers[0].input, "RBM stack",
                         layers[0].input_shape,
                         layers[-1].hidden_shape)

        compile_own_f = partial(RBMStack.compile_functions, self)
        self.callback_add(compile_own_f, Notifier.COMPILE_FUNCTIONS)

    def free_energy_(self, v_in):
        # Simple approximation to free energy of stack, not verified!
        fe_res = self.layers[0].free_energy_(v_in)
        for l in self.layers[1:]:
            if hasattr(l, 'free_energy_'):
                fe_res += l.free_energy_(l.input)
        return fe_res

    def gibbs_step_(self, v_in):
        return self.gibbs_step_range_(v_in, 0, -1)

    def gibbs_step_range_(self, v_in, layer_bottom = 0, layer_top = -1):
        h_in = self.activation_h(v_in, layer_bottom, layer_top)
        h_bin = T.gt(h_in, self.t_rng.uniform((h_in.shape)))
        return self.activation_v(T.cast(h_bin, fx),
                                 layer_bottom, layer_top)

    def compile_gibbs_fun_range(self, layer_bottom = 0, layer_top = -1):
        curr_input = self.layers[layer_bottom].input
        return theano.function([curr_input],
                               self.gibbs_step_range_(curr_input, layer_bottom,
                                                      layer_top),
                               allow_input_downcast=True)

    def compile_functions(self):
        """ Compile theano functions """
        self.free_energy = theano.function([self.variables['input']], 
                                           self.free_energy_(self.input))

        self.gibbs_step = theano.function([self.variables['input']], 
                                          self.gibbs_step_(self.input),
                                          allow_input_downcast=True)



""" MULTI-LAYER CLASSES """

class FFNNSquaredError(Notifier, NNStack, CostSquaredError,
                       SerializeStack,
                       Monitor, Plotter):
    """ *Stack:* Feed forward neural network with squared error cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name, **kwargs)
        CostSquaredError.__init__(self, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)  
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)      

class FFNNKullbackLeibler(Notifier, NNStack, CostKL,
                          SerializeStack,
                          Monitor, Plotter):
    """ *Stack:* Feed forward neural network with Kullback - Leibler cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name, **kwargs)
        CostKL.__init__(self, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class FFNNCatCrossEntropy(Notifier, NNStack, CostCategoricCrossEntropy,
                          SerializeStack, Monitor, Plotter):
    """ *Stack:* Feed forward neural network with categorical cross entropy cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name, **kwargs)
        CostCategoricCrossEntropy.__init__(self, self.activation_h(self.input), 
                                           self.target, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class FFNNCrossEntropy(Notifier, NNStack, 
                       CostCrossEntropy,
                       SerializeStack, Monitor, Plotter):
    """ *Stack:* Feed forward neural network with cross entropy cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name)
        CostCrossEntropy.__init__(self, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class FFNNAutoEncoderCrossEntropy(Notifier, NNStack, CostReconErr,
                              SerializeStack, Monitor, Plotter):
    """ *Stack:* AutoEncoder with cross entropy cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name, **kwargs)
        CostReconErr.__init__(self, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class FFNNAutoEncoderDenoiseReconErr(Notifier, NNStack, CostReconErrDenoise,
                                     SerializeStack,
                                     Monitor, Plotter):
    """ *Stack:* Denoising AutoEncoder with reconstruction error cost """
    def __init__(self, layers, name, noise_level = 0.5, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name, **kwargs)
        CostReconErrDenoise.__init__(self, noise_level, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class NNGenerative(Notifier, NNStackLess, CostCrossEntropyAuto,
                   NNFeatures,
                   ParamsBinder,
                   NotifierForwarder,
                   SerializeStack,
                   Monitor, Plotter, DeepDreamer):
    """ 
    Generative NN - optimizes the input of an NN given a cost function. 
    
    Assign the cost function after initialization with instance.cost = <theano cost function>
    
    Can be used e.g. to find an input which maximizes a certain neuron.
    
    Parameters
    ----------
    
    layers : list
        List of NNLayers as returned from make_net(.)
        
    name : string
        Name of the net
        
    lr : float
        Learning rate for the optimization
        
    momentum : float
        momentum of optimization
        
    n_samples : int
        Number of samples to optimize
        
    """
    def __init__(self, layers, name, lr, momentum = 0.0, n_samples = 1, **kwargs):
        Notifier.__init__(self)
        self.momentum = momentum
        self.sol_shape = list(layers[0].input_shape)
        self.sol_shape[0] = n_samples
        self.solution = theano.shared(np.random.uniform(0, 1,
                                      size = self.sol_shape).astype(fx),
                                      name="solution")
        self.reset_solution()
        NNStackLess.__init__(self, layers, layers[0].input, name, **kwargs)
        DeepDreamer.__init__(self, self.solution, lr, **kwargs)
        NNFeatures.__init__(self)
        ParamsBinder.__init__(self, layers)
        NotifierForwarder.__init__(self, layers)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

    def reset_solution(self):
        self.solution.set_value(np.random.uniform(0, 1,
                                size = self.sol_shape).astype(fx))

    @property
    def cost(self):
        return self.cost_

    @cost.setter
    def cost(self, value):
        self.cost_ = value
        self.opt = self.get_optimizer(value)
