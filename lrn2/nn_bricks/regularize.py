'''
Created on Feb 27, 2015

@author: Stefan Lattner
'''
import theano
import logging
import numpy as np
import theano.tensor as T

from lrn2.nn_bricks.utils import fx, find_equi_branch
from lrn2.nn_bricks.notifier import Notifier
from _functools import partial
# from lrn2.nn_bricks.stacks import NNStack

LOGGER = logging.getLogger(__name__)

class MaxNormRegular(object):
    """
    Max-Norm regularization

    Apply to single layers only, but effects single layers in a stack, too.
    """
    def __init__(self, max_norm, **kwargs):
        self.max_norm = np.cast[fx](max_norm)

        updates = {self.W: self.norm_constraint(self.W, self.max_norm)}

        clip_weights = theano.function([], None,
                                       updates = updates)
        # Clip weights after each batch
        self.callback_add(clip_weights, Notifier.BATCH_FINISHED, forward = True)
        
        MaxNormRegular.compile_functions(self)
        
        self.register_plot(lambda *args : self.get_W_norms(), "W_norm", name_net = self.name,
                           tiling = 'corpus', ptype = 'hist')
        

    def norm_constraint(self, tensor_var, max_norm, norm_axes=None,
                        epsilon=1e-7):
        dtype = np.dtype(fx).type
        norms = self.get_W_norms_(tensor_var, norm_axes)
        target_norms = T.clip(norms, 0, dtype(max_norm))
        constrained_output = \
            (tensor_var * (target_norms / (dtype(epsilon) + norms)))

        return constrained_output
    
    def get_W_norms_(self, tensor_var, norm_axes = None):
        # Code taken from the LASAGNE framework
        ndim = tensor_var.ndim

        if norm_axes is not None:
            sum_over = tuple(norm_axes)
        elif ndim == 2:  # DenseLayer
            sum_over = (0,)
        elif ndim in [3, 4, 5]:  # Conv{1,2,3}DLayer
            sum_over = tuple(range(1, ndim))
        else:
            raise ValueError(
                "Unsupported tensor dimensionality {}."
                "Must specify `norm_axes`".format(ndim)
            )

        return T.sqrt(T.sum(T.sqr(tensor_var), axis=sum_over, keepdims=True))
        
        
    def compile_functions(self):
        self.get_W_norms = theano.function([], self.get_W_norms_(self.W),
                                           allow_input_downcast = True)
        

class ActivationCrop(object):
    """
    Shuffles the weights of Neurons which have a mean activation higher than a
    specified threshold (based on the first batch only!)
    """
    def __init__(self, activation_crop = 1.0, **kwargs):
        self.activation_crop_thresh = activation_crop
        
        self.W_rand = theano.shared(np.zeros_like(self.W.get_value(), dtype=fx))
        self.bh_zero = theano.shared(np.zeros_like(self.bh.get_value(), dtype=fx))
        self.init_rand_W()

        updates = [(self.W, self.activation_crop()[0])]

        shuffle_weights = theano.function([self.variables['input']], None,
                                 updates = updates,
                                 on_unused_input='warn',
                                 allow_input_downcast = True)
        
        shuffle_w_data = lambda **kwargs: shuffle_weights(self.notify(Notifier.GET_DATA, 0)[0])
        # Initialize new random weights after each epoch
        self.callback_add(self.init_rand_W, Notifier.EPOCH_FINISHED, forward = False)
        # Shuffle too active neurons' weights after each epoch
        self.callback_add(shuffle_w_data, Notifier.EPOCH_FINISHED, forward = False)

    def init_rand_W(self, *args, **kwargs):
        self.W_rand.set_value(self.get_W_init())

    def activation_crop(self):
        ndim = self.activation_h(self.input).ndim
        if ndim == 4:
            mean_act = T.mean(self.activation_h(self.input), axis = (0,2,3))
            too_active = T.gt(mean_act, self.activation_crop_thresh).dimshuffle(0, 'x', 'x', 'x')
        elif ndim == 2:
            mean_act = T.mean(self.activation_h(self.input), axis = 0)
            too_active = T.gt(mean_act, self.activation_crop_thresh).dimshuffle(0, 'x')
            
        update_W = self.W_rand * too_active + self.W * (1 - too_active)
        #update_bh = self.bh_zero * too_active + self.bh * (1 - too_active)
        return update_W, #update_bh

class SparsityKL(object):
    """
    Sparsity based on Kullback-Leibler divergence
    """
    def __init__(self, sparsity_kl = 0.5, sparsity_kl_strength = 0.1, **kwargs):
        avg_act = T.mean(self.activation_h(self.input), axis=(0,2,3))
        cost = self.cost
        #self.cost = lambda *args : cost(*args) + T.sum((avg_act - sparsity)**2)
        self.cost = lambda *args : cost(*args) + sparsity_kl_strength * \
                                T.sum(self.bern_bern_kl(avg_act, sparsity_kl))

    def bern_bern_kl(self, X, Y):
        """
        Return the Kullback-Leibler divergence between Bernoulli variables
        represented by their sufficient statistics.
        
        Parameters
        ----------
        X : Theano variable
            An array of arbitrary shape where each element represents
            the statistic of a Bernoulli variable and thus should lie in
            ``(0, 1)``.
        Y : Theano variable
            An array of the same shape as ``target`` where each element represents
            the statistic of a Bernoulli variable and thus should lie in
            ``(0, 1)``.
        
        Returns
        -------
         res : Theano variable
            An array of the same size as ``target`` and ``prediction`` representing
            the pairwise divergences.
        """
        return X * T.log(X / Y) + (1 - X) * T.log((1 - X) / (1 - Y))

class Distributer(object):
    """
    A regularization class should derive from this class in order to enable
    a stack to collect the regularization cost.
    """
    def __init__(self, reg_fun, **kwargs):
        if hasattr(self, 'reg_funs'):
            self.reg_funs.append(reg_fun)
        else:
            self.reg_funs = [reg_fun]

class RegularizationCollector(object):
    """
    For stacks and complex architectures. 
    Enables a stack or a central layer (which will be trained) to collect all
    regularization terms of other layers.
    """
    def __init__(self, layers, **kwargs):
        self.layers_reg = layers
        self.collect_regulars(layers)

    def collect_regulars(self, layers):
        reg_terms = 0
        for l in layers:
            if hasattr(l, 'reg_funs'):
                for f in l.reg_funs:
                    reg_terms += f()
        cost = self.cost
        self.cost = lambda *args : cost(*args) + reg_terms

class NonNegative(Distributer):
    """
    Imposes non-negative weights
    """
    def __init__(self, non_negativity = 1, **kwargs):
        Distributer.__init__(self, self.non_negative)
        self.non_negativity = non_negativity
        try:
            cost = self.cost
            self.cost = lambda *args : cost(*args) + self.non_negative()
        except:
            pass

    def non_negative(self):
        return self.non_negativity * T.sum(T.minimum(0, self.W)**2)
                    #+ self.non_negativity * T.sum(T.minimum(0, 1-self.W)**2)
    
class VarianceFixer(Distributer):
    """
    Fixed variance of hidden unit activations in stack prevents vanishing
    gradients
    """
    def __init__(self, trg_variance = 4., variance_fix_strength = 1000.0, 
                 axis = None, **kwargs):
        Distributer.__init__(self, self.variance_fix)
        self.variance_fix_strength = variance_fix_strength
        self.trg_variance = trg_variance
        self.axis = axis
        cost = self.cost
        self.cost = lambda *args : cost(*args) + self.variance_fix()

    def variance_fix(self):
        return self.variance_fix_strength * T.mean((T.var(self.activation_h(self.input), axis = self.axis) - self.trg_variance) ** 2)
           
        
class Smoother(Distributer):
    """
    For convolution in time: smoothes hidden unit activations in time 
    dimension (axis 2)
    """
    def __init__(self, smooth = 1., **kwargs):
        self.int_smooth = smooth

        cost = self.cost
        self.cost = lambda *args : cost(*args) + self.reg_smoother()

    def reg_smoother(self):
        cost_smooth = T.sum((self.activation_h(self.input) - T.roll(self.activation_h(self.input), shift = 1, axis = 2))**2)
        return cost_smooth * self.int_smooth

class SparsitySum(Distributer):
    """
    Pushes small hidden activations towards 0
    """
    def __init__(self, sparsity_sum = 0, **kwargs):
        self.sparsity_sum = sparsity_sum
        cost = self.cost
        self.cost = lambda *args : cost(*args) + self.get_sum_reg()

    def get_sum_reg(self):
        return T.sum(self.activation_h(self.input)) * self.sparsity_sum

class SparsityLeeConv(Distributer):
    """
    Sparsity implementation as described by Lee et al (2007).

    For stacks, add class but note that stack params will be ignored
    the params of single layers are taken into account.
    """
    def __init__(self, sparsity, sparsity_strength = 0, narrow=0., power = 2, **kwargs):
        Distributer.__init__(self, self.get_ghbias_reg)
        self.sparsity = sparsity
        self.int_lee = sparsity_strength
        self.narrow = narrow
        self.power = power
        cost = self.cost

        self.cost = lambda *args : cost(*args) + self.get_ghbias_reg()


    def reg_sparsity(self):
        """ Determine regularisation term for sparsity """
        cost_sparsity = T.sum((T.mean(self.activation_h(self.input), axis=(0,2,3)) -
                                self.sparsity)**self.power) #* self.n_hidden
        return cost_sparsity

    def reg_selectivity(self):
        """ Determine regularisation term for selectivity """
        cost_selectivity = T.sum((T.mean(self.activation_h(self.input), axis=(0,1,3)) -
                                   self.sparsity)**self.power) #* self.n_hidden
        return cost_selectivity

    def reg_narrow(self):
        """ Determine regularisation term for 'narrowness' """
        cost_narrow = T.sum((self.activation_h(self.input) - self.sparsity)**2)**0.5
        return cost_narrow

    def get_ghbias_reg(self):
        """ Calculate regularisation component of hbias gradients """
        reg_sp = (self.int_lee * self.reg_sparsity() +
                  self.int_lee * self.reg_selectivity() +
                  self.narrow * self.reg_narrow())
        return reg_sp

class SparsityLee(Distributer):
    """
    Sparsity implementation as described by Lee et al (2007).

    For stacks, add class but note that stack params will be ignored
    the params of single layers are taken into account.
    
    Parameters
    ----------
    
    sparsity : float. optional
        target sparsity
    
    sparsity_strength : float, optional
        sparsity strength
        
    narrow : float, optional
        similar to L1 weight regularization - pushes activations close to
        target sparsity even closer.
        
    activations : array-like, optional
        array with theano graphs whose result equals an activation which will
        get regularized. If not set, component should have method 'activation_h'.
        
    strengths : array-like, optional
        if more than one activation should be regularized, different strengths
        can be passed here.
    """
    def __init__(self, sparsity, sparsity_strength = 0, narrow=0.,
                 activations = None, strengths = None, **kwargs):
        Distributer.__init__(self, self.get_ghbias_reg)
        
        self.sparsity = sparsity
        
        if strengths == None:
            self.int_lee = [sparsity_strength,]
        else:
            self.int_lee = strengths
            
        self.narrow = narrow
        self.activations = activations
        cost = self.cost
        
        self.cost = lambda *args : cost(*args) + self.get_ghbias_reg()

    def ensure_activation_set(self):
        if self.activations == None:
            message = "Module 'activation_h', or pass other graphs as 'activations'."
            assert hasattr(self, 'activation_h'), message
            self.activations = [self.activation_h(self.input)]
        
        if len(self.int_lee) == 1 and len(self.activations) > 1:
            self.int_lee *= len(self.activations)
             
        message2 = "number of activations and sparsity values must match."
        assert len(self.activations) == len(self.int_lee), message2

    def reg_sparsity(self):
        """ Determine regularisation term for sparsity """
        self.ensure_activation_set()
        regs = 0
        for act, intense in zip(self.activations, self.int_lee):
            regs += T.sum((T.mean(act, axis=1) - self.sparsity)**2) * intense
        return regs

            
    def reg_selectivity(self):
        """ Determine regularisation term for selectivity """
        self.ensure_activation_set()
        regs = 0
        for act, intense in zip(self.activations, self.int_lee):
            regs += T.sum((T.mean(act, axis=0) - self.sparsity)**2) * intense
        return regs

    def reg_narrow(self):
        """ Determine regularisation term for variance around sparsity val """
        regs = 0
        for act in self.activations:
            regs += T.sum((act - self.sparsity))
        return regs

    def get_ghbias_reg(self):
        """ Calculate regularisation component of hbias gradients """
        reg_sp = (self.reg_sparsity() + self.reg_selectivity() +
                  self.narrow * self.reg_narrow())
        return reg_sp

class WeightRegular(Distributer):
    """
    L1 and L2 weight regularization. Can be applied to any layer in order to
    regularize its shared variable 'W'. Any other set of shared variables can
    be regularized by passing them in a list to the constructor (wl_targets).
    
    Parameters
    ----------
    
    wl1 : float. optional
        L1 weight regularization strength
    
    wl2 : float, optional
        L2 weight regularization strength
        
    offset : float, optional
        shifts regularization center from 0 to another value
        
    wl_targets : array-like, optional
        array with theano shared variables (parameters) which will be
        regularized. If not set, component should have attribute 'W'
    """
    def __init__(self, wl1=0., wl2=0., offset = 0, wl_targets = None, **kwargs):
        Distributer.__init__(self, reg_fun = partial(WeightRegular.cost_reg, self))
        self.wl1 = np.cast[fx](wl1)
        self.wl2 = np.cast[fx](wl2)
        self.offset = offset
        self.wl_targets = wl_targets

        if wl1 > 0 or wl2 > 0:
            cost = self.cost
            self.cost = lambda *args : cost(*args) + self.cost_reg()

    def cost_reg(self):
        """ Calculate regularisation component of weight gradients """
        if self.wl_targets == None:
            message = "Module needs attrib 'W' or pass other as 'wl_targets'."
            assert hasattr(self, 'W'), message
            self.wl_targets = [self.W]
            
        regs = 0
        for trg in self.wl_targets:
            regs += self.wl1 * T.sum(T.abs_(trg - self.offset))
        for trg in self.wl_targets:
            regs += self.wl2 * T.sum(T.abs_(trg - self.offset) ** 2)
        return regs

class WeightRegularRNN(Distributer):
    """
    L1 and L2 weight regularization for RNNs with weights [W, Wx, Wh].
    *Deprecated*: Use class 'WeightRegular' and pass weights via wl_targets.
    """
    def __init__(self, wl1=0., wl2=0., **kwargs):
        Distributer.__init__(self, reg_fun = partial(WeightRegularRNN.cost_reg, self))
        self.wl1 = np.cast[fx](wl1)
        self.wl2 = np.cast[fx](wl2)

        if wl1 > 0 or wl2 > 0:
            cost = self.cost
            self.cost = lambda *args : cost(*args) + self.cost_reg()

    def cost_reg(self):
        """ Calculate regularisation component of weight gradients """
        reg_l1_weights = self.wl1 * T.sum(T.abs_(self.W))
        reg_l2_weights = self.wl2 * T.sum(T.abs_(self.W) ** 2)

        reg_l1_weights_x = self.wl1 * T.sum(T.abs_(self.Wx))
        reg_l2_weights_x = self.wl2 * T.sum(T.abs_(self.Wx) ** 2)

        reg_l1_weights_h = self.wl1 * T.sum(T.abs_(self.Wh))
        reg_l2_weights_h = self.wl2 * T.sum(T.abs_(self.Wh) ** 2)

        return reg_l1_weights + reg_l2_weights + reg_l1_weights_x + \
                reg_l2_weights_x + reg_l1_weights_h + reg_l2_weights_h

class WeightDiversity(Distributer):
    """
    Assumes Gauss-Distributed weights and tries to maximize the
    KL-Divergence between all weights (experimental!) 
    """
    def __init__(self, intensity = 1, **kwargs):
        Distributer.__init__(self, reg_fun = partial(WeightDiversity.cost_diversity, self))
        self.diversity_strength = intensity
        cost = self.cost
        self.cost = lambda *args : cost(*args) + self.cost_diversity()

    def cost_diversity(self):
        std = T.std(self.W, axis=1)
        mean = T.mean(self.W, axis=1, dtype=fx)

        results, _ = theano.scan(lambda s1,m1,s2,m2 :
                                 T.log(s2/s1) + ((s1**2+(m1-m2)**2) / (2*(s2**2))),
                                 sequences=[std, mean], non_sequences=[std, mean])
        return -1.0 * T.mean(results) * self.diversity_strength



class EntropyRegular(Distributer):
    """ Forces hidden units to be either 0 or 1 (low entropy) """
    def __init__(self, entropy_reg = 0.0, **kwargs):
        self.int_entropy = entropy_reg
        cost = self.cost
        self.cost = lambda *args : cost(*args) + self.get_entropy_reg()

    def get_entropy_reg(self):
        epsilon = 1e-7
        p = self.activation_h(self.input)
        p = T.switch(T.eq(p, 0), epsilon, p)
        p = T.switch(T.eq(p, 1), 1-epsilon, p)
        entropy = -p*T.log2(p)-(1-p)*T.log2(1-p)
        return T.mean(entropy)


class SparsityGOH(object):
    """
    Sparsity regularization as introduced by Goh et al (2010).

    Apply to single RBM layers only, does not effect single layers in a stack
    """
    def __init__(self, mu = 0.05, phi = 0.6, **kwargs):
        self.interp = 0.5
        self.mu = mu
        self.phi = phi
        # Add branch after gradient is calculated (as sorting is not
        # differentiable)
        self.callback_add(self.callback_grad, Notifier.GRADIENT_CALCULATED,
                          forward = True)

    def bias_h(self, v_in):
        """
        Calculate latent activation biases, combined for sparsity and
        selectivity.
        """
        h_act = self.activation_h(v_in)

        rank_0 = ((h_act.argsort(axis=0)
                   ).argsort(axis=0).astype(fx) + 1.
                  ) / T.shape(h_act)[0].astype(fx)

        rank_1 = ((h_act.argsort(axis=1)
                   ).argsort(axis=1).astype(fx) + 1.
                  ) / T.shape(h_act)[1].astype(fx)

        # Interpolate towards the average of the sparsity and selectivity bias
        # matrices.
        lat_act = (1. - self.interp) * (rank_0 ** ((1. / self.mu) - 1.)) \
                   + self.interp * (rank_1 ** ((1. / self.mu) - 1.))

        # inverse of sigmoid
        lat_act_logit = T.log(lat_act) - T.log(1. - lat_act)

        return lat_act_logit

    def input_h_goh(self, v_in):
        # Blending of actual input_h and GOH target activations
        return self.input_h(v_in) * (1 - self.phi) + \
                                             self.bias_h(v_in) * self.phi

    def callback_grad(self, updates):
        # Callback function for the Optimizer.
        # Allows to change the update list *after* gradient calculation.
        # Here, we replace the hidden unit input with a biased input (input_h_goh).
        result = []
        for update in updates:
            node = find_equi_branch(update[1], self.input_h(self.input))
            if node is not None:
                # ensure that the broadcastpattern at the root of subtrees
                # to be substituted are identical
                branch_goh = T.patternbroadcast(self.input_h_goh(self.input),
                                                node.broadcastable)
                replaced = theano.clone(update[1], replace={node: branch_goh})
                result.append((update[0], replaced))
            else:
                LOGGER.warning("Could not substitute theano branch with "
                               "sparsity regularizer. GOH sparsity regularization "
                               "will not be available.")

        return result

class SparsityGOHConv(SparsityGOH):
    """
    Sparsity regularization as introduced by Goh et al (2010) for convolutional
    networks.

    Apply to single CRBM layers only, does not effect single layers in a stack
    """
    def __init__(self, mu = 0.05, phi = 0.6, **kwargs):
        super(SparsityGOHConv, self).__init__(mu, phi)

    def bias_h(self, v_in):
        """
        Calculate latent activation biases, combined for sparsity and
        selectivity.
        """
        h_act = self.activation_h(v_in)

        h_act = h_act.dimshuffle(1,0,2,3)
        shape_before = h_act.shape
        h_act = h_act.reshape((h_act.shape[0], -1))

        rank_0 = ((h_act.argsort(axis=0)
                   ).argsort(axis=0).astype(fx) + 1.
                  ) / T.shape(h_act)[0].astype(fx)

        rank_1 = ((h_act.argsort(axis=1)
                   ).argsort(axis=1).astype(fx) + 1.
                  ) / T.shape(h_act)[1].astype(fx)

        # Interpolate towards the average of the sparsity and selectivity bias
        # matrices.
        lat_act = (1. - self.interp) * (rank_0 ** ((1. / self.mu) - 1.)) \
                   + self.interp * (rank_1 ** ((1. / self.mu) - 1.))

        lat_act = lat_act.reshape(shape_before)
        lat_act = lat_act.dimshuffle(1,0,2,3)

        # inverse of sigmoid
        lat_act_logit = T.log(lat_act) - T.log(1. - lat_act)

        return lat_act_logit
