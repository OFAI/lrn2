'''
Created on Nov 24, 2015

@author: Stefan Lattner
'''
import logging
import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv
from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.optimize import Optimizer

LOGGER = logging.getLogger(__name__)

class NNFeatures(object):
    '''
    Calculates some features of internal NN states and input
    for GD approximation during generation.
    '''
    def __init__(self):
        NNFeatures.compile_functions(self)

    def chromagram_(self, img, y_mask = None):
        if y_mask:
            img = img[:,:,:,y_mask]
        width = 4
        MAJOR = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
        MINOR = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
        filt = np.array([MAJOR, MINOR]).astype(fx).reshape(2,1,1,-1)
        filt = np.repeat(filt, width, axis=2)
        img_filt = theano.shared(filt, 'filt_key_profiles')
        conv_out = dnn_conv(
            img = img,
            kerns = img_filt,
            conv_mode = 'cross',
            border_mode = 'full',
            subsample = (1,1)
        )
        key_prof = conv_out[:,:,:,:-T.mod(conv_out.shape[3], 12)]
        key_prof = T.reshape(key_prof, newshape=(conv_out.shape[0],
                                                 conv_out.shape[1],
                                                 conv_out.shape[2], -1, 12))
        key_prof = T.sum(key_prof, axis = 3)
        key_prof = key_prof - T.min(key_prof, axis = 3, keepdims = True)
        key_prof = key_prof / (T.max(key_prof, axis = 3, keepdims = True) + 1e-6)
        return key_prof

    def onsets_(self, solution, y_mask = None):
        if y_mask:
            solution = solution[:,:,:,y_mask]
        shape_pr = solution.shape[2:]
        # linearize
        solution = T.reshape(T.transpose(solution), newshape = (-1,), ndim=1)
        zero = theano.shared(np.cast['float32']([0,]))
        # On/Offsets
        solution = T.extra_ops.diff(solution)
        solution = T.concatenate((zero, solution))
        # Remove Offsets
        solution = T.switch(T.lt(solution, 0), 0, solution)
        # Shape to 2d piano roll
        solution = solution.reshape((shape_pr[1], -1), ndim = 2).transpose()
        # Sum down to t
        solution = T.sum(solution, axis=1)
        return solution

    def rhythm_(self, solution):
        """ Rhythm over a measure (auto-correlation of onsets)"""
        shape_pr = solution.shape[2:]
        solution = T.reshape(T.transpose(solution), newshape = (-1,), ndim=1)
        zero = theano.shared(np.cast['float32']([0,]))
        solution = T.extra_ops.diff(solution)
        solution = T.concatenate((zero, solution))
        solution = T.switch(T.lt(solution, 0), 0, solution)
        solution = solution.reshape((shape_pr[1], -1), ndim = 2).transpose()
        solution = T.sum(solution, axis=1).reshape((1, 1, 1, shape_pr[0]))
        corr = T.nnet.conv.conv2d(
                        input = solution,
                        filters = solution[:,:,:,::-1],
                        border_mode = 'full',
                        )
        corr = corr.reshape((-1,), ndim=1)
        corr = corr[corr.shape[0]/2+1:corr.shape[0]/2+16]
        corr -= T.mean(corr)
        corr /= T.std(corr)
        return corr

    def polyphony_(self, solution, y_mask = None):
        if y_mask:
            solution = solution[:,:,:,y_mask]
        solution = T.sum(solution, axis=3)
        return solution

    def sim_matrix_(self, img, img_filt, filt_size = (9,9), measure = 'dot'):
        measures = ['dot', 'cos']
        assert measure in measures, "measure has to be in {0}".format(measures)
        s = img_filt.shape
        img_filt = T.reshape(img_filt,
                             newshape = (-1, s[1], filt_size[0], filt_size[1]),
                             ndim = 4, name = 'img_filt')
#         img_filt = img_filt.dimshuffle(3,0,1,2)
        conv_out = dnn_conv(
            img = img,
            kerns = img_filt,
            conv_mode = 'cross',
            border_mode = (0,0), #(filt_size[0] // 2, filt_size[1] // 2),
        )
        if measure == 'dot':
            return conv_out
        elif measure == 'cos':
            filt_norm = T.sqrt(T.sum(T.sqr(img_filt), axis = (1,2,3), keepdims = False))
            img_sqr = T.sqr(img)
            filt_ones = T.ones((1, s[1], filt_size[0], filt_size[1]), dtype = fx)
            conv_out2 = dnn_conv(
                img = img_sqr,
                kerns = filt_ones,
                conv_mode = 'cross',
                border_mode = (0,0), #(filt_size[0] // 2, filt_size[1] // 2),
            )
            # TODO: GO on from here
            norm = T.sqrt(conv_out2) * filt_norm.dimshuffle('x', 0, 'x', 'x')
            return conv_out / norm

    def correlate_(self, a, b, filt_size = (8,8)):
        return T.sum(T.maximum(-1e8, self.sim_matrix_(b, a, filt_size)))

    def self_sim_matrix_(self, v_in, filt_size = (16,12), y_rng = None,
                         measure = 'dot'):
#         v_in = max_pool_2d(v_in, ds = (2,2), ignore_border = True)
        if y_rng:
            v_in = v_in[:,:,:,y_rng[0]:y_rng[1]]
        self_sim = self.sim_matrix_(v_in, v_in, filt_size, measure = measure)
        return self_sim

    def manhatten_corr(self, a, b):
        # [0,0,0,1,1,1,2,2,2]
        i = T.arange(a.shape[2]).repeat(a.shape[3])
        # [1,2,3,1,2,3,1,2,3]
        j = T.tile(T.arange(a.shape[3]), (a.shape[2],))

        manhatten, _ = theano.scan(lambda i,j : T.sum(T.abs_(T.roll(T.roll(a, shift = j, axis = 3),
                                                                    shift = i, axis = 2) - b)),
                                   sequences = [i,j])

        return T.sum(manhatten)

    def gram_matrix_(self, v_in):
        """ Calculates a gram matrix capturing the correlations between unit
        activations for each instance of the given batch (by using the dot prod)
        """
        if self.convolutional:
            act = max_pool_2d(self.activation_h(v_in), ds = (1,1))
            act = act.reshape((act.shape[0], act.shape[1], -1))
            results, _ = theano.scan(lambda x: T.dot(x, x.T), sequences = [act])
            return results
        else:
            LOGGER.error("gram matrix for non-convolutional layers not implemented.")


    def compile_functions(self):
        #self.gram_matrix = theano.function([self.input], self.gram_matrix_(self.input))
        filt_size = T.vector('filt_size', dtype = 'int32')
        self.self_sim_matrix = theano.function([self.variables['input'], filt_size],
                                               self.self_sim_matrix_(self.input, filt_size))
        self.chromagram = theano.function([self.variables['input']], self.chromagram_(self.input))
        self.rhythm = theano.function([self.variables['input']], self.rhythm_(self.input))
        self.polyphony = theano.function([self.variables['input']], self.polyphony_(self.input))


class DeepDreamer(object):
    def __init__(self, solution, lr):
        self.opt = None
        self.solution = solution
        self.lr = lr
        self.cost_ = None

    def dream(self, steps = 1):
        assert self.cost_ is not None, "cost function (attribute 'NNGenerative.cost') has to be set manually after initialization"
        cost = 0
        if self.opt is None:
            self.opt = self.get_optimizer(self.cost_)

        for epoch in range(steps):
            cost = self.opt.train()
            self.solution.set_value(np.maximum(self.solution.get_value(),0))
            self.solution.set_value(np.minimum(self.solution.get_value(),1))
            #print "cost =", cost

        return cost

    def get_optimizer(self, cost):
        return Optimizer(cost, params = [self.solution], variables = None,
                         data = None, batch_size = 1, lr = self.lr,
                         momentum = self.momentum, notifier = None)
