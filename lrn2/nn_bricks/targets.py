'''
Created on Dec 18, 2015

@author: Stefan Lattner
'''

import theano
import numpy as np
from theano.sandbox.cuda.dnn import dnn_conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from lrn2.nn_bricks.utils import fx
from _functools import partial
from lrn2.nn_bricks.notifier import Notifier

"""
Generative targets, depending on internal states of a network
"""
class Paradigmatic(object):
    '''
    Paradigmatic target for convolutional networks
    '''
    def __init__(self, input_sym, size, input_shape, conv_shaper = True):
        self.size = size
        self.conv_shaper = conv_shaper
        Paradigmatic.setup_kerns(self, size, input_shape, conv_shaper)
        Paradigmatic.set_target(self, input_sym)
        # if e.g. layer gets part of a stack, self.input might have changed
        
    def set_target(self, v_in):
        if self.conv_shaper:
            b_mode = (self.size[0] // 2, self.input_shape[2] // 2)
        else:
            b_mode = (self.size[0] // 2, self.size[1] // 2)
        
        conv_out = dnn_conv(
            img = v_in,
            kerns = self.hist_kerns,
            conv_mode = 'cross',
            border_mode = b_mode,
        )
        self.target = gpu_contiguous(conv_out)
        
    def setup_kerns(self, size, input_shape, conv_shaper = True):
        if not conv_shaper:
            filters = np.zeros((input_shape[1], input_shape[1], 
                               size[0], size[1]), dtype = fx)
            for filt in range(input_shape[1]):
                filters[filt, filt, :, :] = 1
            self.hist_kerns = theano.shared(filters)
        else:
            filters = np.zeros((input_shape[1], input_shape[0], 
                               size[0], input_shape[1]), dtype = fx)
            for filt in range(input_shape[1]):
                filters[filt, :, :, filt] = 1
            self.hist_kerns = theano.shared(filters, name = "hist_kerns")
        