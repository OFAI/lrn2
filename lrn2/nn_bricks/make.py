'''
Created on Feb 20, 2015

@author: Stefan Lattner
'''

import logging
import numpy as np
import theano.tensor as T
from theano.tensor.nnet import nnet

from lrn2.util.utils import ndims
from lrn2.nn_bricks.utils import fx
from lrn2.nn_bricks.plot import Plotter
from lrn2.nn_bricks.monitor import Monitor
from lrn2.nn_bricks.approximate import Approximator
from lrn2.nn_bricks.generate import NNFeatures
from lrn2.nn_bricks.notifier import Notifier
from lrn2.nn_bricks.serialize import SerializeLayer
from lrn2.nn_bricks.cost import CostCD, CostPCD, CostReconErr, CostCrossEntropy,\
    CostSquaredError, CostCategoricCrossEntropy, CostCategoricCrossEntropyAuto,\
    CostLogLikelihoodBinomial
from lrn2.nn_bricks.layers import ToDense, MaxPooler, MaxPoolerOverlapping, CRBM,\
    RBM, RNN, CNN, NN_BN, CNN_BN, DCNN, TransitionFunc, ConvShaping,\
    ConvDShaping, Normalizing, UpSampling, NNAuto, RNN_Gated, LSTM
from lrn2.nn_bricks.units import UnitsCRBMSigmoid, UnitsDropOut, UnitsCRBMGauss,\
    UnitsNNLinear, UnitsNNSigmoid, UnitsRBMSigmoid, UnitsRBMGauss, UnitsNNReLU,\
    UnitsNNSoftmax, UnitsRBMReLU, UnitsCNNReLU, UnitsCNNSigmoid, UnitsCNNLinear,\
    UnitsNNTanh
from lrn2.nn_bricks.regularize import SparsityGOH, WeightRegular,\
    MaxNormRegular, SparsityLee, SparsityLeeConv, NonNegative, SparsitySum,\
    ActivationCrop, WeightRegularRNN

LOGGER = logging.getLogger(__name__)


""" SINGLE-LAYER CLASSES """

class RBMConvPCD(Notifier, 
                UnitsCRBMSigmoid, 
                SparsityLeeConv,
                #SparsityGOHConv,
                NNFeatures,
                CRBM, CostPCD,
                WeightRegular,
                ActivationCrop,
                MaxNormRegular,
                Approximator,
                Monitor, SerializeLayer, Plotter):
    """ Convolutional Restricted Boltzmann Machine Layer with PCD cost """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_conv(kwargs['name'])
        Notifier.__init__(self)
        UnitsCRBMSigmoid.__init__(self)
        CRBM.__init__(self, **kwargs)
        CostPCD.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        ActivationCrop.__init__(self, **kwargs)
        Approximator.__init__(self, **kwargs)
        Monitor.__init__(self)
        SerializeLayer.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class RBMConvCD(Notifier, UnitsCRBMSigmoid, UnitsDropOut, CRBM, CostCD,
                WeightRegular, MaxNormRegular, SerializeLayer, Monitor, Plotter):
    """ Convolutional Restricted Boltzmann Machine Layer with CD cost """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_conv(kwargs['name'])
        Notifier.__init__(self)
        UnitsCRBMSigmoid.__init__(self)
        UnitsDropOut.__init__(self)
        CRBM.__init__(self, **kwargs)
        CostCD.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class RBMConvGauss(Notifier, UnitsCRBMGauss, CRBM, CostCD, WeightRegular, 
                   SerializeLayer, Monitor, Plotter):
    """ Convolutional RBM Layer with CD cost and gaussian units"""
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_conv(kwargs['name'])
        Notifier.__init__(self)
        UnitsCRBMGauss.__init__(self)
        CRBM.__init__(self, **kwargs)
        CostCD.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class RBMPCD(Notifier, UnitsRBMSigmoid, UnitsDropOut, RBM, CostPCD, 
                WeightRegular, SerializeLayer, Monitor, Plotter):
    """ RBM layer with sigmoid units and PCD cost """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsRBMSigmoid.__init__(self)
        UnitsDropOut.__init__(self, **kwargs)
        RBM.__init__(self, **kwargs)
        CostPCD.__init__(self, **kwargs)
        WeightRegular.__init__(self,  **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class RBMCDGOH(Notifier, UnitsRBMSigmoid, SparsityGOH, RBM, CostCD, 
                  WeightRegular,
                  SerializeLayer, Monitor, Plotter):
    """ RBM layer with sigmoid units, CD cost and GOH sparsity regularization """
    def __init__(self, **kwargs):                             
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsRBMSigmoid.__init__(self)
        SparsityGOH.__init__(self,  **kwargs)
        RBM.__init__(self,  **kwargs)
        CostCD.__init__(self, **kwargs)
        WeightRegular.__init__(self,  **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class RBMPCDGOH(Notifier, UnitsRBMSigmoid,
                SparsityGOH,
                ActivationCrop,
                RBM, CostPCD, WeightRegular, SerializeLayer, Monitor,
                Plotter):
    """ RBM layer with sigmoid units, PCD cost and GOH sparsity regularization """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsRBMSigmoid.__init__(self)
        SparsityGOH.__init__(self, **kwargs)
        RBM.__init__(self, **kwargs)
        CostPCD.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        ActivationCrop.__init__(self, **kwargs)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class RBMCDGauss(Notifier, UnitsRBMGauss,
                        SparsityLee,
                        MaxNormRegular,
                        RBM,
                        CostCD,
                        WeightRegular, SerializeLayer, Monitor,
                        Approximator, Plotter):
    """ RBM layer with gaussian units, CD cost and Lee sparsity regularization """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsRBMGauss.__init__(self)
        SparsityLee.__init__(self, **kwargs)
        RBM.__init__(self, **kwargs)
        CostCD.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Approximator.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class RBMPCDReLU(Notifier, UnitsRBMReLU,
                UnitsDropOut,
                SparsityLee, MaxNormRegular,
                RBM, CostCD, WeightRegular, SerializeLayer, Monitor,
                ActivationCrop, Plotter):
    """ RBM layer with ReLU units, CD cost and Lee sparsity regularization """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsRBMReLU.__init__(self)
        UnitsDropOut.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        RBM.__init__(self, **kwargs)
        CostCD.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        ActivationCrop.__init__(self, **kwargs)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
         
class NNConvLinearBN(Notifier, UnitsCNNLinear,
                     UnitsDropOut,
                     CNN_BN, 
                     CostCrossEntropy,
                     WeightRegular,
                     MaxNormRegular,
                     SparsityLeeConv,
                     SerializeLayer, Monitor, Plotter):
    """ Convolutional Layer with linear activation and batch normalization """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNLinear.__init__(self, **kwargs)
        UnitsDropOut.__init__(self, **kwargs)
        CNN_BN.__init__(self, **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class NNConvSigmoidBN(Notifier, UnitsCNNSigmoid,
             UnitsDropOut,
             CNN_BN, 
             CostCrossEntropy,
             WeightRegular,
             MaxNormRegular,
             SparsityLeeConv,
             SerializeLayer, Monitor, Plotter):
    """ Convolutional Layer with sigmoid activation and batch normalization """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNSigmoid.__init__(self, **kwargs)
        UnitsDropOut.__init__(self, **kwargs)
        CNN_BN.__init__(self, **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class NNConvSigmoidAuto(Notifier, UnitsCNNSigmoid,
             UnitsDropOut,
             CNN, 
             CostReconErr,
             WeightRegular,
             SparsitySum,
             MaxNormRegular,
             SparsityLeeConv,
             NNFeatures,
             SerializeLayer, Monitor, Plotter):
    """ Single Audo-Encoder Layer with sigmoid activation """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNSigmoid.__init__(self,  **kwargs)
        UnitsDropOut.__init__(self, **kwargs)
        CNN.__init__(self, **kwargs)
        CostReconErr.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        SparsitySum.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        NNFeatures.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class NNConvReLUBN(Notifier, UnitsCNNReLU,
                 UnitsDropOut,
                 CNN_BN,
                 CostCrossEntropy,
                 MaxNormRegular,
                 WeightRegular,
                 SparsityLeeConv,
                 SerializeLayer, Monitor, Plotter):
    """ Convolutional Layer with ReLU activation and batch normalization """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        try:
            UnitsCNNReLU.__init__(self, downsample_out=kwargs['downsample_out'])
        except KeyError:
            raise ValueError("Entry 'downsample_out' in kwargs needed.")
        UnitsDropOut.__init__(self, **kwargs)
        CNN_BN.__init__(self,  **kwargs)
        CostCrossEntropy.__init__(self)
        WeightRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class NNDConvSigmoid(Notifier, UnitsCNNSigmoid,
             DCNN,
             CostCrossEntropy,
             MaxNormRegular,
             WeightRegular,
             SparsityLeeConv,
             SerializeLayer, Monitor, Plotter):
    """ De-Convolution layer - works only for 1d convolution yet. """
    def __init__(self, variables, filter_shape, input_shape, out_shape, name, 
                 **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNSigmoid.__init__(self, **kwargs)
               
        assert not -1 in out_shape, "For de-convolutional layers, you need to " + \
                        "define the 'out_shape' in the configuration file."
        
        filt_size = (input_shape[1],out_shape[1],filter_shape[2],out_shape[3])
        LOGGER.debug("out_size {0}".format(out_shape))
        LOGGER.debug("filt_size {0}".format(filt_size))      
        
        DCNN.__init__(self, variables, name, input_shape, filt_size, out_shape, 
                      **kwargs)

        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        
        SparsityLeeConv.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class NNDConvReLU(Notifier, UnitsCNNReLU,
                 DCNN,
                 CostCrossEntropy,
                 MaxNormRegular,
                 WeightRegular,
                 SparsityLeeConv,
                 SerializeLayer, Monitor, Plotter):
    """ De-Convolution Layer with Relu activation - works only for 1d convolution yet. """
    def __init__(self, variables, filter_shape, input_shape, out_shape, name, 
                 **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNReLU.__init__(self, **kwargs)
               
        assert not -1 in out_shape, "For de-convolutional layers, you need to " + \
                            "define the 'out_shape' in the configuration file."
        
        filt_size = (input_shape[1],out_shape[1],filter_shape[2],out_shape[3])
        LOGGER.debug("out_size {0}".format(out_shape))
        LOGGER.debug("filt_size {0}".format(filt_size))    
        
        DCNN.__init__(self, variables, name, input_shape, filt_size, out_shape, 
                      **kwargs)

        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class NNLinearBN(Notifier, UnitsNNLinear, UnitsDropOut, NN_BN,
                 CostCrossEntropy, WeightRegular, MaxNormRegular,
                 SparsityLee, SerializeLayer, Monitor, Plotter):
    """ 
    NN Layer with linear activation and batch normalization.
    """ 
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNLinear.__init__(self)
        UnitsDropOut.__init__(self, **kwargs)
        NN_BN.__init__(self, **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class NNSigmoidBN(Notifier, UnitsNNSigmoid, 
                  UnitsDropOut,
                  NN_BN, 
                  CostCrossEntropy,
                  WeightRegular,
                  MaxNormRegular,
                  SparsityLee,
                  SerializeLayer, Monitor,
                  Plotter):
    """ 
    NN Layer with sigmoid activation and batch normalization.
    """ 
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNSigmoid.__init__(self)
        UnitsDropOut.__init__(self, **kwargs)
        NN_BN.__init__(self, **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class NNReLUBN(Notifier, UnitsNNReLU, 
               NN_BN,
               CostSquaredError,
               WeightRegular, 
               MaxNormRegular, 
               SparsityLee, 
               SerializeLayer, Monitor, Plotter):
    """ 
    NN Layer with ReLU activation and batch normalization.
    """ 
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNReLU.__init__(self)
        NN_BN.__init__(self, **kwargs)
        CostSquaredError.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)


class NNSoftmaxBN(Notifier, UnitsNNSoftmax, NN_BN, CostCategoricCrossEntropy,
                  WeightRegular, MaxNormRegular, SerializeLayer, Monitor,
                  Plotter):
    """ 
    NN Layer with Softmax activation and batch normalization.
    """ 
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNSoftmax.__init__(self)
        NN_BN.__init__(self, **kwargs)
        CostCategoricCrossEntropy.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class AutoEncoderConvCrossEntropy(Notifier, UnitsCNNSigmoid,
                                     UnitsDropOut,
                                     CNN, 
                                     CostCategoricCrossEntropyAuto,
                                     WeightRegular,
                                     SparsitySum,
                                     MaxNormRegular,
                                     SparsityLeeConv,
                                     NNFeatures,
                                     SerializeLayer, Monitor, Plotter):
    """ Single Auto-Encoder Layer with cross entropy cost """
    def __init__(self, **kwargs):
        assert kwargs["convolutional"] == True, "Set 'convolutional = True' in configuration."
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        Plotter.__init__(self)
        UnitsCNNSigmoid.__init__(self, **kwargs)
        UnitsDropOut.__init__(self, **kwargs)
        CNN.__init__(self, **kwargs)
        CostCategoricCrossEntropyAuto.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        MaxNormRegular.__init__(self, **kwargs)
        SparsityLeeConv.__init__(self, **kwargs)
        SparsitySum.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        NNFeatures.__init__(self)
        Monitor.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class AutoEncoder(Notifier, UnitsNNLinear,
                  UnitsDropOut,
                  NonNegative,
                  NNAuto, CostSquaredError,
                  SparsityLee,
                  WeightRegular,
                  SerializeLayer, Monitor, Plotter):
    """ One-layered Auto-Encoder with Squared Error cost """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNLinear.__init__(self)
        UnitsDropOut.__init__(self, **kwargs)
        NonNegative.__init__(self, **kwargs)
        NNAuto.__init__(self, **kwargs)
        CostSquaredError.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        WeightRegular.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        

class RNNPredict(Notifier, UnitsNNSigmoid,
                 RNN, CostCrossEntropy, SparsityLee,
                 WeightRegularRNN,
                 SerializeLayer, Monitor, Plotter):
    """ Sigmoid prediction RNN with cross-entropy cost """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNSigmoid.__init__(self)

        RNN.__init__(self, act_fun_out=lambda x : T.nnet.sigmoid(x), **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        WeightRegularRNN.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class RNNGated(Notifier, UnitsNNTanh,
                 RNN_Gated, CostCrossEntropy, SparsityLee,
                 WeightRegularRNN,
                 SerializeLayer, Monitor, Plotter):
    """
    Gated RNN, as proposed in 'Learning Phrase Representations using RNN 
    Encoder-Decoder for Statistical Machine Translation', Cho et. al. 2014
    """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNTanh.__init__(self)

        RNN_Gated.__init__(self, act_fun_out=lambda x : T.nnet.sigmoid(x), **kwargs)
        CostCrossEntropy.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        WeightRegularRNN.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class RNNLSTM(Notifier, UnitsNNTanh,
           LSTM, CostLogLikelihoodBinomial, SparsityLee,
           WeightRegularRNN,
           SerializeLayer, Monitor, Plotter):
    """ Long-short term memory, Hochreiter et. al. 1997 """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        UnitsNNTanh.__init__(self)
        
        LSTM.__init__(self, act_fun_out=lambda x : T.nnet.sigmoid(x), **kwargs)
        CostLogLikelihoodBinomial.__init__(self, **kwargs)
        SparsityLee.__init__(self, **kwargs)
        WeightRegularRNN.__init__(self, **kwargs)
        SerializeLayer.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class MaxPooling(Notifier, MaxPooler):
    """ Standard max-pooling layer """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        MaxPooler.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class MaxPoolingOverlap(Notifier, MaxPoolerOverlapping):
    """
    Overlapping max-pooling (for convolutiona in time -
    yields spatio-temporal representations
    """
    def __init__(self, input_sym, in_size, name = "MaxPoolerOverlapping",
                 **kwargs):
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        MaxPoolerOverlapping.__init__(self, **kwargs)  
        self.notify(Notifier.MAKE_FINISHED)      
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class UpSampler(Notifier, UpSampling, Plotter):
    """ Upsampling of the hidden activation """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        UpSampling.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class TransitionFunctionConv(Notifier, TransitionFunc):
    """ Projects v_in through a theano function 'trans_func' """
    def __init__(self, **kwargs):
        if kwargs['convolutional']:
            kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        else:
            kwargs['variables'] = create_in_trg(kwargs['name'])
        try:
            _ = kwargs['trans_func']
        except KeyError:
            raise ValueError("Please pass the parameter 'trans_func'")
        Notifier.__init__(self)
        TransitionFunc.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class Normalizer(Notifier, Normalizing):
    """ min-max normalization of input """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)
        Normalizing.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class ConvShaper(Notifier, ConvShaping):
    """ 
    When convolution only in axis 2 with one input map, putting this layer
    after the output results in one input map for next layer.
    Accelerates computation of a stack. 
    """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        ConvShaping.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
class ConvDShaper(Notifier, ConvDShaping):
    """ Opposite of ConvShaper """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg_conv(kwargs['name'])
        Notifier.__init__(self)
        ConvDShaping.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)

class ConvToDense(Notifier, ToDense):
    """ Transforms convolutional output in dense input """
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in(kwargs['name'])
        Notifier.__init__(self)
        ToDense.__init__(self, **kwargs)
        self.notify(Notifier.MAKE_FINISHED)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
def create_in(name):
    input_sym = T.matrix(name='input' + name, dtype=fx)
    return {'input': input_sym}

def create_in_trg(name):
    input_sym = T.matrix(name='input' + name, dtype=fx)
    target_sym = T.matrix(name='target' + name, dtype=fx)
    return {'input': input_sym, 'target': target_sym}

def create_in_conv(name):
    input_sym = T.tensor4(name='input' + name, dtype=fx)
    return {'input': input_sym}

def create_in_trg_conv(name):
    input_sym = T.tensor4(name='input' + name, dtype=fx)
    target_sym = T.tensor4(name='target' + name, dtype=fx)
    return {'input': input_sym, 'target': target_sym}

def make_net(config, input_shape, in_maps = 1, **kwargs):
    """
    Creates a network architecture, based on a configuration file.

    Parameters
    ----------

    config : dictionary
        a dictionary created from a config file (which satisfies config_spec.ini)
        with the method config.get_config('config_model.ini', 'config_spec.ini')

    input_shape : int or 2D tuple
        The size of one training example. For convolutional nets, a 2D tuple is
        required, for standard nets, size can be given in 2D or as an integer.

    in_maps : int, optional
        For convolutional nets, defines the number of maps in the main input.
        
    cond_in_fun [kwargs] : function, optional
        For conditional nets, this function will be called with parameters
        (name_layer, ("h" or "v"), input_sym) and the function shall return a theano
        symbolic variable representing the bias for hidden or visible units.

    Returns
    -------
    list
        list of layers, as specified in config file

    """
    layers = []
    in_size_before = None
    filt_size = None
    
    for layer_name, curr_sec in config["TOPOLOGY"].items():
        
        type_net = curr_sec["type_net"]
        LOGGER.info("Creating layer {0} ({1})...".format(layer_name, type_net))
        convolutional = curr_sec["convolutional"]
        no_param_layer = curr_sec["no_param_layer"]
        filter_shape = curr_sec['filter_shape']
        batch_size = curr_sec['batch_size']

        if len(layers) > 0 and not convolutional \
                                and layers[-1].convolutional \
                                    and not isinstance(layers[-1], ConvToDense):
            raise Exception("Use a 'ConvToDense' layer to connect a convolutional "
                            "layer with a standard NN layer. Hint: Check, if "
                            "all convolutional layers have 'convolutional = True' "
                            "set in the config.")

        if convolutional and not no_param_layer and filter_shape == [-1, -1]:
                LOGGER.warning("Filter size of convolutional layer is set to "
                               "(-1,-1), this may cause problems. "
                               "Did you forget to specify it in the "
                               "config file (e.g. filter_shape = -1, 10)?")
        if len(layers) > 0 and convolutional:
            # convolutional
            rand_in = np.random.random(in_size_before).astype(fx)
            LOGGER.debug("probe in shape = {0}".format(in_size_before))
            in_size = layers[-1].out(rand_in).shape[1:]
            LOGGER.debug("resulting out shape = {0}".format(in_size))
            # can be chosen -1 to use input size
            filter_shape[0] = layers[-1].out(rand_in).shape[2] \
                                    if filter_shape[0] < 0 else filter_shape[0]
            filter_shape[1] = layers[-1].out(rand_in).shape[3] \
                                    if filter_shape[1] < 0 else filter_shape[1]

            filt_size = [filter_shape[0], filter_shape[1]]
            in_size_before = layers[-1].out(rand_in).shape
            
        elif convolutional:
            # convolutional
            # first layer
            assert ndims(input_shape) > 1, "Input size has to be 2D for convolutional nets."

            in_size = in_maps, input_shape[0], input_shape[1]
            # can be chosen -1 to use input size
            filter_shape[0] = input_shape[0] if filter_shape[0] < 0 else filter_shape[0]
            filter_shape[1] = input_shape[1] if filter_shape[1] < 0 else filter_shape[1]

            filt_size = [filter_shape[0], filter_shape[1]]
            in_size_before = batch_size, in_maps, input_shape[0], input_shape[1]
        elif len(layers) > 0:
            try:
                in_size = [layers[-1].hidden_shape[0],]
            except IndexError:
                rand_in = np.random.random(in_size_before).astype(fx)
                LOGGER.debug("probe in shape = {0}".format(in_size_before))
                in_size = [layers[-1].out(rand_in).shape[1]]
                LOGGER.debug("resulting out shape = {0}".format(in_size))
            in_size_before = in_size
        else:
            # first layer
            try:
                in_size = [input_shape[1],]
            except (IndexError, TypeError):
                in_size = [input_shape,]
            in_size_before = in_size

        kwargs.update(curr_sec)
        kwargs.update({"name": layer_name,
                       "input_shape": in_size,
                       "filter_shape": filt_size})
        layers.append(make_layer(**kwargs))
               
    return layers

def make_layer(type_net, name, batch_size, input_shape, filter_shape,
               n_hidden, convolutional, out_shape = (-1,-1,-1), **kwargs):
    """
    Creates an NN layer using one of the pre-defined mix-in classes.

    Parameters
    ----------

    type_net : string
        The type_net of the desired layer, one type_net of the list specified in
        config_spec.ini (e.g. 'ConvRBMPCDDrop').

    batch_size : int
        the batch size

    n_hidden : int
        number of hidden units

    input_shape : tuple or int
        The size of the input data to the layer. Integer for standard NNs,
        tuple for convolutional NNs.

    filter_shape : tuple, optional
        size of filter (for convolutional nets)

    wl1 : float, optional
        L1 weight regularization

    wl2 : float, optional
        L2 weight regularization

    mu : float, optional
        'mu' parameter for GOH sparsity regularization

    phi : float, optional
        'phi' parameter for GOH sparsity regularization

    bias_h : float, optional
        initial bias for hidden units (negative bias is a 'cheap' sparsity
        enforcer)

    dropout_h : float, optional
        dropout on hidden units (e.g. 0.5)

    dropout_v : float, optional
        dropout on visible units (e.g. 0.2)
        
    max_norm : float, optional
        maximal norm of weights (weights will be lowered in case norm is exceeded)

    cd_n : int, optional
        if CD or PCD training, specify the number of gibbs steps per update.
        default = 1 (i.e. CD1)

    sparsity : float, optional
        sparsity regularization target (e.g. LEE sparsity)
        
    sparsity_strength : float, optional
        sparsity strength
        
    downsample : 2d int tuple, optional
        in case the layer is a downsampling layer, the ds factor can be
        specified here. e.g. (2,1) halfs size of first dimension
        
    cond_in_fun : function, optional
        For conditional nets, this function will be called with parameters
        (name_layer, ("h" or "v"), input_sym) and the function shall return a theano
        symbolic variable representing the bias for hidden or visible units.
        
    upsample : tuple, optional
        For upsampling layers, e.g. (2,1) doubles the size in first dimension

    Returns
    -------
    lrn2.nn_bricks.FFBase
        the constructed layer

    """

    if convolutional:
        input_sym = T.tensor4(name='input' + name, dtype=fx)
        target_sym = T.tensor4(name='target' + name, dtype=fx)
        filter_shape = n_hidden, input_shape[0], filter_shape[0], filter_shape[1]
    else:
        input_sym = T.matrix(name='input' + name, dtype=fx)
        target_sym = T.matrix(name='target' + name, dtype=fx)
        filter_shape = []
        out_shape = []

    kwargs.update({
                   "type_net": type_net,
                   "name": name,
                   "filter_shape": filter_shape,
                   "input_shape" : input_shape,
                   "out_shape": out_shape,
                   "input_sym": input_sym,
                   "target_sym": target_sym,
                   "batch_size": batch_size,
                   "n_hidden": n_hidden,
                   "out_shape": out_shape,
                   "convolutional": convolutional
                   })
    
    layer = None
    if type_net == "Custom":
        assert "custom_layers" in kwargs.keys(), "For custom types pass custom classes in a list kwargs['custom_layers']."
        for c in kwargs["custom_layers"]:
            if kwargs["custom_type"] == c.__name__:
                layer = c(**kwargs)
        if not layer:
            raise ValueError("Layer {0} not found in list kwargs['custom_layers']")
    else:        
        exec "layer = {0}(**kwargs)".format(type_net)
        
    assert layer is not None, "The keyword of the config file is not yet implemented".format(type_net)

    return layer
