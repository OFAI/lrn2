'''
Created on Sep 23, 2014

@author: Stefan Lattner
'''
import os
import abc
import time
import string
import logging
import argparse
import numpy as np
import theano.tensor as T

from logging import FileHandler
from lrn2.data.corpus import Corpus
from lrn2.nn_bricks.plot import Plotter
from lrn2.util.config import get_config
from lrn2.nn_bricks.make import make_net
from lrn2.nn_bricks.layers import FFBase
from lrn2.data.domain.music import PitchVP
from lrn2.nn_bricks.notifier import Notifier
from lrn2.nn_bricks.train import train_cached
from lrn2.nn_bricks.units import UnitsNNLinear
from lrn2.util.utils import read_input_filelist
from lrn2.data.domain.viewpoint import ViewPoint
from lrn2.nn_bricks.stacks import FFNNCrossEntropy
from lrn2.nn_bricks.utils import fx, get_from_cache_or_compute
from lrn2.data.formats.midi import load_midi_files, MIDIInterface

class MidiPitchVP(MIDIInterface, PitchVP): pass

logging.basicConfig()
LOGGER = logging.getLogger("lrn2")
LOGGER.setLevel(logging.DEBUG)

"""
This is a very simple example of a Transforming Auto-Encoder (Hinton et. al. 2011),
where the whole net resembles a single capsule, and the goal is to learn to
encode an N bit input in a single number. The TransformingLayer simply adds
transformation information to its input (in this case the numerical difference
between in/output).
"""

""" Define custom layer """
class TransformingLayer(FFBase):
    def __init__(self, variables, input_shape, n_hidden, name, target_sym = None,
                 params=None, bias_h = 0., **kwargs):
        
        FFBase.__init__(self, variables, name = name, input_shape = input_shape,
                        hidden_shape = (n_hidden,))
                
    """ Implement FFBase abstract methods """
    def activation_v(self, h_in):
        raise NotImplementedError("Not defined for transforming layer.")

    def activation_h(self, v_in):
        # add transformation input to layer input
        return v_in + self.variables['trans_in']

    def output(self, v_in):
        return self.activation_h(v_in)
    
class Transformer(Notifier, UnitsNNLinear, TransformingLayer, Plotter):
    """ 
    NN Transformer Layer
    """ 
    def __init__(self, **kwargs):
        # Define symbolic variables
        trans_in = T.matrix("transform_input", dtype = fx)
        input_sym = T.matrix(name='input' + kwargs['name'], dtype=fx)
        kwargs['variables'] = {'input': input_sym, 'trans_in': trans_in}
        
        # initialize mix-in classes
        Notifier.__init__(self)
        UnitsNNLinear.__init__(self)
        TransformingLayer.__init__(self, **kwargs)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
        
""" Custom viewpoint """
class PitchIntervalNumericVP(ViewPoint):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(PitchIntervalNumericVP, self).__init__()

    @abc.abstractmethod
    def get_pitch_from_raw_data(self, data):
        raise NotImplementedError("Method supposed to be implemented in the "
                                  "format specific interface.")

    @property
    def size(self):
        return 1

    def raw_to_repr(self, raw_data, label):
        intervals = np.r_[np.diff(self.get_pitch_from_raw_data(raw_data)), 0]
        intervals = np.reshape(intervals, (intervals.shape[0], 1))
        return 1.0 * intervals

class MidiPitchIntervalVP(MIDIInterface, PitchIntervalNumericVP): pass

def setup_midi_corpus(datafiles, ngram_size, rebuild = False):
    viewpoints = ( MidiPitchVP(min_pitch = 36, max_pitch = 100), )

    corpus = Corpus(load_midi_files, viewpoints)

    LOGGER.info('loading data files')
    corpus.load_files(datafiles, clear = rebuild)
    corpus.set_to_ngram(ngram_size = ngram_size)
        
    return corpus

def setup_interval_corpus(datafiles, ngram_size, rebuild = False):
    viewpoints = ( MidiPitchIntervalVP(), )

    corpus = Corpus(load_midi_files, viewpoints)

    LOGGER.info('loading data files')
    corpus.load_files(datafiles, clear = rebuild)
    corpus.set_to_ngram(ngram_size = 1)
        
    return corpus

def prepare(args, config):
    out_dir = os.path.join(config['OUTPUT']['output_path'], args.run_keyword)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    time_str = string.join(list(np.cast['str'](time.gmtime())), sep='_')
    handler = FileHandler(os.path.join(out_dir, 'output_{0}.log'.format(time_str)))
    LOGGER.addHandler(handler)
    with open(os.path.join(out_dir, "config_{0}.ini".format(time_str)), 'w') as f:
        config.write(f)
    return out_dir, handler

def add_path(d, path):
    result = []
    for i in range(len(d)):
        result.append((os.path.join(path, d[i][0]), d[i][1]))

    return tuple(result)
    
def test_custom(args, config):
    ngram_size = config['INPUT']['ngram_size']

    _, handler = prepare(args, config)

    traindata = add_path(read_input_filelist(args.traindata), args.d)
    traindata = np.asarray(traindata)
    
    corpus = get_from_cache_or_compute("cache1.pyc.bz", setup_midi_corpus, 
                                       (traindata, ngram_size),
                                       {"rebuild": args.rebuild_corpus}, 
                                       refresh_cache = args.rebuild_corpus)
    
    corpus_int = get_from_cache_or_compute("cache2.pyc.bz", setup_interval_corpus, 
                                           (traindata, ngram_size),
                                           {"rebuild": args.rebuild_corpus}, 
                                           refresh_cache = args.rebuild_corpus)   
    
    data_src = corpus.ngram_data
    data_trg = np.roll(np.copy(data_src), -1, axis=0)
    data_transform = corpus_int.ngram_data
    
    nn_layers = make_net(config, input_shape = [ngram_size, corpus.size], 
                         custom_layers = [Transformer,])
    
    nn_transform = FFNNCrossEntropy(nn_layers, "TransformingAutoencoder")
    
    data = {'input': data_src, 'target': data_trg, 'trans_in': data_transform}
    
    test_scale(nn_transform, corpus.size)
        
    train_cached(nn_transform, data, config,
                 args.run_keyword, validation = None,
                 postfix = "layer_wise", tile_fun = corpus.tile_ngram_data,
                 re_train = args.re_train, load_only = args.no_training,
                 exclude = [], load_existing = args.load_existing, 
                 handler = handler)

def test_scale(nn, size):
    probe = np.eye(size, dtype = fx)
    dummy = np.zeros((probe.shape[0], probe.shape[1]*2))
    nn.register_plot(lambda *args : nn.layers[1].out(probe, dummy), 'probe', 
                     name_net = nn.name,
                     ptype = 'curve')
    
def main():
    parser = argparse.ArgumentParser(description = "Run a lrn2 custom layer example. "
                                     "In this case, the custom layer is a transformation "
                                     "layer, and the resulting stack can be seen "
                                     "as a single capsule of a transforming auto-encoder "
                                     "(see Hinton et. al. 2011).")

    parser.add_argument("run_keyword", metavar = "run_keyword", help = "Keyword for the current test")

    parser.add_argument("modelconfig", help = "model config file (stack, higher level)")

    parser.add_argument("traindata", help = ("text file listing tab separated "
                                             "(midifile, label) pairs (one per line), "
                                             "TXT files expected. For this example, "
                                             "use monophonic MIDIs (e.g. Essen Folksong Collection "
                                             "http://www.esac-data.org/). To create a "
                                             "file list, use lrn2/util/create_filelist.py."))

    parser.add_argument("-d", default=".", help = ("midi folder"))

    parser.add_argument("--re-train", action = "store_true", default = False,
                       help = "ignores stored training files corresponding to the run_keyword used")

    parser.add_argument("--no-training", action = "store_true", default = False,
                       help = "loads stored parameters, no further training takes place")

    parser.add_argument("--load-existing", action = "store_true", default = False,
                       help = "loads existing parameters, trains non-existing")


    parser.add_argument("--rebuild-corpus", action = "store_true", default = False,
                        help = "reloads data and rebuilds temporary files for accelerating data import")

    args = parser.parse_args()

    config = get_config(args.modelconfig)

    test_custom(args, config)

if __name__ == '__main__':
    main()

