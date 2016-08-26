'''
Created on Sep 23, 2014

@author: Stefan Lattner
'''
import os
import time
import string
import logging
import argparse
import numpy as np

from logging import FileHandler
from lrn2.data.corpus import Corpus
from lrn2.util.config import get_config
from lrn2.data.domain.music import PitchVP
from lrn2.nn_bricks.make import make_net
from lrn2.util.utils import read_input_filelist
from lrn2.nn_bricks.train import train_cached
from lrn2.data.formats.midi import load_midi_files, MIDIInterface
from lrn2.nn_bricks.utils import get_from_cache_or_compute
from collections import OrderedDict

class MidiPitchVP(MIDIInterface, PitchVP): pass

logging.basicConfig()
LOGGER = logging.getLogger("lrn2")
LOGGER.setLevel(logging.DEBUG)

def setup_midi_corpus(datafiles, ngram_size, rebuild = False):
    viewpoints = ( MidiPitchVP(min_pitch = 36, max_pitch = 100), )

    corpus = Corpus(load_midi_files, viewpoints)

    LOGGER.info('loading data files')
    corpus.load_files(datafiles, clear = rebuild)
    corpus.set_to_ngram(ngram_size = ngram_size)
        
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
    
def test_rnn(args, config):
    # ngram_size = 1 for RNNs
    ngram_size = config['INPUT']['ngram_size']

    _, handler = prepare(args, config)

    traindata = add_path(read_input_filelist(args.traindata), args.d)
    traindata = np.asarray(traindata)
    
    corpus = get_from_cache_or_compute("cache1.pyc.bz", setup_midi_corpus, 
                                       (traindata, ngram_size),
                                       {"rebuild": args.rebuild_corpus}, 
                                       refresh_cache = args.rebuild_corpus)
    
    rnn_layer = make_net(config, input_shape = corpus.size, plot_dparams = False)[0]

    data = OrderedDict((('input', corpus.ngram_data),
                       ('target', np.roll(corpus.ngram_data, shift = -1, axis = 0))))

    train_cached(rnn_layer, data, config,
                 args.run_keyword, validation = None,
                 postfix = "layer_wise", tile_fun = corpus.tile_ngram_data,
                 re_train = args.re_train, load_only = args.no_training,
                 exclude = [], load_existing = args.load_existing, 
                 handler = handler, plot_zero_epoch = False)

def main():
    parser = argparse.ArgumentParser(description = "Train an RNN on monophonic music.")

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

    test_rnn(args, config)

if __name__ == '__main__':
    main()

