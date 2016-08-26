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

from functools import partial
from logging import FileHandler
from lrn2.data.corpus import Corpus
from lrn2.util.config import get_config
from lrn2.nn_bricks.make import make_net
from lrn2.nn_bricks.train import train_cached
from lrn2.data.domain.image import ImageBinaryVP
from lrn2.nn_bricks.stacks import FFNNCrossEntropy
from lrn2.nn_bricks.utils import fx, get_from_cache_or_compute
from lrn2.data.formats.mnist import load_mnist_files, MNISTInterface
from collections import OrderedDict

class MnistVP(MNISTInterface, ImageBinaryVP): pass

logging.basicConfig()
LOGGER = logging.getLogger("lrn2")
LOGGER.setLevel(logging.DEBUG)

def setup_mnist_corpus(datafiles, labels, ngram_size,
                       shuffle = True, limit_instance_count = None,
                       subsets = ("train",)):

    file_loader = partial(load_mnist_files, subsets = subsets)

    viewpoints = [MnistVP(shape = (28, 28))]

    corpus = Corpus(file_loader, viewpoints, convolutional=True)

    LOGGER.info('loading data files')
    corpus.load_files(datafiles)

    LOGGER.info('constructing ngrams')
    corpus.set_to_ngram(ngram_size)

    if shuffle:
        corpus.shuffle_instances()

    if limit_instance_count is not None:
        corpus.limit_instance_count(limit_instance_count)

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

def test_mnist(args, config):
    """ for images, leave ngram_size at one (unless you want to
    learn representations of more than one subsequent images at once) """
    ngram_size = config['INPUT']['ngram_size']

    out_dir, handler = prepare(args, config)

    data_fn = os.path.join(os.getcwd(), 'data', 'mnist', 'mnist.pkl.gz')
    tmp_fn_corp_train = os.path.join(out_dir, "corpus_cache_train.pyc.bz")
    tmp_fn_corp_test = os.path.join(out_dir, "corpus_cache_test.pyc.bz")
    
    LOGGER.info("Initializing corpus...")
    train_corpus = get_from_cache_or_compute(tmp_fn_corp_train,
                                             setup_mnist_corpus,
                                             (data_fn, None, ngram_size),
                                             refresh_cache = args.rebuild_corpus)

    valid_corpus = get_from_cache_or_compute(tmp_fn_corp_test,
                                            setup_mnist_corpus,
                                            (data_fn, None, ngram_size, 
                                             True, None, ('valid',)), 
                                            refresh_cache = args.rebuild_corpus)

    LOGGER.info("Initializing and training NN...")
    
    # Create NN layers
    layers = make_net(config, train_corpus.shape[2:])
    
    # Define input data
    labels_bin = binary_labels(train_corpus.ngram_labels).astype(fx)
    data = OrderedDict((('input', train_corpus.ngram_data),
                        ('target', labels_bin)))

    # Validation data
    labels_bin_valid = binary_labels(valid_corpus.ngram_labels).astype(fx)
    data_valid = OrderedDict((('input', valid_corpus.ngram_data),
                             ('target', labels_bin_valid)))

    # Pack layers in stack with cross entropy cost
    stack = FFNNCrossEntropy(layers, 'mnist_classifier')
    
    # Train the whole stack in discriminative task
    LOGGER.info("Training with backpropagation (whole stack)...")
    train_cached(stack, data, config, args.run_keyword,
                 validation = data_valid,
                 tile_fun = train_corpus.tile_ngram_data,
                 re_train = args.re_train, handler = handler,
                 load_existing = args.load_existing, plot_zero_epoch = True)

def binary_labels(labels_str):
    result = []
    for l in labels_str:
        result.append([0]*10)
        result[-1][int(l)] = 1
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description = "Run an MNIST classification task with convolutional layers.")

    parser.add_argument("run_keyword", metavar = "run_keyword", help = "Keyword for the current test")

    parser.add_argument("modelconfig", help = "model config file")

    parser.add_argument("--re-train", action = "store_true", default = False,
                        help = "re-trains RBM, ignoring cached files corresponding to the run_keyword used")
    
    parser.add_argument("--load-existing", action = "store_true", default = False,
                       help = "loads existing parameters, trains non-existing")
    
    parser.add_argument("--rebuild-corpus", action = "store_true", default = False,
                        help = "reloads data and rebuilds temporary files for accelerating data import")

    args = parser.parse_args()

    config = get_config(args.modelconfig)

    test_mnist(args, config)

if __name__ == '__main__':
    main()
