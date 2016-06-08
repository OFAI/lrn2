'''
Created on Feb 20, 2015

@author: Stefan Lattner
'''

import os
import time
import logging

from lrn2.nn_bricks.monitor import Monitor
from lrn2.nn_bricks.optimize import Optimizer
from lrn2.nn_bricks.notifier import Notifier
from lrn2.nn_bricks.stacks import NNStack
from lrn2.nn_bricks.plot import dummy_tiler
from lrn2.nn_bricks.utils import batch_by_batch, compute_w_backup, \
     create_dir
from lrn2.data.live_corpus import LiveCorpus, ProjectFeatureSpaceIterator
from lrn2.data.formats.iterable import load_iter_items
from lrn2.data.domain.raw import RawVP
from _functools import partial
import copy
import sys

LOGGER = logging.getLogger(__name__)

# HINT: Use a tmp_folder with high I/O performace (e.g. SSD disk)
projection_settings_default = {'before_projection': None,
                               'meta_batches': 10,
                               'tmp_folder': ".",
                               'merge_files': 10}
    
def train_layer_wise(net, data, config, run_keyword, validation = None,
                     postfix = "", tile_fun = dummy_tiler, re_train = False,
                     load_only = False, exclude = [],
                     load_existing = False, handler = None,
                     project_settings = projection_settings_default,
                     **kwargs):
    
    """
    Trains a network (list of RBM layers, derived from RBMBase) or a stack
    (derived from NNStack) layer by layer, according to a configuration file
    (whose structure reflects that of util.config_spec.ini).

    Parameters
    ----------

    net : array-like
        List of NN Layers, derived from the FFBase class as returned from
        make.make_net().

    data : dict of array-likes
        Dictionary of 2D or 4D arrays. The data to train the network with.
        2-dimensional for usual NNs, 4-dimensional for convolutional NNs.
        Convention: data['input'] is main input data, data['target'] is main target data

    config : dictionary
        a dictionary created from a config file (which satisfies config_spec.ini)
        with the method config.get_config('config_model.ini', 'config_spec.ini')

    run_keyword : string, optional
        Arbitrary keyword for the current training run. Defines the name of the
        training output folder.

    validation : dict, optional
        validation set; dict of 2D or 4D arrays. data to which variables will be bound to.

    postfix : string, optional
        Might be used to differ between training runs within the same run_keyword.
        Allows for an additional (sub-)categorization.

    tile_fun : function
        Tiling function for output plots during training, which takes a 2D array
        (standard) or 4D array (for convolutional nets) and returns a 4D
        representation for plotting (a 2D outlay for 2D tiles).

    re_train : boolean, optional
        Per default, former training runs are continued (if run_keyword was used
        before). If re_train equals True, training starts from epoch 0.
        This is equal to deleting stored training files in the output folder.

    load_only : boolean, optional
        Tells, if layers should be loaded only and no further training should
        be performed.

    exclude: array-like, optional
        1D list of parameters to exclude from optimization

    project_settings: dict, optional
        Settings for projection through layers when using the LiveCorpus
        
    before_projection: function
        a function to be called before projecting one layer up

    meta_batches: int
        for projecting one layer up, combine multiple batches to save in
        temporary files. May be necessary when there are too much batches
        and limit of open files is reached
        
    tmp_folder: string
        Folder where tmp files are stored.
        Use a folder with high I/O performace (e.g. SSD disk)
    """

    out_dir = create_dir(config['OUTPUT']['output_path'], run_keyword)

    if isinstance(net, NNStack):
        net = net.layers

    img_interval = config['OUTPUT']['img_interval']
    dump_interval = config['OUTPUT']['dump_interval']
    
    if handler is not None:
        LOGGER.addHandler(handler)

    assert len(net) == len(config["TOPOLOGY"].keys()), \
        "Length of net ({0}) != defined layers in config file ({1}).".format(len(net), len(config["TOPOLOGY"].keys()))
        
    data = copy.copy(data)
    
    for i, (layer_name, curr_sec) in enumerate(config["TOPOLOGY"].items()):
        batch_size = curr_sec['batch_size']
        n_epochs = curr_sec['epochs']
        lr = curr_sec['learning_rate']
        reduce_lr = curr_sec['reduce_lr']
        momentum = curr_sec["momentum"]
        no_param_layer = curr_sec["no_param_layer"]
        
        if not no_param_layer:
            out_dir_layer = create_dir(out_dir, layer_name)
            tmp_fn = os.path.join(out_dir_layer,
                                  "{0}_train_cache_{1}.pyc.bz"
                                  .format(layer_name, postfix))
            net[i] = compute_w_backup(net[i], tmp_fn, train,
                                      (net[i], data, batch_size, n_epochs, lr,
                                       reduce_lr, momentum, validation,
                                       out_dir_layer, img_interval,
                                       dump_interval, tile_fun, exclude),
                                      refresh_cache = re_train,
                                      load_only = load_only,
                                      load_existing = load_existing,
                                      **kwargs)

        # Project data to next layer
        if not load_only and len(net) > i+1:
            if data == None:
                before_projection = project_settings['before_projection']
                meta_batches = project_settings['meta_batches']
                tmp_folder = project_settings['tmp_folder']
                merge = project_settings['merge_files']
                LOGGER.info("Projecting through layer {0}...".format(net[i].name))
                if before_projection is not None:
                    before_projection()
                first_batch = net[i].notify(Notifier.GET_DATA, 0)[0]
                out_size = net[i].out(first_batch).shape
                ngram_size = out_size[2]
                new_corpus = LiveCorpus(load_iter_items, viewpoints = (RawVP(out_size[3]),),
                                        ngram_size = ngram_size,
                                        convolutional = len(out_size) > 2,
                                        verbose = False, shuffle = True,
                                        step_width = ngram_size, use_labels = False)
                new_corpus.remove_tmp_files(tmp_folder)
                files = new_corpus.convert_files(ProjectFeatureSpaceIterator(net[i],
                                                                             True,
                                                                             meta_batches = meta_batches),
                                                 tmp_folder)
                assert len(files) > 0, "No files after projection. If there is very little data, consider to reduce the 'meta_batches' parameter."
                
                files = new_corpus.merge_files(files, merge=merge, axis=0)
                new_corpus.open_files(files, tmp_folder)
                net[i+1].callback_del(Notifier.GET_DATA)
                net[i+1].callback_add(partial(new_corpus.get_data_callback,
                                              batch_size = batch_size),
                                      Notifier.GET_DATA)
                net[i+1].corpus = new_corpus
                tile_fun = new_corpus.tile_ngram_data
            else:
                # Project to next layer (train and validation set)
                out_size = list(net[i].out(data['input'][:batch_size]).shape[1:])
                data['input'] = batch_by_batch(net[i].out, data['input'], batch_size, out_size)
                tile_fun = dummy_tiler
            if validation is not None:
                validation = [batch_by_batch(net[i].out, validation['input'],
                                             batch_size, out_size)]
    return net

def train_cached(net, data, config, run_keyword, validation = None,
                 postfix = "", tile_fun = dummy_tiler, exclude = [],
                 re_train = False, load_only = False, load_existing = False,
                 continue_existing = False, handler = None, **kwargs):
    """
    Trains a single layer or a stack with backpropagation by using the cost of
    the input net, according to a configuration file (whose structure reflects
    that of util.config_spec.ini)
    
    The trained net will get automatically cached and if re_train == False,
    it won't get trained again, but only the trained net will be returned.

    Parameters
    ----------

    net : FFBase (or derivatives)
        the net (or stack) to train

    data : dictionary of array-likes or None
        Dictionary of 2D or 4D arrays to set the values of the variables of the cost
        (has to match the order of self.variables).
        Convention: data['input'] is main input data, data['target'] is main target data
        If data == None, the net has to have a callback function for the
        notification event 'get_data'.

    config : dictionary
        the configuration file loaded as dictionary. Has to have a section
        [STACK], which reflects that in util.config_spec.ini

    run_keyword : string
        the overall run_keyword which corresponds to the output folder name

    validation : dict, optional
        validation set; dict of 2D or 4D arrays. data to which variables will be bound to.

    postfix : string, optional
        will be used within the name of the output files (e.g. for
        distinguishing training runs with the same run_keyword)

    tile_fun : function, optional
        takes an input matrix and creates a list of matrixes with the shape of
        the desired plots (how to tile the input data)

    exclude : list (of parameters)
        1D list of parameters to exclude from parameter updates

    re_train : boolean, optional
        if true, training starts anew and cache files will be overwritten

    load_only : boolean, optional
        if true, cache files will be loaded but no training takes place


    Returns
    -------

    the (trained) net

    """
    out_dir = os.path.join(config['OUTPUT']['output_path'], run_keyword)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if handler is not None:
        LOGGER.addHandler(handler)

    config_bp = config["STACK"]
    kwargs.update(config_bp)

    tmp_fn = os.path.join(out_dir, "{0}_train_cache_{1}.pyc.bz".format(net.name,
                                                                       postfix))

    if re_train:
        net.reset_monitor()
        
    args = {'validation': validation,
            'out_dir': out_dir, 
            'tile_fun': tile_fun, 
            'exclude': exclude}
    kwargs.update(args)
    return compute_w_backup(net, tmp_fn, train,
                            refresh_cache = re_train, load_only = load_only,
                            load_existing = load_existing,
                            continue_existing = continue_existing,
                            args = (net, data), **kwargs)



def send_plot_command(net, out_dir, tile_fun, curr_epoch, data_batch):
    net.notify(Notifier.TRAINING_STOP)
    net.notify(Notifier.PLOTTING, data_batch=data_batch, out_dir=out_dir, 
               epoch_nr=curr_epoch, tile_fun_corpus=tile_fun)
    net.notify(Notifier.TRAINING_START)


def validate(net, validation, batch_size):
    stop = len(validation.values()[0])
    cost_valid_sum = 0
    for i in range(0, stop, batch_size):
        cost_valid_sum += net.validate(*[entry[i:i + batch_size]
                                         for entry in validation.values()])
    
    cost_valid = 1.0 * cost_valid_sum / (stop // batch_size)
    return cost_valid

def train(net, data, batch_size = 200, n_epochs = 500, learning_rate = 1e-4, 
          reduce_lr = False,
          momentum = 0.0, validation = None, out_dir = '.',
          img_interval = -1, dump_interval = -1, tile_fun = lambda x : x,
          exclude = [], plot_zero_epoch = True, grad_clip = None,
          **kwargs):

    """
    Trains a single layer or a stack with backpropagation using the cost of the
    input net.

    Parameters
    ----------

    net : FFBase (or derivatives)
        the net (or stack) to train

    data : dict of array-likes or None
        Dictionary of 2D or 4D arrays to set the values of the variables of the cost
        (has to match the order of self.variables).
        Convention: data['input'] is main input data, data['target'] is main target data
        If data == None, the net has to have a callback function for the
        notification event 'get_data'.

    batch_size : int
        the mini-batch size for training. if data == None, batch_size does
        not matter.

    n_epochs : int
        number of epochs to train the net

    learning_rate : float
        learning rate (initial, can be reduced during training by setting
        reduce_lr = True)

    reduce_lr : boolean, optional
        If True, learning rate will be reduced to 0 during training

    momentum : float, optional
        the momentum

    validation : dict, optional
        validation set; dict of 2D or 4D arrays.
        data to which variables will be bound to.

    out_dir : string
        the output folder, where training logs and plots will be written to

    img_interval : int, optional
        interval of epochs where images should be plotted to the output folder

    dump_interval : int, optional
        interval of epochs where the whole network should be dumped to the
        output folder

    tile_fun : function, optional
        takes an input matrix and creates a list of matrixes with the shape of
        the desired plots (how to tile the input data)

    exclude : list (of parameters)
        1D list of parameters to exclude from parameter updates

    Returns
    -------

    the (trained) net

    """
    
    assert data is not None or len(net.callbacks[Notifier.GET_DATA]) > 0, \
        "Either set the data parameter, and/or register a 'get_data' callback."

    
    LOGGER.info("\nTrain {0}: {1}...".format(net.name, type(net)))
    net.notify(Notifier.TRAINING_START)

    if tile_fun is None:
        tile_fun = dummy_tiler

    params = [p for p in net.params if id(p) not in [id(e) for e in exclude]]

    lr = learning_rate
    opt = Optimizer(net.cost(), params, net.variables, data,
                    batch_size, lr = lr, momentum = momentum,
                    notifier = net, grad_clip = grad_clip)
    
    net.optimizer = opt

    curr_epoch = net.epochs_trained if isinstance(net, Monitor) else 0

    LOGGER.info("Training starts in epoch {0}.".format(curr_epoch))

    try:
        for curr_epoch in range(curr_epoch, n_epochs):
            if data is not None:
                data_batch = {}
                for key, value in data.items():
                    data_batch[key] = value[:batch_size]
            else:
                data_batch = net.notify(Notifier.GET_DATA, 0)
                data_batch = dict([[net.variables.keys()[i], data_batch[i]]
                                   for i in range(len(data_batch))])
            
            if curr_epoch == 0 and out_dir is not None and plot_zero_epoch:
                send_plot_command(net, out_dir, tile_fun, curr_epoch, data_batch)

            start_time = time.time()

            cost_curr = opt.train()

#             if hasattr(net, 'validate'):
#                 cost_curr = float(net.validate(*(data_batch.values())))

            if isinstance(net, Monitor):
                net.monitor_cost(cost_curr)

            # reduce learning rate
            if reduce_lr:
                opt.learning_rate = lr - curr_epoch * (lr / n_epochs)

            end_time = time.time()
            elapsed_epoch = end_time - start_time

            LOGGER.info("finished epoch {0}/{3} in {1:.2f} seconds (lr: {4:.3e}; cost: {2:.4f})"
                         .format(curr_epoch+1, elapsed_epoch, cost_curr,
                                 n_epochs, float(opt.learning_rate)))

            if validation is not None and hasattr(net, 'validate'):
                cost_valid = validate(net, validation, batch_size)
                LOGGER.info("Validation set cost = {0}".format(cost_valid))
                if isinstance(net, Monitor):
                    net.monitor_cost_val(cost_valid)
                
                
            if dump_interval > 0 and curr_epoch % dump_interval == 0:
                try:
                    net.save(os.path.join(out_dir, "net_{0}_backup_{1}.pyc.bz"
                                          .format(net.name, curr_epoch)))
                except AttributeError:
                    LOGGER.warning("Net could not be saved. Derive from "
                                   "brick SerializeStack or SerializeLayer.")
                    pass

            if curr_epoch % img_interval == 0 and out_dir is not None and curr_epoch > 0:
                send_plot_command(net, out_dir, tile_fun, curr_epoch, data_batch)
                
            if isinstance(net, Notifier):
                net.notify(Notifier.EPOCH_FINISHED, curr_epoch = curr_epoch, 
                           n_epochs = n_epochs)


    except KeyboardInterrupt:
        LOGGER.info("Training interrupted in epoch {0}, {1}"
                    .format(curr_epoch, net.name))
        send_plot_command(net, out_dir, tile_fun, curr_epoch, data_batch)

    net.notify(Notifier.TRAINING_STOP)

    return net
