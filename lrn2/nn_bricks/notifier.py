'''
Created on May 5, 2015

@author: Stefan Lattner
'''
from _collections import defaultdict
from lrn2.util.utils import shape
import logging
from _functools import partial

LOGGER = logging.getLogger(__name__)

class Notifier(object):

    # Can be sent after a layer or a stack has been initialized
    MAKE_FINISHED = "make_finished"
    
    # notifies all registered callbacks to compile their theano functions
    COMPILE_FUNCTIONS = "compile_functions"
    
    # notifies all registered callbacks to register their plotting functions
    REGISTER_PLOTTING = "register_plotting"
    
    # callbacks are called, after each trained batch
    BATCH_FINISHED = "batch_finished"
    
    # EPOCH_FINISHED kwargs: curr_epoch, n_epochs
    EPOCH_FINISHED = "epoch_finished"
    
    # callbacks are called, when the gradient was calculated
    GRADIENT_CALCULATED = "gradient_calculated"
    
    GET_DATA = "get_data"   # Optimizer can fetch data batch by batch
        
    # PLOTTING args: (data_batch, out_dir, epoch_nr, tile_fun_corpus)
    PLOTTING = "plot"

    # E.g. for enabling and suspending dropout
    # (train_stop is also called before plotting during training)
    TRAINING_START = "train_start"
    TRAINING_STOP = "train_stop"
    
    # SAVE no args, but expects (value, name) in return
    # !!! Ensure that name == variable name in self.__dict__
    SAVE = "save"
    
    #LOAD args: state
    #    call state["name"] to set value in your class
    LOAD = "load"
    
    LEARNING_RATES = 'learning_rates'
    PARAM_MULT = 'parameter_multiplier'
    MOMENTUM = 'momentum'

    types_forw = (BATCH_FINISHED, EPOCH_FINISHED,
                  PLOTTING, TRAINING_START, TRAINING_STOP,
                  MOMENTUM)

    def __init__(self):
        self.callbacks = defaultdict(lambda : [])

    def callback_del(self, name):
        """ deletes all callback registrations with name 'name' """
        self.callbacks[name] = []
        
    def callback_add(self, fun, name, forward = False):
        """ Register a callback function to an event name """
        self.callbacks[name].append((fun, forward))
#         print "callback_registered:", name, self.name

    def notify(self, type, *args, **kwargs):
        """ Notify all callback functions registered to an event type """
        results = []
        for f, forward in self.callbacks[type]:
            try:
                if kwargs['forward'] == True and forward:
                    kwargs2 = dict([(key, kwargs[key]) for key in kwargs.keys()])
                    kwargs2.pop("forward", None)
                    results.append(f(*args, **kwargs2))
            except KeyError:
                results.append(f(*args, **kwargs))

        return results
    
class NotifierForwarder(object):
    def __init__(self, layers):
        self.layers_notif = layers
        self.register_notiforwarding()
        
    def callback_forward(self, layers):
        self.layers_notif = layers
        
    def register_notiforwarding(self):
        # Forwards all single layer notifiers to the stack notifier
        for t in Notifier.types_forw:
            self.callback_add(partial(self.notiforward, t), t)
            
    def notiforward(self, name, *args, **kwargs):
        kwargs["forward"] = True
        for l in self.layers_notif:
            l.notify(name, *args, **kwargs)