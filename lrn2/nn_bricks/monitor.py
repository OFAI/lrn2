'''
Created on Feb 20, 2015

@author: Stefan Lattner
'''
from lrn2.nn_bricks.notifier import Notifier

class Monitor(object):
    '''
    Processes manipulating a network may write and read miscellaneous data here.
    '''
    def __init__(self):
        self.epochs_trained = 0

        # Store params before update (to infer e.g. changes in params)
        self.last_params = {}

        # Store history of reconstruction errors
        self.cost_curve = []
        
        # Store history of reconstruction errors
        self.cost_curve_val = []
        
        self.callback_add(self.monitor_training, Notifier.EPOCH_FINISHED)
        Monitor.register_plotting(self)

        self.monitor_training(curr_epoch = 0)
        
        self.callback_add(lambda *args : (self.epochs_trained, "epochs_trained"), 
                          Notifier.SAVE)
        self.callback_add(lambda state : 
                          self.set_epochs_trained(state["epochs_trained"]), 
                          Notifier.LOAD)

    def set_epochs_trained(self, epochs):
        self.epochs_trained = epochs
        
    def monitor_cost(self, cost):
        self.cost_curve.extend([cost])
        
    def monitor_gparams(self, gparams):
        self.gparams_mon = gparams
        
    def monitor_cost_val(self, cost):
        self.cost_curve_val.extend([cost])

    def store_last_params(self):
        if hasattr(self, 'params'):
            self.last_params = dict([(p.name, p.get_value()) 
                                     for p in self.params])
        
    def register_plotting(self):
        self.register_plot(lambda *args : self.cost_curve, 'cost_curve',
                           ptype = 'curve', name_net = self.name,
                           forward = False)
        self.register_plot(lambda *args : self.cost_curve_val, 'cost_curve_val',
                           ptype = 'curve', name_net = self.name,
                           forward = False)

    def monitor_training(self, **kwargs):
        """ Stores some relevant information to monitor training """
        self.epochs_trained = kwargs["curr_epoch"]
        self.store_last_params()
