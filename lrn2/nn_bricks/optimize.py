'''
Created on Feb 20, 2015

@author: Stefan Lattner
'''

import theano
import logging
import numpy as np
import theano.tensor as T

from _functools import partial
from lrn2.util.utils import shape
from lrn2.nn_bricks.utils import fx
from _collections import defaultdict
from lrn2.nn_bricks.notifier import Notifier

LOGGER = logging.getLogger(__name__)

class Optimizer(object):
    """
    Optimizes a theano graph with gradient descent.

    Parameters
    ----------

    cost : theano graph
        the cost function (a theano graph)

    params : list
        a list of parameters to optimize (part of the cost graph)

    variables: dictionary
        a dict of tensor.TensorVariable (symbols), which are part of the
        cost function (typically input and output variables). Those will be
        instantiated by mini batches of the actual input and output data.

    data : dictionary
        a dict of 2d array-likes or None (in case of data input through callback)
        of the same length than variables.
        The keys have to correspond to the respective symbols of variables.
        This is the actual data the model will be trained on.

    batch_size : int
        the batch size

    lr : float
        the learning rate

    momentum : float, optional
        the momentum

    grad_clip : float, optional
        clips gradients before updating the parameters
        (at the leafs of the graph). Default = None

    nan_protection : boolean, optional
        replaces NaNs in the gradient with 0. Default = True

    notifier : lrn2.nn_bricks.Notifier
        an external notifier which contains callbacks reacting on
        Notifier.LEARNING_RATES, Notifier.PARAM_MULT, Notifier.MOMENTUM,
        Notifier.GRADIENT_CALCULATED
        Attention: Notifier.GET_DATA callbacks will be updated by the Optimizer!
        
    validate : theano graph
        if separate validation function than the cost to be minimized should
        be output as cost during training. (e.g. in Boltzmann Machines the
        free energy should be minimized, but the reconstruction error should
        be shown during training and in statistics) 

    """
    def __init__(self, cost, params, variables, data, batch_size, lr,
                 momentum = 0., grad_clip = None,
                 nan_protection = True, notifier = None, validate = None):

        self.params = params

        LOGGER.debug("Optimizing parameter(s): {0}".format(params))
        param_count = np.sum(np.asarray([np.prod(p.get_value().shape) for p in params]))
        LOGGER.debug("Number parameter(s): {0}".format(param_count))
        LOGGER.debug("Given variable(s): {0}".format(variables))
        if data is not None:
            LOGGER.debug("Given data: {0}".format([[k, shape(data[k])] for k in data.keys()]))

        self.gparams = [theano.shared(x.get_value() *
                                      np.cast[x.get_value().dtype](0.0),
                                      name = "%s_grad" % x.name)
                        for x in self.params]

        self.cost = cost

        if variables is None:
            self.variables = {}
        else:
            self.variables = variables

        self.data = data

        self.lr = theano.shared(np.cast[fx](lr))
        self.mom = theano.shared(np.cast[fx](momentum))
        self.grad_clip = grad_clip

        self.nan_protection = nan_protection

        self.batch_size = batch_size

        if data is not None:
            assert len(data) == len(self.variables), \
                "You assigned {0} symbolic variable(s), but you provide {1} data set(s): \nVariables: {2}\nDatasets: {3}" \
                                                .format(len(self.variables),
                                                len(data), self.variables,
                                                self.data.keys())
            if notifier is None:
                notifier = Notifier()

            notifier.callback_del(Notifier.GET_DATA)
            for key in variables.keys():
                notifier.callback_add(partial(self.get_data_callback, key=key),
                                      Notifier.GET_DATA)

        self.notifier = notifier
        self.validate = validate

        self.train_model = self.init_updates()

    def inspect_inputs(self, i, node, fn):
        print i, node, "input(s) value(s):", [inp[0] for inp in fn.inputs],

    def inspect_outputs(self, i, node, fn):
        print "output(s) value(s):", [output[0] for output in fn.outputs]

    def init_updates(self):
        gparams = T.grad(self.cost, self.params, disconnected_inputs = 'warn')

        # Remove NaNs
        if self.nan_protection:
            gparams = [T.switch(T.isnan(g), 0., g) for g in gparams]

        # Gradient clipping
        if self.grad_clip is not None:
            gparams = [T.minimum(g, self.grad_clip) for g in gparams]
            gparams = [T.maximum(g, -1. * self.grad_clip) for g in gparams]

        lr = defaultdict(lambda *args : self.lr)
        try:
            lr.update(dict(self.notifier.notify(Notifier.LEARNING_RATES)))
        except Exception:
            pass

        mult = defaultdict(lambda *args : np.cast[fx](1))
        try:
            mult.update(dict(self.notifier.notify(Notifier.PARAM_MULT)))
        except Exception:
            pass

        mom = defaultdict(lambda *args : self.mom)
        try:
            mom.update(dict(self.notifier.notify(Notifier.MOMENTUM)))
        except Exception:
            pass

        # Parameter updates
        updates_param = [
            (param, param * mult[param.name] - \
             lr[param.name] * ((1 - mom[param.name]) * gparam_cur + mom[param.name] * gparam_last))
            for param, gparam_cur, gparam_last in zip(self.params, gparams, self.gparams)
        ]

        # gradient updates for momentum
        updates_gparam = [
            (gparam_last, gparam_cur)
            for gparam_last, gparam_cur in zip(self.gparams, gparams)
        ]

        updates = updates_param + updates_gparam

        # Callback to an external function. E.g. there are non-integrable nodes
        # which should be added after the gradient calculation.
        if self.notifier is not None:
            if len(self.notifier.callbacks[Notifier.GRADIENT_CALCULATED]) > 0:
                grads_new = self.notifier.notify(Notifier.GRADIENT_CALCULATED, updates)
                if grads_new is not None and len(grads_new) > 0:
                    updates = np.vstack(grads_new)

        # ensure that the broadcastpattern before and after the update is identical
        updates = [(k, T.patternbroadcast(v, k.broadcastable))
                   for k, v in updates]

        validate = self.validate if self.validate is not None else self.cost
                        
        return theano.function(inputs=self.variables.values(),
                               outputs=validate,
                               updates=updates,
                               allow_input_downcast = True,
                               on_unused_input = 'warn')


    def train(self):
        """
        Executes a backpropagation training step for all parameters and data.

        Returns
        -------

        The current cost (averaged over mini-batches)

        """
        count = 0
        cost_sum = 0
        if self.data is not None or self.notifier is not None:
            while True:
                # get next mini-batch
                if self.notifier is not None:
                    batches = self.notifier.notify(Notifier.GET_DATA, count)
                else:
                    batches = []

                if batches is None:
                    break

                if None in batches:
                    break

                if len(batches[0]) == 0:
                    break

                assert len(batches) == len(self.variables), \
                "You assigned {0} symbolic variable(s), but you provide {1} data set(s): {2}" \
                                                .format(len(self.variables),
                                                len(batches), self.variables)
                len_curr_batch = len(batches[0])
                for i in range(len(batches)):
                    if len_curr_batch != len(batches[i]):
                        raise ValueError("You provided {0} datasets with different number of instances. "
                                         "In particular, we found a batch with size {1} (dataset {3}) and another batch "
                                         "with size {2} (dataset {4}).".format(len(batches), len_curr_batch, len(batches[i]), 0, i))

                # train model
                cost = self.train_model(*batches)
                cost_sum += cost
                count += 1

                if self.notifier:
                    self.notifier.notify(Notifier.BATCH_FINISHED)
        else:
            # No data: Possible, if cost function depends only on params
            cost_sum = self.train_model()
            count = 1

        if count == 0:
            return 0.

        return 1.0 * cost_sum / count

    def get_data_callback(self, batch_nr, key=''):
        curr_data = self.data[key]
        batch_size = self.batch_size

        assert len(curr_data) >= batch_size, \
            "Batch size ({0}) has to be <= than instance count ({1})." \
                                    .format(batch_size, len(curr_data))

        if (batch_nr + 1) * batch_size > len(curr_data):
            return None

        batches = curr_data[batch_nr * batch_size:(batch_nr + 1) * batch_size]

        return batches

    @property
    def callback_grad(self):
        return self._callback_grad

    @callback_grad.setter
    def callback_grad(self, value):
        self._callback_grad = value

    @property
    def learning_rate(self):
        return self.lr.get_value()

    @learning_rate.setter
    def learning_rate(self, value):
        self.lr.set_value(value)

    @property
    def momentum(self):
        return self.mom.get_value()

    @momentum.setter
    def momentum(self, value):
        self.mom.set_value(value)
