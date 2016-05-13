'''
Created on Feb 18, 2015

@author: Stefan Lattner
'''
import os
import logging
from theano.compile.sharedvalue import SharedVariable
from lrn2.nn_bricks.notifier import Notifier
from lrn2.nn_bricks.utils import save_pyc_bz, load_pyc_bz

LOGGER = logging.getLogger(__name__)

class SerializationError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class SerializeParams(object):
    """
    Enables any class providing a list of items of type
    theano.compile.sharedvalue.TensorSharedVariable, named 'params' to get serialized.
    Warning: Every shared variable has to have a distinct name which is the same
    as in the dictionary (variable name = variable.name)!
    """
    def __get_state__(self):

        state = {}

        types = dict([[type(p), p] for p in self.params]).items()
        assert len(types) == 1 and isinstance(types[0][1], SharedVariable), \
                        "Only shared variables are accepted in list 'params'"
        assert not None in [p.name for p in self.params], \
                        "All parameter in self.params have to have a name."

        state.update(dict([[p.name, p.get_value()] for p in self.params]))
        
#         LOGGER.debug("Storing variables: {0}".format(state.keys()))
        return state

    def __set_state__(self, state):
        for key, param in state.items():
            try:
                if isinstance(self.__dict__[key], SharedVariable):
                    self.__dict__[key].set_value(param)
#                     LOGGER.debug("Param loaded: {0}".format(key))
            except KeyError:
                valid_entries = [k for k, value in self.__dict__.items() 
                                 if isinstance(value, SharedVariable)]
                LOGGER.error("Problem at deserializing parameter {0} - ensure that variable name == variable.name.\nStored entries: {1}\nValid entries: {2}".format(key, state.keys(), valid_entries))
                pass


class SerializeByNotifier(object):
    """
    Extends the minimal Serializer by also saving the number of trained epochs.
    """
    def __init__(self):
        SerializeParams.__init__(self)

    def __get_state__(self):
        state = {}

        try:
            to_save = self.notify(Notifier.SAVE)
            for value, name in to_save:
                state[name] = value
        except AttributeError as e:
            LOGGER.error(SerializationError("Problems at serialization: {1}. ".format(self, e) +
                                     "Consider using a smaller Serializer or derive the "
                                     "layer from a class which provides the requested variable."))

        return state

    def __set_state__(self, state):
        try:
            self.notify(Notifier.LOAD, state = state)
        except KeyError as e:
            LOGGER.error(SerializationError("Problems at de-serialization: {1}. ".format(self, e) +
                                     "The stored file does not provide the requested member variable. "
                                     "Do you try to open a file with another serializer than "
                                     "the one the file was saved with?"))
            pass

class SerializeLayers(object):
    def save(self, out_dir, postfix):
        for i in range(len(self.layers)):
            fn = filename_cache_layer("{0}_{1}".format(self.name, self.layers[i].name), out_dir, postfix)
            try:
                self.layers[i].save(fn)
            except AttributeError:
                pass

    def load(self, out_dir, postfix, warn_only = False):
        for i in range(len(self.layers)):
            fn = filename_cache_layer("{0}_{1}".format(self.name, self.layers[i].name), out_dir, postfix)
            try:
                self.layers[i].load(fn, warn_only = warn_only)
            except AttributeError:
                pass

class SerializeLayer(SerializeByNotifier, SerializeParams):
    def __init__(self):
        SerializeParams.__init__(self)
        SerializeByNotifier.__init__(self)

    def save(self, fn):
        state = {}
        LOGGER.debug('Serializing Layer: {0}'.format(fn))
        state.update(SerializeByNotifier.__get_state__(self))
        state.update(SerializeParams.__get_state__(self))
#         LOGGER.debug("params to save:", state.keys())
        save_pyc_bz(state, fn)

    def load(self, fn, warn_only = True):
        LOGGER.debug('Deserializing Layer: {0}'.format(fn))
        try:
            state = load_pyc_bz(fn)
            SerializeByNotifier.__set_state__(self, state)
            SerializeParams.__set_state__(self, state)
        except (EOFError, IOError):
            if warn_only:
                LOGGER.warning('File {0} not found or corrupted.'.format(fn))
            else:
                raise


class SerializeStack(SerializeLayers, SerializeByNotifier):
    def __init__(self):
        SerializeLayers.__init__(self)
        SerializeByNotifier.__init__(self)

    def save(self, fn, postfix="", state = {}):
        #LOGGER.debug('Serializing Stack: {0}'.format(fn))
        out_dir = os.path.dirname(fn)
        SerializeLayers.save(self, out_dir, postfix)
        state.update(SerializeByNotifier.__get_state__(self))
        save_pyc_bz(state, fn)

    def load(self, fn, postfix="", warn_only = True):
        #LOGGER.debug('Deserializing Stack: {0}'.format(fn))
        out_dir = os.path.dirname(fn)
        try:
            state = load_pyc_bz(fn)
            SerializeByNotifier.__set_state__(self, state)
        except (EOFError, IOError):
            if warn_only:
                LOGGER.warning('File {0} not found or corrupted.'.format(fn))
            else:
                raise
        SerializeLayers.load(self, out_dir, postfix, warn_only = warn_only)

def filename_cache_layer(name, out_dir, postfix = ""):
    return os.path.join(out_dir,
                        "{0}_train_cache_stacked_{1}.pyc.bz".format(name, postfix))
