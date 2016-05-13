#!/usr/bin/env python

import logging
import numpy as np

from numpy import ndarray
from lrn2.util.utils import ensure_ndarray
from lrn2.nn_bricks.plot import dummy_tiler

LOGGER = logging.getLogger(__name__)

class Corpus(object):
    """
    Initialize a Corpus object. `file_loader` is a function that
    reads a list of files, and returns a generator object over
    pairs of (piece_id [of type str], data [of type ndarray]).

    Parameters
    ----------

    file_loader : function
        this function gets a pointer to files (e.g. a list of [filename,
        label] pairs) as a parameter and returns a list of instance / label pairs,
        or a generator object which does this (where 'instance' is the raw data
        as loaded from a file).

    viewpoints : array-like
        list of class instantiations, derived from lrn2.data.domain.ViewPoint,
        which define how data should be represented to the net (converts raw data
        to an e.g. binary representation).
        Multiple viewpoints allow for multiple "views" on the data,
        the corpus concatenates data represented in those views to create
        the actual input to the model.

    convolutional : boolean, optional
        tells if data should be shaped in 4 dimensions (necessary for most
        convolutional layers). If True, all viewpoints need an attribute 'shape',
        which defines the 2d shape in convolution. default = False
        
    conv_maps : int, optional
        the number of convolutional maps of the input, default = 1  
    """
    def __init__(self, file_loader, viewpoints = [], convolutional = False,
                 conv_maps = 1):

        self.file_loader = file_loader
        self.viewpoints = viewpoints
        self.data = []
        self.labels = []
        self.ngram_data = None
        self.ngram_size = 0
        self.ngram_labels = None
        self.ngrams_per_set = []
        self.convolutional = convolutional
        self.conv_maps = conv_maps
        self.idxs_shuffled = None

    def add_viewpoint(self, vp):
        """
        Add a ViewPoint object to the corpus. This object will be used
        to extract information from the raw data that is loaded from
        the data files.

        Parameters
        ----------

        vp : ViewPoint
            the viewpoint object to be added. This object must have a
            method `raw_to_repr`, that converts raw data into a
            binary representation of the viewpoint

        """
        self.viewpoints.append(vp)

    def load_files(self, filenames, clear = False):
        """Load data from a list of filenames, and store the data internally.

        Parameters
        ----------

        filenames : dictionary
            a dictionary of filenames to read data from

        labels : dictionary
            a dictionary of labels

        clear : bool
            if True, discard data that was loaded by earlier calls to `load_files`

        """
        if clear:
            self.data = []
            self.labels = []

        for instance_set, label in self.file_loader(filenames):
            vp_repr_data = []

            for vp in self.viewpoints:
                vp_repr_data.append(ensure_ndarray(vp.raw_to_repr(instance_set,
                                                                  label)))
            fn_bin_data = np.hstack(vp_repr_data)

            self.data.append(fn_bin_data)
            self.labels.append(label)


    def set_to_ngram(self, ngram_size = 5, step_width=1):
        """
        Create ngrams from the data that was loaded into the
        corpus. The ngram data is stored as a sparse matrix in
        `self.ngram_data`

        Parameters
        ----------

        ngram_size : int
            the size of the ngrams to create

        """

        self.ngram_size = ngram_size

        data = []
        ngram_labels = []
        ngrams_per_set = []

        for value, label in zip(self.data, self.labels):
            n_ngrams = value.shape[0] - self.ngram_size + 1
            for i in range(0, n_ngrams, step_width):
                a = value[i : i + self.ngram_size,:]
                a = a.reshape((1, -1))
                data.append(a)

            ngrams_created = len(range(0, n_ngrams, step_width))
            ngram_labels.extend([label]*ngrams_created)
            ngrams_per_set.extend([ngrams_created])


        self.ngrams_per_set = np.array(ngrams_per_set)
        self.ngram_data = np.vstack(data)
        self.ngram_labels = np.array(ngram_labels)

        if self.convolutional:
            # Convolutional nets need 4-dimensional input
            width, height = self.check_vps_2d()
            shape_f_conv = [-1, self.conv_maps, width, height]
            self.ngram_data = np.reshape(np.asarray(self.ngram_data), 
                                         shape_f_conv,
                                         order='F')
            
    def check_vps_2d(self):
        last_width = None
        for vp in self.viewpoints:
            assert hasattr(vp, 'shape'), \
              "Please define a property 'shape' in viewpoint '{0}'.".format(vp)
            if last_width == None:
                last_width = vp.shape[1]
            else:
                assert last_width == vp.shape[1], \
                  "For convolution with multiple viewpoints, all shape[1] " + \
                  "have to be equal: {0} != {1}".format(last_width, vp.shape[1])
        return last_width, self.size // last_width // self.conv_maps

    def merge_sets(self):
        """
        Merges the instance sets of the corpus, so that the data of several
        files are considered as one stream for ngram generation
        """
        self.data = np.vstack(self.data)
        self.data = np.reshape(self.data, [1] + list(self.data.shape))
        self.labels = [self.labels[-1]]

    def shuffle_instances(self, idxs = None):
        """
        Shuffles training instances and corresponding labels
        If idxs is given, instances are shuffled according to them
        Returns the shuffled indices
        """
        self.unshuffle_instances()

        LOGGER.debug("Shuffling training instances...")

        if idxs == None:
            idxs = range(self.ngram_data.shape[0])
            np.random.shuffle(idxs)
        else:
            assert len(idxs) == self.ngram_data.shape[0], \
            "Length of (shuffled) idxs array has to be equal length of data instances."

        self.ngram_data = self.ngram_data[idxs, :]
        if self.ngram_labels is not None:
            self.ngram_labels = self.ngram_labels[idxs]

        self.idxs_shuffled = idxs
        return idxs

    def unshuffle_instances(self):
        if self.idxs_shuffled is not None:
            LOGGER.debug("Unshuffling training instances...")
            # Unshuffle
            ngram_data_tmp = np.copy(self.ngram_data)
            self.ngram_data[self.idxs_shuffled, :] = ngram_data_tmp

            if self.ngram_labels is not None:
                ngram_labels_tmp = np.copy(self.ngram_labels)
                self.ngram_labels[self.idxs_shuffled] = ngram_labels_tmp

            self.idxs_shuffled = None

    def limit_instance_count(self, n):
        """
        Randomly chooses n instances (and their labels) while preserving the
        original order of the lists.

        Parameters
        ----------
        n : int
            the resulting number of instances in the corpus

        """
        LOGGER.info("Restricting to {0} random instances (preserve order) ..."\
                .format(n))
        idxs = range(self.ngram_data.shape[0])
        np.random.shuffle(idxs)
        idxs = idxs[:n]
        idxs = np.sort(idxs)
        self.ngram_data = self.ngram_data[idxs, :]

        if self.ngram_labels is not None:
            self.ngram_labels = np.asarray([self.ngram_labels[i] for i in idxs])

    def limit_instance_range(self, idx_first, idx_last):
        """
        Chooses n rows of the training set while preserving the order.

        Parameters
        ----------
        idx_first : int
            The first index of the subset (inclusive)
        idx_last : int
            The last index of the subset (exclusive)

        """
        LOGGER.info("Taking instance {0} to {1} ...".format(idx_first, idx_last))

        self.ngram_data = self.ngram_data[idx_first : idx_last, :]
        if self.ngram_labels:
            self.ngram_labels = self.ngram_labels[idx_first : idx_last]

    def filter_instances(self, filt):
        """
        Filters training instances according to a filter function.

        Parameters
        ----------
        filter : function
            a function of the form f(instance, label) which returns True if
            the entry should be filtered, False otherwise

        """

        N1 = len(self.data)
        self.data, self.labels = zip(*[(i_set, label) for i_set, label
                                       in zip(self.data, self.labels)
                                       if filt(i_set, label)])
        N2 = len(self.data)
        LOGGER.info("Filtered {0} instances.".format(N2 - N1))

    def ngrams_with_label(self, label):
        """
        Returns all ngrams with specified label
        """
        idxs = np.where(np.asarray(self.ngram_labels) == label)[0]
        return self.ngram_data[idxs]

    @property
    def n_dimensions(self):
        """
        The number of dimensions of an ngram. This equals the sum of
        the sizes of the viewpoints times the ngram size.

        """
        if self.ngram_data is None:
            LOGGER.error('no ngram data found; need to call `load_files` and `set_to_ngram` first')
        else:
            return self.ngram_data.shape[1]

    @property
    def n_ngrams(self):
        """
        The number of ngram instances currently loaded.

        """

        if len(self.ngram_data) == 0:
            LOGGER.error('no ngram data found; need to call `load_files` and `set_to_ngram` first')
        else:
            return self.ngram_data.shape[0]

    @property
    def size(self):
        """
        The size of one instance
        """
        return np.sum([v.size for v in self.viewpoints])
    
    @property
    def shape(self):
        """
        The shape of the whole data
        """
        if self.convolutional:
            width, height = self.check_vps_2d()
            return (self.n_ngrams, self.conv_maps, width, height)
        else:
            return (self.n_ngrams, self.size)

    def tile_ngram_data(self, data):
        """
        Tile ngram data according to the viewpoints and current ngram
        size of the corpus.

        Parameters
        ----------

        data : ndarray
            the data to tile

        Returns
        -------

        list
            a list(instances) of lists(viewpoints) of 2D ndarrays (tile)

        """

        assert isinstance(data, ndarray) # input data has to be an ndarray

        if len(data.shape) == 1:
            data = np.asarray([data])

        if len(data.shape) == 4:
            # convolutional net: four input dimensions
            tiles = []
            for ngram in data:
                for map in ngram:
                    vps = []
                    curr_idx = 0
                    for vp in self.viewpoints:
                        curr_vp = map[:,curr_idx:curr_idx + vp.shape[1]]
                        tr = vp.repr_to_visual(curr_vp)
                        vps.append(tr)
                        curr_idx += vp.shape[1]
                    tiles.append(vps)

            return tiles
        elif len(data.shape) == 2:
            # standard net: two input dimensions
            try:
                ngram_shape = (self.ngram_size, self.size)
                tiles = []

                for i in range(data.shape[0]):
                    ngram = np.reshape(data[i, :], ngram_shape)
                    ngram_i_tiles = []
                    start = 0

                    for vp in self.viewpoints:
                        end = start + vp.size
                        ngram_i_tiles.append(vp.repr_to_visual(ngram[:,start:end]))
                        start = end

                    tiles.append(ngram_i_tiles)
                return tiles
            except ValueError:
                # If something goes wrong, use the dummy tiler.
                return dummy_tiler(data)
        else:
            raise ValueError("Only two or four input dimensions can be tiled, "
                             "but we've got {0}.".format(len(data.shape)))
