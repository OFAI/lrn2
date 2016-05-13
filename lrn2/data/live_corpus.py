'''
Created on May 22, 2015

@author: Stefan Lattner
'''

import os
import sys
import numpy as np
from lrn2.util.utils import ensure_ndarray, shape
import logging
from numpy import ndarray
from lrn2.nn_bricks.plot import dummy_tiler
from _functools import partial
from lrn2.nn_bricks.utils import fx
import glob
from lrn2.nn_bricks.notifier import Notifier
from __builtin__ import KeyError

LOGGER = logging.getLogger(__name__)

class LiveCorpus(object):
    '''
    When working with a big amount of data, this class may be useful by loading
    mini-batch by mini-batch in the memory instead of the whole dataset.
    
    The LiveCorpus converts raw input files into a desired input representation
    (using the viewpoint's raw_to_repr() method) and stores that representation
    in temporary files. From those files, it loads batch by batch and provides
    them if requested via a callback function (get_data_callback).
    
    After initialization use
    corpus.load_files(datafiles, folder_tmp, rebuild)
    '''
    def __init__(self, file_loader, viewpoints, convolutional = False, 
                 ngram_size = None, step_width = 1, shuffle = False,
                 use_labels = False, limit_instances = None,
                 linearized = True, verbose = False, conv_maps = 1, **kwargs):
        """
        Parameters
        ----------
        
        file_loader : function
            function which takes a list of (file, label) pairs and returns
            a list of instance / label pairs, or a generator object which does
            this (where 'instance' is the raw data as loaded from a file).
        
        viewpoints : list of viewpoint instantiations
            list of class instantiations, derived from lrn2.data.domain.ViewPoint,
            which define how data should be represented to the net (converts raw data
            to an e.g. binary representation).
            Multiple viewpoints allow for multiple "views" on the data,
            the corpus concatenates data represented in those views to create
            the actual input to the model.

        convolutional : boolean, optional
            tells if data should be shaped in 4 dimensions (necessary for most
            convolutional layers). Note that the LiveCorpus has a slightly
            different way to shape convolutional data than the Corpus.
            The first two dimensions are defined by the batch_size and the
            parameter conv_maps, as in the Corpus class. The third dimension,
            however, is defined by the ngram_size (in the config file).
            The fourth dimension is defined by the size property of the viewpoint.
            default = False 
            
        ngram_size : int
            size of one instance (use 1 for images and > 1 for e.g. time series)
            
        step_width : int
            step width over the data between ngrams
            
        shuffle : boolean
            shuffle instances
            
        use_labels : boolean
            data can be labeled according to its source file (not well tested)
            
        limit_instances : int
            limit instance count to specific number. If shuffle = True, subset
            will change at each run
            
        linearized : boolean
            if data is stored in a linearized fashion
            
        verbose : boolean
            causes verbose console output
            
        conv_maps : int
            number of convolutional input maps
        """
        self.use_labels = use_labels
        self.step_width_ = step_width
        self.shuffle_ = shuffle
        self.file_loader = file_loader
        self.viewpoints = viewpoints
        self.filenames = []
        self.files_open = []
        self.files_open_labels = []
        self.sizes = [0]
        self.ngram_size = ngram_size
        self.idx = None
        self.convolutional = convolutional
        self.verbose = verbose
        self.last_perc = -1
        self.limit_instances_ = limit_instances
        self.linearized = linearized
        try:
            self.preprocess = kwargs["preprocess"]
        except KeyError:
            self.preprocess = lambda x : x
        
    def load_files(self, filenames, folder_tmp = "/tmp", rebuild = True):
        """
        To be used after initialization. Prepares data representation using
        given viewpoints and stores them in temporary files.
        
        Parameters
        ----------
        
        filenames : list
            A list of pairs (file, label). Wherever no label was specified for 
            a file, label is None
            
        folder_tmp : string
            A folder to store the data for fast reading. Preferable on an SSD HD.
            
        rebuild : boolean
            If files should be rebuilt.
        """
        pointers = os.path.join(folder_tmp, "files_live_corpus.npy")
        if rebuild or not os.path.isfile(pointers):
            self.remove_tmp_files(folder_tmp)
            fns = self.convert_files(filenames, folder_tmp)
            fns = self.merge_files(fns, merge=100, axis=0)
            np.save(pointers, fns)
        else:
            fns = np.load(pointers)
        
        self.open_files(fns, folder_tmp)

    def merge_files(self, files, merge = 100, axis=0):
        """
        Merges *.npy files by concatenating *merge* files and save it with
        filename of the last file in a group. Returns an updated filelist.
        """
        assert len(files) > 0, "No files to merge."
        LOGGER.info("merging files (in batches of {0}, concat axis {1})".format(merge, axis))
        merged = None
        count = 0
        files_res = []
        for f in files[:]:
            new_file = np.load(f)
            try:
                if merged is None:
                    merged = new_file
                else:
                    merged = np.concatenate((merged, new_file), axis=axis)
    
                count += 1
                if count > merge:
                    np.save(f, merged)
                    files_res.append(f)
                    merged = None
                    count = 0
            except Exception as e:
                LOGGER.error(e)

        if count > 0:
            np.save(f, merged)
            files_res.append(f)
            merged = None
            count = 0

        return files_res

    def convert_files(self, filenames, folder_tmp):
        """
        Converts files in raw format to files in input representation using the
        viewpoints 'raw_to_repr' method.
        """
        LOGGER.info("converting files...")
        files_all = []
        for i, (instance_set, label) in enumerate(self.file_loader(filenames)):
            try:
                fn = os.path.basename(filenames[i][0].split('\t')[0])
                fn = fn.split('.')[0]
            except Exception as e:
                if isinstance(e, TypeError):
                    fn = str(label).split('.')[0]
                else:
                    LOGGER.exception(e)
                    fn = str(label).split('.')[0]

            file_path = os.path.join(folder_tmp, fn + ".lrn2.npy")
            file_path_labels = os.path.join(folder_tmp, fn + "_labels.lrn2.npy")

            if self.verbose:
                LOGGER.info("processing file {0}...".format(file_path))
            if not os.path.isfile(file_path):
                vp_bin_data = []

                for vp in self.viewpoints:
                    curr_data = ensure_ndarray(vp.raw_to_repr(instance_set,
                                                                     label))
                    vp_bin_data.append(curr_data)

                if len(self.viewpoints) > 1:
                    fn_repr_data = np.hstack(vp_bin_data)
                else:
                    fn_repr_data = vp_bin_data[0]

                if self.use_labels:
                    labels = [label] * len(fn_repr_data)

                if self.verbose:
                    LOGGER.info("saving file {0}...".format(file_path))

                np.save(file_path, fn_repr_data)

                if self.use_labels:
                    np.save(file_path_labels, labels)

                files_all.append(file_path)
            else:
                if self.verbose:
                    LOGGER.info("Skipping conversion - file {0} already exists."
                                .format(file_path))
                files_all.append(file_path)

        return files_all

    def open_files(self, filenames, folder_tmp):
        """
        Open certain files of a folder
        """
        for f in filenames:
            fn = os.path.basename(f).split('.')[0]
            file_path = os.path.join(folder_tmp, fn + ".lrn2.npy")
            file_path_labels = os.path.join(folder_tmp, fn + "_labels.lrn2.npy")
            if self.verbose:
                LOGGER.info("Opening source file {0}".format(file_path))
            self.files_open.append(np.load(file_path, mmap_mode='r'))
            if self.use_labels:
                self.files_open_labels.append(np.load(file_path_labels, mmap_mode='r'))
            if self.verbose:
                LOGGER.debug("File {0} has shape {1}.".format(f, self.files_open[-1].shape))

        self.sizes = [len(f) for f in self.files_open]
        
        LOGGER.info("Sum and sizes of open files: {0},{1}".format(sum(self.sizes), self.sizes))
        
        if self.ngram_size is not None:
            self.reset_idx()

    def remove_tmp_files(self, folder_tmp):
        files_to_delete = glob.glob(folder_tmp + "/*.lrn2.npy")
        [os.remove(f) for f in files_to_delete]

    def open_files_all(self, folder_tmp):
        """
        Open all files of a folder
        """
        files_to_open = glob.glob(folder_tmp + "/*.lrn2.npy")
        [files_to_open.remove(x) for x in glob.glob(folder_tmp + "/*_labels.lrn2.npy")]

        self.open_files(files_to_open, folder_tmp)
        
    def set_to_ngram(self, ngram_size = 5, step_width=1):
        self.ngram_size = ngram_size
        self.step_width = step_width
        self.reset_idx()

    def get_block(self, from_idx, to_idx, kind="data"):
        assert kind in ("data", "labels"), "'kind' has to be 'data' or 'labels'"

        data_fun = self.files_open if kind is 'data' else self.files_open_labels

        count = 0
        sizes = self.sizes
        while from_idx >= sum(sizes[:count+1]):
            count += 1

        offset = sum(sizes[:count])
        offset2 = sum(sizes[:count+1])

        if to_idx > offset2:
            if count+1 > len(sizes):
                raise IndexError("Index out of bounds: {0}".format(to_idx))
            else:
#                 print "merge:", count, from_idx, to_idx, offset, offset2
#                 print "shape1 = ", shape(data_fun[count])
#                 print "shape2 = ", shape(data_fun[count+1])
                block = np.r_[data_fun[count][from_idx - offset:],
                              data_fun[count+1][:to_idx - offset2]]
        else:
            block = data_fun[count][from_idx - offset : to_idx - offset]

        if self.convolutional:
            if len(shape(block)) > 2:
                return np.swapaxes(block, 0, 1)
            return np.reshape(block, (self.conv_maps, block.shape[0], 
                                      block.shape[1]))

#         print "block shape = ", shape(block)
        return np.reshape(block, (-1,)).astype(fx)

    def get_block_size(self):
        assert self.ngram_size is not None, "Call corpus.set_to_ngram(ngram_size, step_width) before."
        if self.convolutional:
            return self.ngram_size
        else:
            if self.linearized:
                return self.ngram_size * self.size # If stored linearized
            else:
                return self.ngram_size

    def get_data_callback(self, batch_nr, batch_size = 10, kind = "data"):
        assert kind in ("data", "labels"), "'kind' has to be 'data' or 'labels'"
        assert len(self.files_open) > 0, "Call 'open_files()' before accessing data."

        n_batches = self.n_batches(batch_size)

        if self.verbose:
            LOGGER.debug("Batch {0}/{1}".format(batch_nr, n_batches))

        block_size = self.get_block_size()
        assert n_batches > 0, "Instance count ({0}) has to be > batch size * step_width ({1} * {2}). Hint: You need to define a batch-size for the max-pooling layer." \
                                .format(self.instance_count, batch_size, self.step_width)

        ngrams = []

        if self.idx is None:
            self.reset_idx()

        if batch_nr <= n_batches:
            i = batch_nr * batch_size
            while len(ngrams) < batch_size and i < len(self.idx):
                ngrams.append(self.preprocess(self.get_block(self.idx[i],
                                                             self.idx[i] + block_size,
                                                             kind = kind)))
                i += 1
        else:
            self.last_perc = -1
            return None

        if self.convolutional and len(ngrams) < batch_size:
            # Return only full batches
            return []
        return ensure_ndarray(ngrams)

    @property
    def ngram_data(self):
        # Compatibility method to Corpus
        return self.get_data(labels = False)
        
    def get_data(self, labels = True):
        ngrams = []
        labels = []
        block_size = self.get_block_size()
#         print "get data block size ", block_size
#         print "get data idx size ", len(self.idx)
        for idx in self.idx:
            ngrams.append(self.get_block(idx, idx + block_size))
            if self.use_labels:
                labels.append(self.get_block(idx, idx + block_size, kind = 'labels'))

        if self.use_labels and labels:
            return [ngrams, labels]

        result = np.vstack(ngrams)
        return result.reshape((result.shape[0],-1,result.shape[1],result.shape[2]))


    def reset_idx(self, idx = None):
        block_size = self.get_block_size()
        to = self.instance_count
        self.idx = np.asarray(xrange(0, to - block_size + 1,
                                     self.step_width))
        if idx is not None:
            assert len(self.idx) == len(idx), "Given indices not of equal length to existing indices."
            self.idx = idx
        else:
            if self.shuffle:
                np.random.shuffle(self.idx)
            
        if self.limit_instances_ is not None:
            self.idx = self.idx[:self.limit_instances_]

    @property
    def limit_instances(self):
        return self.limit_instances_
    
    @limit_instances.setter
    def limit_instances(self, value):
        self.limit_instances_ = value
        self.reset_idx()
    
    def shuffle_instances(self):
        self.shuffle = True
        
    def unshuffle_instances(self):
        self.shuffle = False
        
    @property
    def shuffle(self):
        return self.shuffle_

    @shuffle.setter
    def shuffle(self, value):
        if self.shuffle_ is not value:
            self.shuffle_ = value
            self.reset_idx()

    @property
    def step_width(self):
        return self.step_width_

    @step_width.setter
    def step_width(self, value):
        if self.step_width_ is not value:
            self.step_width_ = value
            self.reset_idx()

    def n_batches(self, batch_size):
        # Number of batches in corpus
        return int(self.instance_count / batch_size / self.step_width)

    def get_data_iterator(self, batch_size = 500):
        self.idx = None
        return LiveCorpusIterator(self, batch_size = batch_size)

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

        assert isinstance(data, ndarray), "Input data has to be of type ndarray."

        if len(data.shape) == 1:
            data = np.asarray([data])
            
        if len(data.shape) == 4:
#             if sum([vp.size for vp in self.viewpoints]) != data.shape[3]:
#                         return dummy_tiler(data)
            # convolutional net: four input dimensions
            tiles = []
            for ngram in data:
                for map in ngram:
                    vps = []
                    curr_idx = 0
                    for vp in self.viewpoints:
                        curr_vp = map[:,curr_idx:curr_idx + vp.size]
                        tr = vp.repr_to_visual(curr_vp)
                        vps.append(tr)
                        curr_idx += vp.size
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

    @property
    def n_dimensions(self):
        """
        The number of dimensions of an ngram. This equals the sum of
        the sizes of the viewpoints times the ngram size.
        """
        return self.size * self.ngram_size

    @property
    def instance_count(self):
        return sum(self.sizes)

    @property
    def size(self):
        return sum([vp.size for vp in self.viewpoints])


class LiveCorpusIterator(object):
    '''
    Provides an iterator for the LiveCorpus
    '''
    def __init__(self, live_corpus, batch_size = 500):
        self.corpus = live_corpus
        self.batch_size = batch_size

        self.count = 0
        self.corpus_callback = partial(live_corpus.get_data_callback,
                                       batch_size = self.batch_size)

    def __iter__(self):
        return self

    def __len__(self):
        return self.corpus.instance_count / self.batch_size

    def seek(self, index):
        self.count = index

    def next(self):
        self.count += 1
        try:
            next_batch = self.corpus_callback(self.count-1)
            if next_batch == None or len(next_batch) == 0:
                raise StopIteration
            else:
                return next_batch
        except IndexError:
            raise StopIteration


class ProjectFeatureSpaceIterator(object):
    '''
    Provides an iterator for projecting through a net
    '''
    def __init__(self, net, concat_batches = False, meta_batches = 1):
        self.concat_batches = concat_batches
        self.meta_batches = meta_batches
        self.net = net
        self.count = 0

    def __iter__(self):
        return self

    def __len__(self):
        return None

    def seek(self, index):
        self.count = index

    def next(self):
        out = []
        for _ in range(self.meta_batches):
            self.count += 1
            try:
                next_batch = self.net.notify(Notifier.GET_DATA, self.count-1)[0]
                if next_batch is None or len(shape(next_batch)) < 2:
                    raise StopIteration
                else:
                    out.extend(self.net.out(next_batch))
            except IndexError:
                raise StopIteration
        if self.concat_batches:
            if len(shape(out)) > 3:
                return np.swapaxes(np.hstack(out), 0, 1)
            else:
                return np.vstack(out)
        else:
            return out
