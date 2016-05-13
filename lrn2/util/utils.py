'''
Created on Jul 8, 2014

@author: Stefan Lattner
'''

import numpy as np
import os
import csv
from numpy import ndarray
from scipy.sparse.csr import csr_matrix
from lrn2.util.gpu_batch import GPU_Batch
from collections import OrderedDict
from cPickle import UnpicklingError
import logging

LOGGER = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_input_filelist(fn, label_generator = None, delimiter = '\t', quotechar ='"'):
    """
    Read a list of filenames from file `fn`. Each line should either
    list only a filename, or alternatively a filename followed by a
    label (separated by one or more whitespaces).

    Parameters
    ----------

    fn : str

        filename

    label_generator : function

        a function that takes a filename as input and returns a label for it

    delimiter : str

        string to use as delimiter in the file

    quotechar : str

        string to use as quote character in the file

    Returns
    -------

    list
        A list of pairs (file, label). Wherever no label was specified for a file, label is None

    """

    files_labels = []

    if label_generator is None:
        label_generator = lambda x: None

    with open(fn, 'rU') as csv_file:
        reader = csv.reader(csv_file, delimiter = delimiter,
                            quotechar = quotechar)
        for row in reader:
            if len(row) == 0:
                continue
            else:
                fn = row[0]
            if len(row) < 2:
                files_labels.append((fn, label_generator(fn)))
            else:
                label = row[1]
                files_labels.append((fn, label))

    return files_labels

def csv_to_dict(fn, delimiter = '\t', quotechar ='"'):
    """
    Reads a csv-file using the first two columns as {key : value} pair
    """
    with open(fn, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter = delimiter,
                            quotechar = quotechar)
        d = OrderedDict((row[0], None if len(row) == 1 else row[1]) for row in reader)

    return d

def dict_to_csv(dict, fn, delimiter = '\t', quotechar = '"'):
    """
    Writes a dictionary to a csv-file, using first two colums for
    {key : value} pair
    """
    with open(fn, 'wb') as csv_file:
        write = csv.writer(csv_file, delimiter = delimiter,
                           quotechar = quotechar, quoting=csv.QUOTE_MINIMAL)
        for key, value in dict.items():
            write.writerow([key, value])

def ensure_ndarray(obj):
    if isinstance(obj, csr_matrix):
        return obj.todense()
    elif isinstance(obj, np.ndarray):
        return obj
    elif type(obj) in (list, tuple):
        return np.array(obj)
    else:
        raise Exception('Do not know how to represent {0} as an ndarray'\
                        .format(type(obj)))

def ensure_gpu_batch(obj):
    if isinstance(obj, GPU_Batch):
        return obj
    else:
        return GPU_Batch(obj, 1)

def ensure_list(obj, no_it = False):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, csr_matrix):
        return obj.todense().tolist()
    elif isinstance(obj, ndarray):
        return obj.tolist()
    else:
        try:
            return list(obj)
        except:
            if no_it:
                return [obj]
            else:
                raise Exception('Do not know how to represent {0} as a list'\
                                .format(type(obj)))

def in_ipynb():
    try:
        from IPython.core.getipython import get_ipython
        cfg = get_ipython().config
        if cfg['IPKernelApp']['parent_appname'] == 'ipython-notebook':
            return True
        else:
            return False
    except (ImportError, NameError, AttributeError):
        return False

def write_to_file(fn, content):
    with open(fn, "a") as myfile:
        myfile.write(content + "\n")

def ensure_file_exists(path_list):
    for config_path in path_list:
        if not os.path.exists(config_path):
            ex = "File {0} not found.".format(config_path)
            raise Exception(ex)

def to_numpy(*args):
    result = []
    for array in args:
        if isinstance(array, ndarray):
            result.append(array)
        elif array is None:
            result.append(None)
        else:
            result.append(np.array(array))

    return result

def extract_col_pair(matrix, i, j):
    first_col = np.array(matrix)[:, i]
    first_col = np.reshape(first_col, (first_col.shape[0], 1))

    second_col = np.array(matrix)[:, j]
    second_col = np.reshape(second_col, (second_col.shape[0], 1))

    return np.hstack((first_col, second_col))

def ndims(x):
    try:
        return len(x.shape)
    except:
        try:
            y=x[0]
        except (IndexError,TypeError,ValueError):
            return 1
        else:
            return ndims(y)+1

def shape(x):
    try:
        return list(x.shape)
    except:
        try:
            y=x[0]
        except (IndexError,TypeError,ValueError):
            return []
        else:
            return [len(x)] + shape(y)

def uniquifier(seq):
    # Removes all double entries of seq
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]