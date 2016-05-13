import os
import gzip
import cPickle
import logging

LOGGER = logging.getLogger(__name__)

class MNISTInterface(object):
    def get_image_from_raw_data(self, data):
        return data

def load_mnist_files(filenames="./data/mnist/mnist.pkl.gz", subsets = ("train", "valid", "test")):

    if not os.path.exists(filenames):
        download_mnist(filenames)

    for instance, label in load_data(filenames, subsets):
        yield (instance, label)

def download_mnist(fn):

    import urllib
    origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
    LOGGER.info('Downloading data from %s' % origin)
    location = os.path.dirname(fn)
    if not os.path.exists(location):
        try:
            os.makedirs(location)
        except OSError as e:
            if e.errno == 13:
                LOGGER.error('No permission to create directory {0}'.format(location))
            else:
                raise e

    urllib.urlretrieve(origin, fn)


def load_data(dataset, subsets = ("train", "valid", "test")):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    LOGGER.info('... loading MNIST data')

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    sets = {"train" : train_set, "valid" : valid_set, "test" : test_set}

    for s in subsets:
        instances, labels = sets[s]
        for i in range(len(instances)):
            yield (instances[i], labels[i])

