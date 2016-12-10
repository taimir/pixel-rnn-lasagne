import numpy as np
import gzip
import pickle
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)

mnist_path = os.path.join(os.path.dirname(__file__), 'data/mnist.pkl.gz')


def get_data():
    dataset = mnist_path
    import os

    # Download the MNIST dataset if it is not present
    # data_dir, data_file = os.path.split(dataset)
    if not os.path.isfile(dataset):
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        log.info('Downloading MNIST from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)


def load_data():
    get_data()
    """
    Load data
    """
    log.info("Loading data ...")
    with gzip.open(mnist_path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
        train_x, _ = train_set
        train_x = np.reshape(train_x*256, newshape=(train_x.shape[0], 1, 28, 28))
        valid_x, _ = valid_set
        valid_x = np.reshape(valid_x*256, newshape=(valid_x.shape[0], 1, 28, 28))
        test_x, _ = test_set
        test_x = np.reshape(test_x*256, newshape=(test_x.shape[0], 1, 28, 28))

        return {'x_train': np.asarray(train_x, dtype='int32'),
                'x_valid': np.asarray(valid_x, dtype='int32'),
                'x_test': np.asarray(test_x, dtype='int32')}

