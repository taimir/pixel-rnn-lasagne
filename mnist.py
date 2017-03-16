import numpy as np
import gzip
import pickle
import os
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)

mnist_path = os.path.join(os.path.dirname(__file__), 'data/mnist.pkl.gz')


def sample(images):
    return np.random.binomial(1, p=images).astype(np.float32)


def get_data():
    dataset = mnist_path
    # Download the MNIST dataset if it is not present
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
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        train_x, _ = train_set
        train_x = np.reshape(train_x, newshape=(train_x.shape[0], 1, 28, 28))
        train_x = (train_x > 0.5).astype(np.int32)
        valid_x, _ = valid_set
        valid_x = np.reshape(valid_x, newshape=(valid_x.shape[0], 1, 28, 28))
        valid_x = (valid_x > 0.5).astype(np.int32)
        test_x, _ = test_set
        test_x = np.reshape(test_x, newshape=(test_x.shape[0], 1, 28, 28))
        test_x = (test_x > 0.5).astype(np.int32)

        return {'x_train': train_x,
                'x_valid': valid_x,
                'x_test': test_x}
