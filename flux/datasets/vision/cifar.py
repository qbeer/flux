"""
Parsing code for the CIFAR-10 and CIFAR-100 datasets
"""
import pickle
import numpy as np

from typing import List

from flux.util.logging import log_message
from flux.backend.data import maybe_download_and_store_tar
from flux.backend.globals import DATA_STORE


class Cifar10:
    """Class for the CIFAR-10 dataset
    http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """

    def __init__(self, one_hot: bool=False, force_rebuild: bool=False, nohashcheck: bool=True) -> None:

        self.one_hot = one_hot
        self.keys = maybe_download_and_store_tar(url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', root_key='cifar-10')

        # Extract each batch
        log_message('Extracting CIFAR-10 data...')
        for i in range(1, 6):
            fpath = DATA_STORE['cifar-10/cifar-10-batches-py/data_batch_{}'.format(str(i))]
            with open(fpath, 'rb') as f:
                d = pickle.load(f, encoding='latin1')
                data = np.array(d["data"])
                labels = np.array(d["labels"])
            if i == 1:
                self.X_train: np.ndarray = data
                self.Y_train: np.ndarray = labels
            else:
                self.X_train = np.concatenate([self.X_train, data], axis=0)
                self.Y_train = np.concatenate([self.Y_train, labels], axis=0)

        with open(DATA_STORE['cifar-10/cifar-10-batches-py/test_batch'], 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            self.X_test: np.ndarray = np.array(d["data"])
            self.Y_test: np.ndarray = np.array(d["labels"])

        # Normalize and reshape the training and test images so that they lie between 0 and 1
        # as well as are in the right shape
        self.X_train = np.dstack((self.X_train[:, :1024], self.X_train[:, 1024:2048],
                                  self.X_train[:, 2048:])) / 255.
        self.X_train = np.reshape(self.X_train, [-1, 32, 32, 3])
        self.X_test = np.dstack((self.X_test[:, :1024], self.X_test[:, 1024:2048],
                                 self.X_test[:, 2048:])) / 255.
        self.X_test = np.reshape(self.X_test, [-1, 32, 32, 3])

        if self.one_hot:
            self.Y_train = np.eye(10)[self.Y_train]
            self.Y_test = np.eye(10)[self.Y_test]

    @property
    def train_images(self) -> np.ndarray:
        return self.X_train

    @property
    def test_images(self) -> np.ndarray:
        return self.X_test

    @property
    def train_labels(self) -> np.ndarray:
        return self.Y_train

    @property
    def test_labels(self) -> np.ndarray:
        return self.Y_test

    @property
    def images(self):
        return np.concatenate([self.X_train, self.X_test])

    @property
    def labels(self):
        return np.concatenate([self.Y_train, self.Y_test])


class Cifar100:
    """Class for the CIFAR-10 dataset
    http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """

    def __init__(self, one_hot: bool=False, force_rebuild: bool=False, nohashcheck: bool=True) -> None:

        self.one_hot = one_hot
        self.keys = maybe_download_and_store_tar(url='http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', root_key='cifar-100')

        # Extract each batch
        log_message('Extracting CIFAR-100 data...')
        with open(DATA_STORE['cifar-100/cifar-100-python/train'], 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            self.X_train: np.ndarray = np.array(d["data"])
            self.Y_train: np.ndarray = np.array(d["fine_labels"])
            self.Y_train_coarse: np.ndarray = np.array(d["coarse_labels"])
        with open(DATA_STORE['cifar-100/cifar-100-python/test'], 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            self.X_test: np.ndarray = np.array(d["data"])
            self.Y_test: np.ndarray = np.array(d["fine_labels"])
            self.Y_test_coarse: np.ndarray = np.array(d["coarse_labels"])
        with open(DATA_STORE['cifar-100/cifar-100-python/meta'], 'rb') as f:
            d = pickle.load(f, encoding='latin1')
            self.fine_label_names: List[str] = d["fine_label_names"]
            self.coarse_label_names: List[str] = d["coarse_label_names"]

        # Normalize and reshape the training and test images so that they lie between 0 and 1
        # as well as are in the right shape
        self.X_train = np.dstack((self.X_train[:, :1024], self.X_train[:, 1024:2048],
                                  self.X_train[:, 2048:])) / 255.
        self.X_train = np.reshape(self.X_train, [-1, 32, 32, 3])
        self.X_test = np.dstack((self.X_test[:, :1024], self.X_test[:, 1024:2048],
                                 self.X_test[:, 2048:])) / 255.
        self.X_test = np.reshape(self.X_test, [-1, 32, 32, 3])

        if self.one_hot:
            self.Y_train = np.eye(100)[self.Y_train]
            self.Y_test = np.eye(100)[self.Y_test]

    @property
    def train_images(self) -> np.ndarray:
        return self.X_train

    @property
    def test_images(self) -> np.ndarray:
        return self.X_test

    @property
    def train_labels(self) -> np.ndarray:
        return self.Y_train

    @property
    def test_labels(self) -> np.ndarray:
        return self.Y_test

    @property
    def images(self):
        return np.concatenate([self.X_train, self.X_test])

    @property
    def labels(self):
        return np.concatenate([self.Y_train, self.Y_test])
