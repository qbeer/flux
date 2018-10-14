"""
Parsing code for the MNIST dataset
"""

import numpy as np
import tensorflow as tf

import gzip
import pickle
import tqdm

from flux.util.logging import log_message
from flux.backend.data import maybe_download_and_store_single_file
from flux.backend.globals import DATA_STORE


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with gzip.open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_images, unused
        rows = read32(f)
        cols = read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file')
        if rows != 28 or cols != 28:
            raise ValueError('Invalid MNIST image file: Expected 28x28 images')


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with gzip.open(filename, 'rb') as f:
        magic = read32(f)
        read32(f)  # num_items, unused
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file')


def build_dataset(image_key: str, label_key: str, one_hot: bool=False):
    with gzip.open(DATA_STORE[image_key], 'rb') as image_file:
        with gzip.open(DATA_STORE[label_key], 'rb') as label_file:

            read32(image_file)
            num_images = read32(image_file)
            rows = read32(image_file)
            cols = read32(image_file)

            read32(label_file)
            num_labels = read32(label_file)

            if num_images != num_labels:
                raise ValueError('Invalid MNIST image file: Num_Images != Num Labels')

            img_shape = rows * cols

            img_array = np.zeros(shape=(num_images, img_shape), dtype=np.float32)
            if one_hot:
                label_array = np.zeros(shape=(num_images, 10), dtype=np.int32)
            else:
                label_array = np.zeros(shape=(num_images), dtype=np.int32)

            for idx in tqdm.tqdm(range(num_images)):
                # Read an image
                image = np.frombuffer(image_file.read(img_shape), dtype=np.uint8).astype(np.float32) / 255.0
                # Read a label
                label = np.frombuffer(label_file.read(1), dtype=np.uint8).astype(np.int32)
                img_array[idx] = image
                label_array[idx] = np.eye(10)[label] if one_hot else label

    return (img_array, label_array)


class MNIST():
    def __init__(self, one_hot: bool=False, force_rebuild: bool=False, nohashcheck: bool=True) -> None:

        # Download the MNIST data
        self.train_images_key = maybe_download_and_store_single_file(url='http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', key='mnist/train_images')
        self.train_labels_key = maybe_download_and_store_single_file(url='http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', key='mnist/train_labels')
        self.test_images_key = maybe_download_and_store_single_file(url='http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', key='mnist/test_images')
        self.test_labels_key = maybe_download_and_store_single_file(url='http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', key='mnist/test_labels')

        # Build the dataset
        check_image_file_header(DATA_STORE[self.train_images_key])
        check_labels_file_header(DATA_STORE[self.train_labels_key])
        check_image_file_header(DATA_STORE[self.test_images_key])
        check_labels_file_header(DATA_STORE[self.test_labels_key])

        # Decode the images
        if not DATA_STORE.is_valid('mnist/pickle') or force_rebuild:
            log_message('Extracting Training Images...')
            self.train_images, self.train_labels = build_dataset(self.train_images_key, self.train_labels_key, one_hot)
            log_message('Extracting Test Images...')
            self.test_images, self.test_labels = build_dataset(self.test_images_key, self.test_labels_key, one_hot)

            pickle_dict = {
                'train_im': self.train_images,
                'train_lb': self.train_labels,
                'test_im': self.test_images,
                'test_lb': self.test_labels,
            }

            with open(DATA_STORE.create_key('mnist/pickle', 'mnist.pkl', force=True), 'wb') as pkl_file:
                pickle.dump(pickle_dict, pkl_file)
            DATA_STORE.update_hash('mnist/pickle')
        else:
            with open(DATA_STORE['mnist/pickle'], 'rb') as pkl_file:
                pickle_dict = pickle.load(pkl_file)
                self.train_images = pickle_dict['train_im']
                self.test_images = pickle_dict['test_im']
                self.train_labels = pickle_dict['train_lb']
                self.test_labels = pickle_dict['test_lb']

    @property
    def train_db(self):
        return tf.data.Dataset.from_tensor_slices((self.train_images, self.train_labels))

    @property
    def test_db(self):
        return tf.data.Dataset.from_tensor_slices((self.test_images, self.test_labels))
