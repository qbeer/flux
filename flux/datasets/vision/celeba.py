r"""
CelebA dataset formating.

"""
from flux.datasets.dataset import Dataset
from flux.backend.data import maybe_download_and_store_google_drive
from flux.util.logging import log_message, log_warning
from flux.backend.globals import DATA_STORE
from flux.processing.vision.util import load_image


from typing import Dict

import os
import tqdm
import random
from tabulate import tabulate

import tensorflow as tf
import numpy as np

TRAIN_PARTITION = 0.8
VAL_PARTITION = 0.1
TEST_PARTITION = 0.1
NUM_ATTR = 40

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class CelebA(Dataset):

    def __init__(self, num_parallel_reads: int=1, force_build=False, force_download=False, shuffle=True):
        file_pair = {"img_align_celeba.zip":"0B7EVK8r0v71pZjFTYXZWM3FlRnM",
                     "list_attr_celeba.txt":"0B7EVK8r0v71pblRyaVFSWGxPY0U"}
        self.root_key = "celebA"
        log_message("Retrieving CelebA data")
        self.keys = maybe_download_and_store_google_drive(file_pair, root_key=self.root_key, force_download=force_download, use_subkeys=False)
        self.selected_attrs = None
        # Extract each batch
        log_message('Extracting CelebA data...')

        self._train_db = None
        self._val_db = None
        self.num_parallel_reads = num_parallel_reads
        # Extract labels
        self.attr2idx: Dict = {}
        self.idx2attr: Dict = {}
        log_message("Extracting CelebA labels first")
        info_files = DATA_STORE[self.keys[1]]
        self._process_attr(info_files)
        if force_build:
            if shuffle:
                random.shuffle(self._img_meta)
            # Build Dataset
            self._build_dataset("train", shuffle)
            self._build_dataset("val", shuffle)

        record_root = os.path.join(self.root_key, "tfrecord")
        train_root = os.path.join(record_root, "train")
        val_root = os.path.join(record_root, "val")

        self.train_fpath = DATA_STORE[train_root]
        self.val_fpath = DATA_STORE[val_root]
        self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(self.train_fpath))
        self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(self.train_fpath))

        log_message("Built Complete")

    @property
    def attr_names(self, ) -> str:
        return " ".join(self._attr_names)

    def _process_attr(self, attr_files):
        labels_raw = open(attr_files, 'r').readlines()

        self._num_examples = int(labels_raw[0])
        self._attr_names = labels_raw[1].strip("\n").split(" ")[:-1]
        if self.selected_attrs is None:
            self.selected_attrs = self._attr_names
        self._img_meta = labels_raw[2:]
        for i, attr_name in enumerate(self._attr_names) :
            self.attr2idx[attr_name] = i # Building bidictionary
            self.idx2attr[i] = attr_name

    def _build_dataset(self, dataset: str, shuffle: bool) -> None:

        if dataset not in ['train', 'val']:
            raise ValueError("Must be building either training or validation dataset")

        record_root = os.path.join(self.root_key, "tfrecord")
        # Open the TFRecordWriter
        if dataset == 'train':
            record_root = os.path.join(record_root, "train")
            data_size = self._num_examples * TRAIN_PARTITION
        else:
            record_root = os.path.join(record_root, "val")
            data_size = self._num_examples * VAL_PARTITION
        # Construct the record reader
        tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(record_root, 'shuffle.tfrecords' if shuffle else "data.tfrecords", force=True))

        # Loop over the data and parse
        errors = 0
        log_message('Building {} dataset...'.format(dataset))
        img_path = DATA_STORE[self.keys[0]]
        for i in tqdm.tqdm(range(int(data_size))):
            img_meta = self._img_meta[i].strip("\n").split(" ")
            file_name = os.path.join(img_path, img_meta[0])
            values = img_meta[1:]
            label = []
            
            for attr_name in self.selected_attrs :
                idx = self.attr2idx[attr_name]
                if values[idx] == '1' :
                    label.append(1.0)
                else :
                    label.append(0.0)
            assert(len(label) == NUM_ATTR) # All labels should have 40 items.  (One hot)
            label = np.array(label, dtype=np.float32)
            # Load the image
            image = load_image(file_name)
            if image is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(file_name, errors))
                continue

            # Add the image data
            feature = {
                "label": _float_feature(label),
                'image_shape': _int64_feature(image.shape),
                'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
            }

            # Write the TF-Record
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tf_record_writer.write(example.SerializeToString())
        
        tf_record_writer.close()
        DATA_STORE.update_hash(record_root)


    def _map_fn(self, serialized_example):

        # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={
                    'label': tf.FixedLenFeature([NUM_ATTR], tf.float32),
                    'image_shape': tf.FixedLenFeature([3], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string),
                    })

        image_shape = features['image_shape']
        image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), image_shape)

        return (features["label"], image)

    

    @property
    def train_db(self):
        if self._train_db is None:
            self._train_db = tf.data.TFRecordDataset(
                self.train_fpath, num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._train_db

    @property
    def val_db(self):
        if self._val_db is None:
            self._val_db = tf.data.TFRecordDataset(
                self.val_fpath, num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._val_db

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                        ['Num Val Examples', self.num_val_examples],],
                        ["Attributes", self.attr_names]))