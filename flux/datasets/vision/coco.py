"""
Utils for parsing the MSCOCO dataset
"""

import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tqdm
from tabulate import tabulate

from flux.backend.data import maybe_download_and_store_zip
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.processing.vision.util import load_image
from flux.util.logging import log_message


def build_fpath_from_image_id(root_filepath: str, image_id: int) -> str:
    return os.path.join(root_filepath, 'COCO_train2014_{0:012d}.jpg'.format(image_id))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class COCOCaptions(object):
    """ MS COCO Caption Dataset

    For now, we host the data locally on one of the CannyLab machines - meaning that we're
    technically going around the rules. Thus - I've added password protection. Make sure that 
    you have the password for the data when downloading.
    """
    def __init__(self, num_parallel_reads: int=1, force_rebuild: bool=False) -> None:

        # Query for the data password
        if not DATA_STORE.is_valid('coco2014/data/train/annotations') or force_rebuild:
            maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip', 'coco2014/data/train/annotations')[0]
        if not DATA_STORE.is_valid('coco2014/data/val/annotations') or force_rebuild:
            maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip', 'coco2014/data/val/annotations')[0]
        if not DATA_STORE.is_valid('coco2014/data/train/images') or force_rebuild:
            maybe_download_and_store_zip('http://images.cocodataset.org/zips/train2014.zip', 'coco2014/data/train/images')
        if not DATA_STORE.is_valid('coco2014/data/val/images') or force_rebuild:
            maybe_download_and_store_zip('http://images.cocodataset.org/zips/val2014.zip', 'coco2014/data/val/images')

        # TODO (davidchan@berkeley.edu) Need to make sure that this works - there could be download issues, but it's hard to say
        self.train_json_key = 'coco2014/data/train/annotations/v2_mscoco_train2014_annotations'
        self.val_json_key = 'coco2014/data/val/annotations/v2_mscoco_val2014_annotations'

        # Now that we have the data, load and parse the JSON files
        need_rebuild_train = force_rebuild
        if not DATA_STORE.is_valid('coco2014/tfrecord/train') or need_rebuild_train:
            need_rebuild_train = True
            with open(DATA_STORE[self.train_json_key], 'r') as annotation_file:
                self.train_json = json.loads(annotation_file.read())
        
        need_rebuild_val = force_rebuild
        if not DATA_STORE.is_valid('coco2014/tfrecord/val') or need_rebuild_val:
            need_rebuild_val = True
            with open(DATA_STORE[self.val_json_key], 'r') as annotation_file:
                self.val_json = json.loads(annotation_file.read())

        # Load the vocab files
        if not DATA_STORE.is_valid('coco2014/captions/dictionary') or force_rebuild:
            self.dictionary = NLPDictionary()
            need_rebuild_train = True
            need_rebuild_val = True
        else:
            self.dictionary = NLPDictionary()
            self.dictionary.load(DATA_STORE['coco2014/captions/dictionary'])

        # Setup some default options for the dataset
        self.max_word_length = 50
        self.max_char_length = 16
        self._val_db = None
        self._train_db = None
        self.num_parallel_reads = num_parallel_reads
        
        # Build the tfrecord dataset from the JSON
        if need_rebuild_train:
            self._build_dataset('train')
        if need_rebuild_val:
            self._build_dataset('val')

        self.train_fpath = DATA_STORE['coco2014/tfrecord/train']
        self.val_fpath = DATA_STORE['coco2014/tfrecord/val']

        # Compute the size of the datasets
        self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['coco2014/tfrecord/train']))
        self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['coco2014/tfrecord/val']))

        # Save the vocab
        with open(DATA_STORE.create_key('coco2014/captions/dictionary', 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dictionary, pkl_file)
            DATA_STORE.update_hash('coco2014/captions/dictionary')

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictionary)

    def _build_dataset(self, dataset: str) -> None:

        if dataset not in ['train', 'val']:
            raise ValueError("Must be building either training or validation dataset")

        # Open the TFRecordWriter
        if dataset == 'train':
            record_root = 'coco2014/tfrecord/train'
            json = self.train_json
            root_fpath = DATA_STORE['coco2014/data/train/images']
        else:
            record_root = 'coco2014/tfrecord/val'
            json = self.val_json
            root_fpath = DATA_STORE['coco2014/data/val/images']

        # Construct the record reader
        tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(record_root, 'data.tfrecords', force=True))

        # Loop over the data and parse
        log_message('Building {} dataset...'.format(dataset))
        for entry in tqdm.tqdm(json['annotations']):
            # Load the image
            image = load_image(build_fpath_from_image_id(root_fpath, entry['image_id']))

            # Parse the caption
            caption_raw = entry['caption']
            caption_dense, caption_len = self.dictionary.dense_parse(caption_raw, word_padding=self.max_word_length, char_padding=self.max_char_length)

            # Add the image data 
            feature = {
                'caption_word_embedding': _int64_feature(np.ravel(caption_dense[0]).astype(np.int64)),
                'caption_char_embedding': _int64_feature(np.ravel(caption_dense[1]).astype(np.int64)),
                'caption_length': _int64_feature([caption_len]),
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
            features={'caption_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                      'caption_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                      'caption_length': tf.FixedLenFeature([1], tf.int64),
                      'image_shape': tf.FixedLenFeature([3], tf.int64),
                      'image': tf.FixedLenFeature([], tf.string),
                      })

        image_shape = features['image_shape']
        image = tf.reshape(tf.decode_raw(features['image'], tf.uint8), [image_shape[0], image_shape[1], image_shape[2]])

        # This tuple is the longest, most terrible thing ever
        return (features['caption_word_embedding'], features['caption_char_embedding'], features['caption_length'], image)

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
                        ['Num Val Examples', self.num_val_examples],
                        ['Word Vocab Size', self.word_vocab_size],
                        ['Char Vocab Size', self.char_vocab_size]]))
