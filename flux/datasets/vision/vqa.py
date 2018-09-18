"""
Utils for parsing the MSCOCO/VQA dataset
"""

import json
import os
import pickle

import numpy as np
import tensorflow as tf
import tqdm
from tabulate import tabulate

from typing import Sequence, Dict

from flux.backend.data import maybe_download_and_store_zip
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.processing.vision.util import load_image, encode_jpeg
from flux.util.logging import log_message, log_warning
from flux.datasets.dataset import Dataset

def build_fpath_from_image_id(root_filepath: str, image_id: int, dataset: str) -> str:
    return os.path.join(root_filepath, '{}2014'.format(dataset), 'COCO_{0}2014_{1:012d}.jpg'.format(dataset, image_id))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class VQA(Dataset):
    """ VQA Caption Dataset Downloader
    """
    def __init__(self, num_parallel_reads: int=1, force_rebuild: bool=False, ignore_hashes=False, image_shape: Sequence[int] = [224,224]) -> None:

        self.image_resize_shape = image_shape

        # Get all of the necessary data
        self.train_a_json_key = maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Annotations_Train_mscoco.zip', 'coco2014/data/train/annotations')[0]
        self.val_a_json_key = maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Annotations_Val_mscoco.zip', 'coco2014/data/val/annotations')[0]
        self.train_q_json_key = maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Questions_Train_mscoco.zip', 'coco2014/data/train/questions')[0]
        self.val_q_json_key = maybe_download_and_store_zip('http://visualqa.org/data/mscoco/vqa/v2_Questions_Val_mscoco.zip', 'coco2014/data/val/questions')[0]
        maybe_download_and_store_zip('http://images.cocodataset.org/zips/train2014.zip', 'coco2014/data/train/images', use_subkeys=False)
        maybe_download_and_store_zip('http://images.cocodataset.org/zips/val2014.zip', 'coco2014/data/val/images', use_subkeys=False)

        # Compute the size of the datasets
        self.num_train_examples = 443757 #sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['vqa/tfrecord/train']))
        self.num_val_examples = 214654 #sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['vqa/tfrecord/val']))

        # Now that we have the data, load and parse the JSON files
        need_rebuild_train = force_rebuild
        if not ignore_hashes and (need_rebuild_train or not DATA_STORE.is_valid('vqa/tfrecord/train')):
            log_message('Need to rebuild training data. Loading JSON annotations.')
            need_rebuild_train = True
            with open(DATA_STORE[self.train_a_json_key], 'r') as annotation_file:
                self.train_a_json = json.loads(annotation_file.read())
            with open(DATA_STORE[self.train_q_json_key], 'r') as annotation_file:
                self.train_q_json = json.loads(annotation_file.read())
        
        need_rebuild_val = force_rebuild
        if not ignore_hashes and (need_rebuild_val or not DATA_STORE.is_valid('vqa/tfrecord/val')):
            log_message('Need to rebuild validation data. Loading JSON annotations.')
            need_rebuild_val = True
            with open(DATA_STORE[self.val_a_json_key], 'r') as annotation_file:
                self.val_a_json = json.loads(annotation_file.read())
            with open(DATA_STORE[self.val_q_json_key], 'r') as annotation_file:
                self.val_q_json = json.loads(annotation_file.read())

        # Load the vocab files
        if not ignore_hashes and (force_rebuild or not DATA_STORE.is_valid('vqa/dictionary')):
            self.dictionary = NLPDictionary()
            need_rebuild_train = True
            need_rebuild_val = True
        else:
            with open(DATA_STORE['vqa/dictionary'],'rb') as dict_file:
                self.dictionary = pickle.load(dict_file)

        if not ignore_hashes and (force_rebuild or not DATA_STORE.is_valid('vqa/class_map')):
            self.class_map: Dict[str, int] = {}
            need_rebuild_train = True
            need_rebuild_val = True
        else:
            with open(DATA_STORE['vqa/class_map'],'rb') as class_map_file:
                self.class_map_file = pickle.load(class_map_file)

        # Setup some default options for the dataset
        self.max_word_length = 50
        self.max_char_length = 16
        self._val_db = None
        self._train_db = None
        self.num_parallel_reads = num_parallel_reads
        self.sample_triple = None
        # Build the tfrecord dataset from the JSON
        if need_rebuild_train:
            self._build_dataset('train')
        if need_rebuild_val:
            self._build_dataset('val')

        self.train_fpath = DATA_STORE['vqa/tfrecord/train']
        self.val_fpath = DATA_STORE['vqa/tfrecord/val']

        # Save the vocab
        with open(DATA_STORE.create_key('vqa/dictionary', 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dictionary, pkl_file)
            DATA_STORE.update_hash('vqa/dictionary')
        with open(DATA_STORE.create_key('vqa/class_map', 'class_map.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.class_map, pkl_file)
            DATA_STORE.update_hash('vqa/class_map')

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictionary)

    def _build_dataset(self, dataset: str) -> None:

        # Open the TFRecordWriter
        if dataset == 'train':
            record_root = 'vqa/tfrecord/train'
            json_a = self.train_a_json
            json_q = self.train_q_json
            root_fpath = DATA_STORE['coco2014/data/train/images']
            example_numbers = self.num_train_examples
        else:
            record_root = 'vqa/tfrecord/val'
            json_a = self.val_a_json
            json_q = self.val_q_json
            root_fpath = DATA_STORE['coco2014/data/val/images']
            example_numbers = self.num_val_examples

        # Construct the record reader
        tf_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(record_root, 'data.tfrecords', force=True))

        # Loop over the data and parse
        errors = 0
        log_message('Building {} dataset...'.format(dataset))
        for idx, entry in tqdm.tqdm(enumerate(json_q['questions']), total=example_numbers):
            # Load the image
            image = load_image(build_fpath_from_image_id(root_fpath, entry['image_id'], dataset))
            image = encode_jpeg(image)
            if image is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(build_fpath_from_image_id(root_fpath, entry['image_id'], dataset), errors))
                continue

            # Parse the caption
            assert entry['question_id'] == json_a['annotations'][idx]['question_id']
            question_raw = entry['question']
            question_dense, question_len = self.dictionary.dense_parse(question_raw, word_padding=self.max_word_length, char_padding=self.max_char_length)
            answer_raw = json_a['annotations'][idx]['multiple_choice_answer']
            answer_dense, answer_len = self.dictionary.dense_parse(answer_raw, word_padding=self.max_word_length, char_padding=self.max_char_length)

            # Add the class mapping
            if answer_raw not in self.class_map:
                self.class_map[answer_raw] = len(self.class_map)
            answer_class = self.class_map[answer_raw]
                

            # Add the image data 
            feature = {
                'question_word_embedding': _int64_feature(np.ravel(question_dense[0]).astype(np.int64)),
                'question_char_embedding': _int64_feature(np.ravel(question_dense[1]).astype(np.int64)),
                'question_length': _int64_feature([question_len]),
                'answer_word_embedding': _int64_feature(np.ravel(answer_dense[0]).astype(np.int64)),
                'answer_char_embedding': _int64_feature(np.ravel(answer_dense[1]).astype(np.int64)),
                'answer_length': _int64_feature([answer_len]),
                'answer_class': _int64_feature([answer_class]),
                'image': _bytes_feature(tf.compat.as_bytes(image)),
            }

            # Write the TF-Record
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            tf_record_writer.write(example.SerializeToString())
        
        self.sample_triple = (image, question_raw, answer_raw)
        tf_record_writer.close()
        DATA_STORE.update_hash(record_root)

    def _map_fn(self, serialized_example):

        # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={'question_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                      'question_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                      'question_length': tf.FixedLenFeature([1], tf.int64),
                      'answer_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                      'answer_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                      'answer_length': tf.FixedLenFeature([1], tf.int64),
                      'answer_class': tf.FixedLenFeature([1], tf.int64),
                      'image': tf.FixedLenFeature([], tf.string),
                      })

        image = tf.image.decode_jpeg(features['image'], tf.uint8)
        image = tf.cast(image, tf.float32) / 255.0

        # This tuple is the longest, most terrible thing ever
        return (features['question_word_embedding'],
                features['question_char_embedding'],
                features['question_length'],
                features['answer_word_embedding'],
                features['answer_char_embedding'],
                features['answer_length'],
                image,
                features['answer_class'])

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

    @property
    def data_format(self):
        if self.sample_triple is not None:
            return self.sample_triple
        else:
            raise NotImplementedError("Sample doesn't exist.  Dataset may not be built.")

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                        ['Num Val Examples', self.num_val_examples],
                        ['Word Vocab Size', self.word_vocab_size],
                        ['Char Vocab Size', self.char_vocab_size]]))
