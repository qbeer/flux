"""
Data download and parsing for the squad dataset
"""

import json
import pickle
from typing import Optional

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from flux.backend.data import maybe_download_and_store_single_file
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.util.logging import log_message


class NLQA():

    def __init__(self, version='0.2', num_parallel_reads: Optional[int]=None, force_rebuild=False) -> None:

        self.version = version
        self.num_parallel_reads = num_parallel_reads

        if self.version == '0.1':
            # Download the training data
            self.json_key = maybe_download_and_store_single_file(
                url='https://newslens.berkeley.edu/QA_dataset0.1.json', key='newslens/json_0.1')

            self.mwl = 766
            self.mcl = 37

        if self.version == '0.2':
            # Download the training data
            self.json_key = maybe_download_and_store_single_file(
                url='https://newslens.berkeley.edu/QA_dataset0.2.json', key='newslens/json_0.2')
            self.mwl = 595
            self.mcl = 16
        else:
            raise ValueError("Invalid version for NLQA dataset")

        # Read the JSON
        with open(DATA_STORE[self.json_key], 'r') as json_file:
            self.json = json.loads(json_file.read())

        # Parse the JSON
        if not force_rebuild and DATA_STORE.is_valid('newslens/dictionary_{}'.format(self.version)):
            with open(DATA_STORE['newslens/dictionary_{}'.format(self.version)], 'rb') as pkl_file:
                self.dictionary = pickle.load(pkl_file)
        else:
            self.dictionary = NLPDictionary(tokenizer='space',
                                            char_maxlen=self.mcl, word_maxlen=self.mwl, pad_output=True)

        # If the tf-records don't exist, build them
        if force_rebuild or not DATA_STORE.is_valid('newslens/tfrecord/train/data_{}'.format(self.version)) or not DATA_STORE.is_valid('newslens/tfrecord/val/data_{}'.format(self.version)):
            log_message('Building data...')

            # Create the tf-record writer
            train_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key('newslens/tfrecord/train/data_{}'.format(self.version), 'data.tfrecords', force=force_rebuild))
            val_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key('newslens/tfrecord/val/data_{}'.format(self.version), 'data.tfrecords', force=force_rebuild))

            # Parse the data into tf-records
            for index, record in enumerate(self.json):
                if index % 100 == 0:
                    log_message('Finished {}/{}'.format(index, len(self.json)))

                tokens = record['masked_document'].split()
                context_dense = self.dictionary.dense_parse(
                    record['masked_document'])
                question_dense = self.dictionary.dense_parse(
                    record['question'])
                label = record['masked_answer']
                label_indices = [x for x in range(
                    len(tokens)) if tokens[x] == label]

                if np.random.random() < 0.9:
                    val = False
                else:
                    val = True

                for l_ind in label_indices:

                    # Built the dataset/tf-records
                    feature_dict = {}
                    feature_dict['context_word_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=context_dense[0].flatten()))
                    feature_dict['context_char_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=context_dense[1].flatten()))
                    feature_dict['question_word_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=question_dense[0].flatten()))
                    feature_dict['question_char_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=question_dense[1].flatten()))
                    feature_dict['word_maxlen'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.mwl]))
                    feature_dict['char_maxlen'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.mcl]))
                    feature_dict['token_label'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[l_ind]))

                    example = tf.train.Example(
                        features=tf.train.Features(feature=feature_dict))

                    if val:
                        val_record_writer.write(
                            example.SerializeToString())
                    else:
                        train_record_writer.write(
                            example.SerializeToString())

            train_record_writer.close()
            val_record_writer.close()
            DATA_STORE.update_hash(
                'newslens/tfrecord/train/data_{}'.format(self.version))
            DATA_STORE.update_hash(
                'newslens/tfrecord/val/data_{}'.format(self.version))

        # Save the dictionary
        with open(DATA_STORE.create_key('newslens/dictionary_{}'.format(self.version), 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dictionary, pkl_file)
            DATA_STORE.update_hash(
                'newslens/dictionary_{}'.format(self.version))

        # Compute the number of training examples in the document
        self.num_dev_examples = sum(
            1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['newslens/tfrecord/val/data_{}'.format(self.version)]))
        self.num_train_examples = sum(
            1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['newslens/tfrecord/train/data_{}'.format(self.version)]))

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictioanary)

        self._dev_db = None
        self._train_db = None

    @property
    def train_db(self,):
        if self._train_db is None:
            self._train_db = tf.data.TFRecordDataset(
                DATA_STORE['newslens/tfrecord/train/data_{}'.format(self.version)], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._train_db

    @property
    def val_db(self,):
        if self._dev_db is None:
            self._dev_db = tf.data.TFRecordDataset(
                DATA_STORE['newslens/tfrecord/val/data_{}'.format(self.version)], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._dev_db

    def _map_fn(self, serialized_example):
            # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={'context_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'context_char_embedding': tf.FixedLenFeature([self.mwl, self.mcl], tf.int64),
                      'question_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'question_char_embedding': tf.FixedLenFeature([self.mwl, self.mcl], tf.int64),
                      'word_maxlen': tf.FixedLenFeature([], tf.int64),
                      'char_maxlen': tf.FixedLenFeature([], tf.int64),
                      'token_label': tf.FixedLenFeature([], tf.int64),
                      })

        cwe = features['context_word_embedding']
        cce = features['context_char_embedding']
        qwe = features['question_word_embedding']
        qce = features['question_char_embedding']
        tl = tf.cast(features['token_label'], tf.int64)

        return (cwe, qwe, cce, qce, tl, tl, tl)

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                         ['Num Dev Examples', self.num_dev_examples],
                         ['Word Vocab Size', self.word_vocab_size],
                         ['Char Vocab Size', self.char_vocab_size]]))
