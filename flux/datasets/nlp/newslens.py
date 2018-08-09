"""
Data download and parsing for the squad dataset
"""

import json
import pickle
import tqdm
from typing import Optional, List

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from flux.backend.data import maybe_download_and_store_single_file
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.util.logging import log_message


class NLQA():

    def __init__(self, version='0.3', num_parallel_reads: Optional[int]=None,
                 force_rebuild: bool=False, mask: bool=True,
                 add_start_tokens: bool=False, add_stop_tokens: bool=False,
                 use_qam: bool=False) -> None:

        self.version = version
        self.num_parallel_reads = num_parallel_reads
        self.mask = mask
        self.add_start_tokens = add_start_tokens
        self.add_stop_tokens = add_stop_tokens
        self.use_qam = use_qam

        # We keep one copy of masked data, and one copy of unmasked data
        if self.mask:
            self.stem = 'newslens/masked/'
        else:
            self.stem = 'newslens/'

        # We don't use the stem here, because the json files are the same
        if self.version == '0.1':
            # Download the training data
            self.json_key = maybe_download_and_store_single_file(
                url='https://newslens.berkeley.edu/QA_dataset0.1.json', key='newslens/json_0.1')

            self.mwl = 766
            self.mcl = 37
            self.mql = 766

        elif self.version == '0.2':
            # Download the training data
            self.json_key = maybe_download_and_store_single_file(
                url='https://newslens.berkeley.edu/QA_dataset0.2.json', key='newslens/json_0.2')
            self.mwl = 595
            self.mcl = 16
            self.mql = 766

        elif self.version == '0.3':
            # Download the training data
            self.json_key = maybe_download_and_store_single_file(
                url='https://newslens.berkeley.edu/QA_dataset0.3.json', key='newslens/json_0.3')
            self.mwl = 600
            self.mcl = 16
            self.mql = 20
        else:
            raise ValueError("Invalid version for NLQA dataset")

        # Read the JSON
        with open(DATA_STORE[self.json_key], 'r') as json_file:
            self.json = json.loads(json_file.read())

        # Parse the JSON
        if not force_rebuild and DATA_STORE.is_valid(self.stem + 'dictionary_{}'.format(self.version)):
            with open(DATA_STORE[self.stem + 'dictionary_{}'.format(self.version)], 'rb') as pkl_file:
                self.dictionary = pickle.load(pkl_file)
        else:
            self.dictionary = NLPDictionary(tokenizer='space')

        # If the tf-records don't exist, build them
        if force_rebuild or not DATA_STORE.is_valid(self.stem + 'tfrecord/train/data_{}'.format(self.version)) or not DATA_STORE.is_valid(self.stem + 'tfrecord/val/data_{}'.format(self.version)):
            log_message('Building dataset...')

            # Create the tf-record writer
            train_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key(self.stem + 'tfrecord/train/data_{}'.format(self.version), 'data.tfrecords', force=force_rebuild))
            val_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key(self.stem + 'tfrecord/val/data_{}'.format(self.version), 'data.tfrecords', force=force_rebuild))

            # Parse the data into tf-records
            for record in tqdm.tqdm(self.json):
        
                # Handle start and stop tokens on the answer
                if self.add_stop_tokens:
                    if self.mask:
                        answer_text = record['masked_answer'].strip() + ' <STOP>'
                    else:
                        answer_text = record['real_answer'].strip() + ' <STOP>'
                else:
                    if self.mask:
                        answer_text = record['masked_answer']
                    else:
                        answer_text = record['real_answer']

                if self.add_start_tokens:
                    answer_text = '<START> ' + answer_text
                if not self.add_stop_tokens:
                    question_answer_dense, qa_len = self.dictionary.dense_parse(record['question'].strip() + ' ' + answer_text.strip() + '<STOP>', word_padding=self.mwl, char_padding=self.mcl)
                else:
                    question_answer_dense, qa_len = self.dictionary.dense_parse(record['question'].strip() + ' ' + answer_text.strip(), word_padding=self.mwl, char_padding=self.mcl)

                if self.mask:
                    tokens = record['masked_document'].split(' ')
                    context_dense, context_len = self.dictionary.dense_parse(record['masked_document'], word_padding=self.mwl, char_padding=self.mcl)
                    label = record['masked_answer'].split(' ')
                else:
                    tokens = record['unmasked_document'].split(' ')
                    context_dense, context_len = self.dictionary.dense_parse(record['unmasked_document'], word_padding=self.mwl, char_padding=self.mcl)
                    label = record['real_answer'].split(' ')

                answer_dense, answer_len = self.dictionary.dense_parse(answer_text, word_padding=self.mql, char_padding=self.mcl)

                question_dense, question_len = self.dictionary.dense_parse(record['question'], word_padding=self.mql, char_padding=self.mcl)

                # Here's a bit of logic to parse out the tokens properly
                potential_starts = [x for x in range(len(tokens)) if tokens[x] == label[0]]
                label_index_start: List[int] = []
                label_index_end: List[int] = []
                for i in potential_starts:
                    idx = [x for x in range(
                        i, len(tokens)) if tokens[x] == label[-1]]
                    if len(idx) > 0:
                        label_index_start.append(i)
                        label_index_end.append(idx[0])
                label_indices = zip(label_index_start, label_index_end)

                if np.random.random() < 0.95:
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
                    feature_dict['answer_word_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=answer_dense[0].flatten()))
                    feature_dict['question_answer_word_embedding'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=question_answer_dense[0].flatten()))
                    feature_dict['word_maxlen'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.mwl]))
                    feature_dict['char_maxlen'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[self.mcl]))
                    feature_dict['token_label_start'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[l_ind[0]]))
                    feature_dict['token_label_end'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[l_ind[1]]))
                    feature_dict['context_word_len'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[context_len]))
                    feature_dict['question_word_len'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[question_len]))
                    feature_dict['question_answer_word_len'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[qa_len]))
                    feature_dict['answer_word_len'] = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[answer_len]))

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
                self.stem + 'tfrecord/train/data_{}'.format(self.version))
            DATA_STORE.update_hash(
                self.stem + 'tfrecord/val/data_{}'.format(self.version))

        # Save the dictionary
        with open(DATA_STORE.create_key(self.stem + 'dictionary_{}'.format(self.version), 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dictionary, pkl_file)
            DATA_STORE.update_hash(
                self.stem + 'dictionary_{}'.format(self.version))

        # Compute the number of training examples in the document
        self.num_val_examples = sum(
            1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[self.stem + 'tfrecord/val/data_{}'.format(self.version)]))
        self.num_train_examples = sum(
            1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[self.stem + 'tfrecord/train/data_{}'.format(self.version)]))

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictioanary)

        self._dev_db = None
        self._train_db = None

    @property
    def train_db(self,):
        if self._train_db is None:
            self._train_db = tf.data.TFRecordDataset(
                DATA_STORE[self.stem + 'tfrecord/train/data_{}'.format(self.version)], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._train_db

    @property
    def val_db(self,):
        if self._dev_db is None:
            self._dev_db = tf.data.TFRecordDataset(
                DATA_STORE[self.stem + 'tfrecord/val/data_{}'.format(self.version)], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
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
                      'token_label_start': tf.FixedLenFeature([], tf.int64),
                      'token_label_end': tf.FixedLenFeature([], tf.int64),
                      'answer_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'question_answer_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'context_word_len': tf.FixedLenFeature([], tf.int64),
                      'question_word_len': tf.FixedLenFeature([], tf.int64),
                      'question_answer_word_len': tf.FixedLenFeature([], tf.int64),
                      'answer_word_len': tf.FixedLenFeature([], tf.int64),
                      })

        cwe = features['context_word_embedding']
        cce = features['context_char_embedding']
        qwe = features['question_word_embedding']
        qce = features['question_char_embedding']
        tls = tf.cast(features['token_label_start'], tf.int64)
        tle = tf.cast(features['token_label_end'], tf.int64)
        if self.use_qam:
            ans = features['answer_word_embedding']
            awl = tf.cast(features['answer_word_len'], tf.int64)
        else:
            ans = features['question_answer_word_embedding']
            awl = tf.cast(features['question_answer_word_len'], tf.int64)

        cwl = tf.cast(features['context_word_len'], tf.int64)
        qwl = tf.cast(features['question_word_len'], tf.int64)

        return (cwe, qwe, cce, qce, tls, tle, ans, cwl, qwl, awl)

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                         ['Num Val Examples', self.num_val_examples],
                         ['Word Vocab Size', self.word_vocab_size],
                         ['Char Vocab Size', self.char_vocab_size]]))
