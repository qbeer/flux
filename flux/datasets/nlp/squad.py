"""
Data download and parsing for the squad dataset
"""

import json
import pickle

import tensorflow as tf

from typing import Optional
from tabulate import tabulate

from flux.backend.data import maybe_download_and_store_single_file
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.processing.nlp.util import get_token_span_from_char_span
from flux.util.logging import log_message


class Squad():

    def __init__(self, version: str='2.0', num_parallel_reads: Optional[int]=None, force_rebuild=False, nohashcheck=False) -> None:

        self.num_parallel_reads = num_parallel_reads
        self.num_val_examples = None
        self.num_train_examples = None

        if version == '2.0':
            self.training_data_json_key = maybe_download_and_store_single_file(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json', key='squad/train_json')
            self.dev_data_json_key = maybe_download_and_store_single_file(
                url='https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', key='squad/dev_json')

            # Load the JSON from the files
            with open(DATA_STORE[self.training_data_json_key], 'r') as train_json:
                self.train_json = json.loads(train_json.read())
            with open(DATA_STORE[self.dev_data_json_key], 'r') as dev_json:
                self.dev_json = json.loads(dev_json.read())

            # Setup some baked constants in the dataset
            self.mwl = 766
            self.mcl = 37

            # Parse the JSON
            if not force_rebuild and DATA_STORE.is_valid('squad/dictionary', nohashcheck=nohashcheck):
                with open(DATA_STORE['squad/dictionary'], 'rb') as pkl_file:
                    self.dictionary = pickle.load(pkl_file)
            else:
                self.dictionary = NLPDictionary(
                    char_maxlen=self.mcl, word_maxlen=self.mwl, pad_output=True)

            # Build the training set if necessary
            self.num_train_examples = self.build_dataset(train=True, force_rebuild=force_rebuild, nohashcheck=nohashcheck)
            if self.num_train_examples is None:
                self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['squad/tfrecord/train']))

            # Build the validation set if necessary
            self.num_val_examples = self.build_dataset(train=False, force_rebuild=force_rebuild, nohashcheck=nohashcheck)
            if self.num_val_examples is None:
                self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE['squad/tfrecord/dev']))

            self.train_fpath = DATA_STORE['squad/tfrecord/train']
            self.dev_fpath = DATA_STORE['squad/tfrecord/dev']

            # Save the dictionary
            with open(DATA_STORE.create_key('squad/dictionary', 'dict.pkl', force=True), 'wb') as pkl_file:
                pickle.dump(self.dictionary, pkl_file)
                DATA_STORE.update_hash('squad/dictionary')

            self.word_vocab_size = len(self.dictionary.word_dictionary)
            self.char_vocab_size = len(self.dictionary.char_dictioanary)

            self._val_db = None
            self._train_db = None
        else:
            raise NotImplementedError(
                "Only version 2.0 is currently supported")

    def build_dataset(self,train,force_rebuild=False, nohashcheck=False):
        record_root = 'squad/tfrecord/train' if train else 'squad/tfrecord/dev'
        json_data = self.train_json['data'] if train else self.dev_json['data']
        num_errors = 0
        num_documents = 0

        if force_rebuild or not DATA_STORE.is_valid(record_root, nohashcheck=nohashcheck):
            log_message('Building dataset ({})...'.format('Train' if train else 'Valid'))
            tf_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key(record_root, 'data.tfrecords',force=force_rebuild))
            for idx, article in enumerate(json_data):
                if idx % 1 == 0:
                    log_message('[{}/{}] Documents Parsed ({} examples, {} errors)'.format(
                        idx, len(json_data), num_documents, num_errors))
                for paragraph_json in article['paragraphs']:

                    # Compute the context embedding
                    context_tokens = self.dictionary.tokenizer.parse(
                        paragraph_json['context'].strip().replace('\n', ''))
                    context_dense, context_len = self.dictionary.dense_parse_tokens(
                        context_tokens)

                    # Compute the QA embeddings
                    for question_answer in paragraph_json['qas']:
                        question_dense, question_len = self.dictionary.dense_parse(
                            question_answer['question'].strip().replace('\n', ''))

                        # For each answer
                        for answer in question_answer['answers']:
                            answer_dense, answer_len = self.dictionary.dense_parse(
                                answer['text'])

                            # Character span start/end
                            span_start = answer['answer_start']
                            span_end = span_start + len(answer['text'])

                            # Get the token span from the char span
                            token_span_start, token_span_end = get_token_span_from_char_span(
                                paragraph_json['context'].strip().replace('\n', ''), context_tokens, span_start, span_end)

                            if token_span_start < 0 or token_span_end < 0:
                                num_errors += 1
                                break

                            # Now that we've got the contents, let's make a TF-Record
                            # We're going to handle the tf-record writing here for now
                            # TODO: Move the tf-record writing to it's own file
                            feature_dict = self.build_feature_dict(context_dense, question_dense, answer_dense, span_start, span_end, token_span_start, token_span_end, context_len, question_len, answer_len)

                            example = tf.train.Example(
                                features=tf.train.Features(feature=feature_dict))
                            tf_record_writer.write(
                                example.SerializeToString())
                            num_documents += 1
            tf_record_writer.close()
            DATA_STORE.update_hash(record_root)
        return num_documents

    def build_feature_dict(self, context_dense, question_dense, answer_dense, span_start, span_end, token_span_start, token_span_end, context_len, question_len, answer_len):
        # Create the feature dictionary
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
        feature_dict['answer_char_embedding'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=answer_dense[1].flatten()))
        feature_dict['word_maxlen'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[self.mwl]))
        feature_dict['char_maxlen'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[self.mcl]))
        feature_dict['span_start'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[span_start]))
        feature_dict['span_end'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[span_end]))
        feature_dict['token_span_start'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[token_span_start]))
        feature_dict['token_span_end'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[token_span_end]))
        feature_dict['context_word_len'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[context_len]))
        feature_dict['question_word_len'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[question_len]))
        feature_dict['answer_word_len'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[answer_len]))

        return feature_dict

    @property
    def train_db(self,):
        if self._train_db is None:
            self._train_db = tf.data.TFRecordDataset(
                DATA_STORE['squad/tfrecord/train'], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._train_db

    @property
    def val_db(self,):
        if self._val_db is None:
            self._val_db = tf.data.TFRecordDataset(
                DATA_STORE['squad/tfrecord/dev'], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._val_db

    def _map_fn(self, serialized_example):
            # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={'context_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'context_char_embedding': tf.FixedLenFeature([self.mwl, self.mcl], tf.int64),
                      'question_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'question_char_embedding': tf.FixedLenFeature([self.mwl, self.mcl], tf.int64),
                      'answer_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'answer_char_embedding': tf.FixedLenFeature([self.mwl, self.mcl], tf.int64),
                      'word_maxlen': tf.FixedLenFeature([], tf.int64),
                      'char_maxlen': tf.FixedLenFeature([], tf.int64),
                      'span_start': tf.FixedLenFeature([], tf.int64),
                      'span_end': tf.FixedLenFeature([], tf.int64),
                      'token_span_start': tf.FixedLenFeature([], tf.int64),
                      'token_span_end': tf.FixedLenFeature([], tf.int64),
                      'context_word_len': tf.FixedLenFeature([], tf.int64),
                      'question_word_len': tf.FixedLenFeature([], tf.int64),
                      'answer_word_len': tf.FixedLenFeature([], tf.int64)
                      })

        cwe = features['context_word_embedding']
        cce = features['context_char_embedding']
        qwe = features['question_word_embedding']
        qce = features['question_char_embedding']
        tss = tf.cast(features['token_span_start'], tf.int64)
        tse = tf.cast(features['token_span_end'], tf.int64)

        # Other stuff that isn't used yet
        awe = features['answer_word_embedding']
        ace = features['answer_char_embedding']
        wml = tf.cast(features['word_maxlen'], tf.int64)
        cml = tf.cast(features['char_maxlen'], tf.int64)
        sps = tf.cast(features['span_start'], tf.int64)
        spe = tf.cast(features['span_end'], tf.int64)
        cwl = tf.cast(features['context_word_len'], tf.int64)
        qwl = tf.cast(features['question_word_len'], tf.int64)
        awl = tf.cast(features['answer_word_len'], tf.int64)

        # This tuple is the longest, most terrible thing ever
        return (cwe, qwe, cce, qce, tss, tse, awe, ace, wml, cml, sps, spe, cwl, qwl, awl)

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                        ['Num Val Examples', self.num_val_examples],
                        ['Word Vocab Size', self.word_vocab_size],
                        ['Char Vocab Size', self.char_vocab_size]]))
