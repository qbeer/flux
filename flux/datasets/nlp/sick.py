"""
This code is for semantic relatedness dataset.
http://clic.cimec.unitn.it/composes/materials/SICK.zip
"""

import tqdm
import os
import codecs
import pickle

import tensorflow as tf

from typing import Optional, Dict

from flux.backend.data import maybe_download_and_store_zip
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.util.logging import log_message

from flux.datasets.dataset import Dataset

class SICK(Dataset):
    REQ_SIZE = 32509533
    E_LABEL = {'CONTRADICTION': 0, \
               'CONTRADICTS': 0, \
               'ENTAILMENT': 1, \
               'ENTAILS': 1, \
               'NEUTRAL': 2}
    

    def __init__(self, version: str=None, num_parallel_reads: Optional[int]=None, force_rebuild=False) -> None:
        log_message("Building SICK...")
        if not Dataset.has_space(SICK.REQ_SIZE):
            return

        self.num_parallel_reads = num_parallel_reads
        self.num_val_examples = None
        self.num_train_examples = None
        self.num_test_examples = None
        self.mwl = 40
        self.qwl = 40

        url_source = "http://clic.cimec.unitn.it/composes/materials/SICK.zip"
        self.root_key = "sick"

        # Download Files
        self.keys = maybe_download_and_store_zip(url_source, self.root_key, force_download=True)
        dictionary_key = os.path.join(self.root_key, "dictionary")

        self.dictionary = NLPDictionary()
        if not force_rebuild and DATA_STORE.is_valid(dictionary_key):
            self.dictionary.load(DATA_STORE[dictionary_key])

        self.num_train_examples, self.num_val_examples, self.num_test_examples = self._build_dataset(force_rebuild)
        dictfile = DATA_STORE.create_key(dictionary_key, 'dict.pkl', force=force_rebuild)

        self.dictionary.save(dictfile)
        DATA_STORE.update_hash(dictionary_key)

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self._train_db = None
        self._val_db = None
        self._test_db = None
        log_message("Build Complete")

    def build_db(self, mode="train") -> Dict:
        record_root = os.path.join(self.root_key, "tfrecord", mode)
        db = tf.data.TFRecordDataset(
                DATA_STORE[record_root], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return db

    @property
    def train_db(self, ):
        if self._train_db is None:
            self._train_db = self.build_db(mode="train")
        return self._train_db

    @property
    def val_db(self, ):
        if self._val_db is None:
            self._val_db = self.build_db(mode="val")
        return self._val_db

    @property
    def test_db(self, ):
        if self._test_db is None:
            self._test_db = self.build_db(mode="test")
        return self._test_db

    def _build_dataset(self, force_rebuild=True):
        # For now, we will not use the provided vocab
        train_record_root = os.path.join(self.root_key, "tfrecord", "train")
        val_record_root = os.path.join(self.root_key, "tfrecord", "val")
        test_record_root = os.path.join(self.root_key, "tfrecord", "test")
        if force_rebuild or not DATA_STORE.is_valid(train_record_root):
            log_message('Building dataset...')
            tf_train_writer = tf.python_io.TFRecordWriter(\
                DATA_STORE.create_key(train_record_root, 'data.tfrecords',force=force_rebuild))
            tf_test_writer = tf.python_io.TFRecordWriter(\
                DATA_STORE.create_key(test_record_root, 'data.tfrecords',force=force_rebuild))
            tf_val_writer = tf.python_io.TFRecordWriter(\
                DATA_STORE.create_key(val_record_root, 'data.tfrecords',force=force_rebuild))

                
            data_raw = open(DATA_STORE[self.keys[0]], "r").readlines()
            train, val, test = (0, 0, 0)
            keys = data_raw[0].strip("\n").split("\t")
            for line in tqdm.tqdm(data_raw[1:], total=len(data_raw)):
                _, s_a, s_b, e_label, re_score, enAB, enBA, orig_A, orig_B, _, _, t = line.strip("\n").split("\t")
                if t == "TEST":
                    tf_record_writer = tf_test_writer
                    test += 1
                elif t == "TRIAL":
                    tf_record_writer = tf_val_writer
                    val += 1
                else:
                    tf_record_writer = tf_train_writer
                    train += 1
                s_a_dense, s_a_len = self.dictionary.dense_parse(s_a, word_padding=self.mwl, char_padding=0)
                s_b_dense, s_b_len = self.dictionary.dense_parse(s_b, word_padding=self.mwl, char_padding=0)
                orig_a_dense, orig_a_len = self.dictionary.dense_parse(orig_A, word_padding=self.mwl, char_padding=0)
                orig_b_dense = orig_a_dense
                orig_b_len = orig_a_len
                if orig_A != orig_B:
                    orig_b_dense, orig_b_len = self.dictionary.dense_parse(orig_B, word_padding=self.mwl, char_padding=0)
                re_score = float(re_score)
                label_int = SICK.E_LABEL[e_label]
                ab = SICK.E_LABEL[enAB.split("_")[1].upper()]
                ba = SICK.E_LABEL[enBA.split("_")[1].upper()]

                values = (s_a_dense[0], s_a_len, s_b_dense[0], s_b_len, orig_a_dense[0], orig_a_len, orig_b_dense[0], orig_b_len, re_score, label_int, ab, ba)

                feature_dict = build_feature_dict(keys, values)

                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                tf_record_writer.write(example.SerializeToString())
            tf_train_writer.close()
            tf_test_writer.close()
            tf_val_writer.close()
            DATA_STORE.update_hash(train_record_root)
            DATA_STORE.update_hash(val_record_root)
            DATA_STORE.update_hash(test_record_root)
            return (train, val, test)
        else:
            return (sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[train_record_root])), \
                    sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[val_record_root])), \
                    sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[test_record_root])))

    def _map_fn(self, serialized_example):
        # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={'sentence_A_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'sentence_B_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'original_A_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'original_B_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'lengths': tf.FixedLenFeature([4], tf.int64),
                      'label': tf.FixedLenFeature([3], tf.int64),
                      'semantic_score': tf.FixedLenFeature([], tf.float32),
                      })

        sa = features['sentence_A_embedding']
        sb = features['sentence_B_embedding']
        oa = features['original_A_embedding']
        ob = features['original_B_embedding']
        lengths = tf.cast(features["lengths"], tf.int64)
        label = tf.cast(features['label'], tf.int64)
        ss = tf.cast(features['semantic_score'], tf.float32)
        return (sa, sb, oa, ob, lengths, label, ss)

def build_feature_dict(keys, values):
    s_a_dense, s_a_len, s_b_dense, s_b_len, orig_a_dense, orig_a_len, orig_b_dense, \
                orig_b_len, re_score, label_int, ab, ba = values

    feature_dict = {}
    feature_dict['sentence_A_embedding'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=s_a_dense.flatten()))
    feature_dict['sentence_B_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=s_b_dense.flatten()))
    feature_dict['original_A_embedding'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=orig_a_dense.flatten()))
    feature_dict['original_B_embedding'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=orig_b_dense.flatten()))

    feature_dict['lengths'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[s_a_len, s_b_len, orig_a_len, orig_b_len]))
    feature_dict['label'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[ab, ba, label_int]))
    feature_dict['semantic_score'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[re_score]))            
    return feature_dict