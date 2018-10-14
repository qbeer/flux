"""
Data download and parsing for the NMT dataset
Available Languages:
1. En-De
2. En-Vi

"""
import tqdm
import os
import codecs

import tensorflow as tf

from typing import Optional, Dict

from flux.backend.data import maybe_download_and_store_single_file
from flux.backend.globals import DATA_STORE
from flux.processing.nlp.dictionary import NLPDictionary
from flux.util.logging import log_message

from flux.datasets.dataset import Dataset

class NMT(Dataset):
    REQ_SIZE = 32509533

    available_dataset = {"English2Vietnamese": "en-vi", \
                         "English2German": "en-de"}
    def __init__(self, version: str=None, num_parallel_reads: Optional[int]=None, force_rebuild=False, nohashcheck=False) -> None:
        log_message("Building NMT...")
        if not Dataset.has_space(NMT.REQ_SIZE):
            return
        if version == None:
            log_message("Please Select From following translation: en-vi, en-de")
            return
        self.num_parallel_reads = num_parallel_reads
        self.num_val_examples = None
        self.num_train_examples = None
        self.num_test_examples = None
        self.mwl = 40
        self.qwl = 40

        site_prefix = "https://nlp.stanford.edu/projects/nmt/data/"
        root_key = "nmt"

        if version == 'en-vi':
            self.root_key = os.path.join(root_key, "en-vi")
            train_eng_file = os.path.join(site_prefix, "iwslt15.en-vi/train.en")
            train_for_file = os.path.join(site_prefix, "iwslt15.en-vi/train.vi")
            val_eng_file = os.path.join(site_prefix, "iwslt15.en-vi/tst2012.en")
            val_for_file = os.path.join(site_prefix, "iwslt15.en-vi/tst2012.vi")
            
            test_eng_file = os.path.join(site_prefix, "iwslt15.en-vi/tst2013.en")
            test_for_file = os.path.join(site_prefix, "iwslt15.en-vi/tst2013.vi")
            
            vocab_eng_file = os.path.join(site_prefix, "iwslt15.en-vi/vocab.en")
            vocab_for_file = os.path.join(site_prefix, "iwslt15.en-vi/vocab.vi")
            # size = {"train_eng_file": 13603614,
            #         "train_for_file": 18074646,
            #         "val_eng_file": 140250,
            #         "val_for_file": 188396, 
            #         "test_eng_file": 132264,
            #         "test_for_file": 183855, 
            #         "vocab_eng_file": 139741,
            #         "vocab_for_file": 46767}

        elif version == "en-de":
            self.root_key = os.path.join(root_key, "en-de")
            train_eng_file = os.path.join(site_prefix, "wmt14.en-de/train.en")
            train_for_file = os.path.join(site_prefix, "wmt14.en-de/train.de")
            
            val_eng_file = os.path.join(site_prefix, "wmt14.en-de/newstest2012.en")
            val_for_file = os.path.join(site_prefix, "wmt14.en-de/newstest2012.de")
            
            test_eng_file = os.path.join(site_prefix, "wmt14.en-de/newstest2013.en")
            test_for_file = os.path.join(site_prefix, "wmt14.en-de/newstest2013.de")
            
            vocab_eng_file = os.path.join(site_prefix, "wmt14.en-de/vocab.50K.en")
            vocab_for_file = os.path.join(site_prefix, "wmt14.en-de/vocab.50K.de")
            # size = {"train_eng_file": 644874240,
            #         "train_for_file": 717225984,
            #         "val_eng_file": 406528,
            #         "val_for_file": 470016,
            #         "test_eng_file": 355328,
            #         "test_for_file": 405504,
            #         "vocab_eng_file": 404480,
            #         "vocab_for_file": 504832}
            

        # Download Files
        self.train_eng = maybe_download_and_store_single_file(train_eng_file, os.path.join(self.root_key, "train-en"))
        self.train_for = maybe_download_and_store_single_file(train_for_file, os.path.join(self.root_key, "train-for"))
        self.val_eng = maybe_download_and_store_single_file(val_eng_file, os.path.join(self.root_key, "val-en"))
        self.val_for = maybe_download_and_store_single_file(val_for_file, os.path.join(self.root_key, "val-for"))
        self.test_eng = maybe_download_and_store_single_file(test_eng_file, os.path.join(self.root_key, "test-en"))
        self.test_for = maybe_download_and_store_single_file(test_for_file, os.path.join(self.root_key, "test-for"))
        self.vocab_eng = maybe_download_and_store_single_file(vocab_eng_file, os.path.join(self.root_key, "vocab-en"))
        self.vocab_for = maybe_download_and_store_single_file(vocab_for_file, os.path.join(self.root_key, "vocab-for"))

        # Load the vocab files
        src_dictionary_key = os.path.join(self.root_key, "dictionary", "en")
        for_dictionary_key = os.path.join(self.root_key, "dictionary", "for")

        if not DATA_STORE.is_valid(src_dictionary_key) or not DATA_STORE.is_valid(for_dictionary_key) or force_rebuild:
            self.src_dictionary = NLPDictionary()
            self.dst_dictionary = NLPDictionary()
        else:
            self.src_dictionary = NLPDictionary()
            self.dst_dictionary = NLPDictionary()
            self.src_dictionary.load(DATA_STORE[src_dictionary_key])
            self.dst_dictionary.load(DATA_STORE[for_dictionary_key])

        self.num_train_examples = self._build_dataset("train", force_rebuild=force_rebuild)
        self.num_val_examples = self._build_dataset("val", force_rebuild=force_rebuild)
        self.num_test_examples = self._build_dataset("test", force_rebuild=force_rebuild)

        with open(DATA_STORE.create_key(src_dictionary_key, 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.src_dictionary, pkl_file)
            DATA_STORE.update_hash(src_dictionary_key)

        with open(DATA_STORE.create_key(for_dictionary_key, 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dst_dictionary, pkl_file)
            DATA_STORE.update_hash(for_dictionary_key)

        self.word_vocab_size = len(self.src_dictionary.word_dictionary)

        # TODO: Add current vocab size from vocab file

        self._train_db = None
        self._val_db = None


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

    def _build_dataset(self, mode="train", force_rebuild=False, nohashcheck=False):
        # For now, we will not use the provided vocab
        record_root = os.path.join(self.root_key, "tfrecord", mode)
        if force_rebuild or not DATA_STORE.is_valid(record_root, nohashcheck=nohashcheck):
            log_message('Building dataset ({})...'.format(mode))
            tf_record_writer = tf.python_io.TFRecordWriter(\
                DATA_STORE.create_key(record_root, 'data.tfrecords',force=force_rebuild))
            
            if mode == "train":
                eng_file = self.train_eng
                for_file = self.train_for
            if mode == "test":
                eng_file = self.test_eng
                for_file = self.test_for
            else:
                eng_file = self.val_eng
                for_file = self.val_for
                
            with codecs.getreader("utf-8")(tf.gfile.GFile(DATA_STORE[eng_file], mode="rb")) as f:
                eng_data = f.read().splitlines()
            with codecs.getreader("utf-8")(tf.gfile.GFile(DATA_STORE[for_file], mode="rb")) as f:
                for_data = f.read().splitlines()
                    
            for i, line in tqdm.tqdm(enumerate(eng_data)):
                src_dense, src_len = self.src_dictionary.dense_parse(line, \
                                                                        word_padding=self.mwl, \
                                                                        char_padding=0)
                for_line = for_data[i]
                for_dense, for_len = self.dst_dictionary.dense_parse(for_line, \
                                                                    word_padding=self.mwl, \
                                                                    char_padding=0)
                feature_dict = self.build_feature_dict(src_dense[0], for_dense[0], src_len, for_len)

                example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
                tf_record_writer.write(example.SerializeToString())
            tf_record_writer.close()
            DATA_STORE.update_hash(record_root)
            return len(eng_data)
        else:
            return sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[record_root]))

    def _map_fn(self, serialized_example):
        # Parse the DB out from the tf_record file
        features = tf.parse_single_example(
            serialized_example,
            features={'eng_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'foreign_word_embedding': tf.FixedLenFeature([self.mwl], tf.int64),
                      'eng_word_len': tf.FixedLenFeature([], tf.int64),
                      'foreign_word_len': tf.FixedLenFeature([], tf.int64),
                      })

        src = features['eng_word_embedding']
        dst = features['foreign_word_embedding']
        src_len = tf.cast(features['eng_word_len'], tf.int64)
        dst_len = tf.cast(features['eng_word_len'], tf.int64)

        return (src, dst, src_len, dst_len,)

def build_feature_dict(src_dense, for_dense, src_len, for_len):
    feature_dict = {}
    feature_dict['eng_word_embedding'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=src_dense.flatten()))
    feature_dict['foreign_word_embedding'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=for_dense.flatten()))
    feature_dict['eng_word_len'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[src_len]))
    feature_dict['foreign_word_len'] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[for_len]))
    return feature_dict
        
        


