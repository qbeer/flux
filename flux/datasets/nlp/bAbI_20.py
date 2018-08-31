"""
Wrapper for bAbI-20 Dataset

Source: https://research.fb.com/downloads/babi/

"""

import pickle
import tqdm

from flux.backend.data import maybe_download_and_store_tar
from flux.backend.globals import DATA_STORE
import os
import numpy as np
from typing import Dict, List, Tuple
from flux.processing.nlp.dictionary import NLPDictionary
import tensorflow as tf
from flux.util.logging import log_message

UNIT_SIZE = 15
QA = 3

class bAbI_20:
    task_list = [
        'qa1_single-supporting-fact',
        'qa2_two-supporting-facts',
        'qa3_three-supporting-facts',
        'qa4_two-arg-relations',
        'qa5_three-arg-relations',
        'qa6_yes-no-questions',
        'qa7_counting',
        'qa8_lists-sets',
        'qa9_simple-negation',
        'qa10_indefinite-knowledge',
        'qa11_basic-coreference',
        'qa12_conjunction',
        'qa13_compound-coreference',
        'qa14_time-reasoning',
        'qa15_basic-deduction',
        'qa16_basic-induction',
        'qa17_positional-reasoning',
        'qa18_size-reasoning',
        'qa19_path-finding',
        'qa20_agents-motivations',
    ]


    def __init__(self, num_parallel_reads=1, force_rebuild=True, nohashcheck=True, subset="en", wml=8, cml=10):
        self.task_root = "tasks_1-20_v1-2"
        self.subset = subset
        url = "http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz"
        self.keys = maybe_download_and_store_tar(url=url, root_key=self.task_root)
        self.nlp_dict = NLPDictionary()
        self.wml = wml
        self.cml = cml
        self.num_parallel_reads = num_parallel_reads

        dict_name = self.task_root + "/dictionary"
        self.train_record_root = 'tasks_1-20_v1-2/tfrecord/train'
        self.val_record_root = 'tasks_1-20_v1-2/tfrecord/dev'

        if not force_rebuild and DATA_STORE.is_valid(dict_name, nohashcheck=nohashcheck):
            with open(DATA_STORE[dict_name], 'rb') as pkl_file:
                self.dictionary = pickle.load(pkl_file)
        else:
            self.dictionary = NLPDictionary()

        # Build the training set if necessary
        self.num_train_examples = self.build_dataset(train=True, force_rebuild=force_rebuild, nohashcheck=nohashcheck)
        if self.num_train_examples is None:
            self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[self.train_record_root]))

        # Build the validation set if necessary
        self.num_val_examples = self.build_dataset(train=False, force_rebuild=force_rebuild, nohashcheck=nohashcheck)
        if self.num_val_examples is None:
            self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[self.val_record_root]))

        #  = DATA_STORE[self.train_record_root]
        # self.dev_fpath = DATA_STORE[self.val_record_root]

        # Save the dictionary
        with open(DATA_STORE.create_key(dict_name, 'dict.pkl', force=True), 'wb') as pkl_file:
            pickle.dump(self.dictionary, pkl_file)
            DATA_STORE.update_hash(dict_name)

        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictionary)

        self._sample_val_db = None
        self._sample_train_db = None
        self._train_db = None
        self._val_db = None

        #TODO: Add Shuffle Dataset if necessary.

        print("Build Complete.")

    def read_txt(self, filename):
        with open(filename, "r") as r:
            return r.read()

    def read_file_from_db(self, is_train, task_key):
        task_path = task_key + "_train" if is_train else task_key + "_test"
        return DATA_STORE.get_file(task_path)["fpath"]

    
    def build_dataset(self, train, sample=True, force_rebuild=False, nohashcheck=False):
        num_tasks = 0
        record_root = self.train_record_root if train else self.val_record_root
        record_name = "sample.tfrecords" if sample else "data.tfrecords"
        subset = self.subset

        if not train:
            subset = subset + "-valid"

        if not sample:
            subset = subset + "-10k"
        
        if force_rebuild:
            log_message('Building dataset ({})...'.format('Train' if train else 'Valid'))

            task_path = "{0}/{1}/{2}/{3}"
            
            for task in tqdm.tqdm(bAbI_20.task_list):
                task_tf_root = os.path.join(record_root, subset, task)

                tf_record_writer = tf.python_io.TFRecordWriter(
                DATA_STORE.create_key(task_tf_root, record_name,force=force_rebuild))

                task_path = task_path.format(self.task_root, self.task_root, subset, task)
                data_path = self.read_file_from_db(train, task_path)

                txt = self.read_txt(data_path)
                features = self.parse_context_question(txt)

                for feature_dict in features:
                    example = tf.train.Example(\
                                    features=tf.train.Features(feature=feature_dict))
                    tf_record_writer.write(example.SerializeToString())
                tf_record_writer.close()
                DATA_STORE.update_hash(task_tf_root)
                num_tasks += 1
        return num_tasks

            

    def parse_context_question(self, text: str) -> List:
        context = ([], [], [])
        question = ([], [], [])
        answer = ([], [], [])
        source = []
        features = []

        for i, eachline in enumerate(text.split("\n")):
            # use enumerate to double check there is no 
            # special lines
            j = (i % UNIT_SIZE) + 1 
            parsed_line = eachline.split(" ")
            if len(parsed_line) < 2:
                continue
            index = int(parsed_line[0])
            assert index == j
            text = " ".join(parsed_line[1:])
            if index % QA == 0:
                q, a, src = text.split("\t")
                q_dense, q_len = self.nlp_dict.dense_parse(q,
                                                    word_padding=self.wml, char_padding=self.cml)
                a_dense, a_len = self.nlp_dict.dense_parse(a, 
                                                    word_padding=self.wml,
                                                    char_padding=self.cml)
                c_dense, c_len = self.nlp_dict.dense_parse("",
                                                    word_padding=self.wml,
                                                    char_padding=self.cml)
            
                question[0].append(q_dense[0])
                question[1].append(q_dense[1])
                question[2].append(q_len)
                
                answer[0].append(a_dense[0])
                answer[1].append(a_dense[1])
                answer[2].append(a_len)
                
                source.append(src)
            else:
                c_dense, c_len = self.nlp_dict.dense_parse(text,
                                    word_padding=self.wml,
                                    char_padding=self.cml)
            
            context[0].append(c_dense[0])
            context[1].append(c_dense[1])
            context[2].append(c_len)
            
            if index == UNIT_SIZE:
                context = self._to_np_array(context)
                question = self._to_np_array(question)
                answer = self._to_np_array(answer)
                
                source = np.array(source)
                
                feature_dict = self.build_feature_dict(context, question, answer, source)
                
                context = ([], [], [])
                question = ([], [], [])
                answer = ([], [], [])
                source = []
                
                features.append(feature_dict)
        return features

    def build_feature_dict(self, context, question, answer, source):
        context_dense, context_char, c_len = context
        question_dense, question_char, q_len = question
        answer_dense, answer_char, a_len = answer
        feature_dict = {}
        feature_dict['context_word_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_dense.flatten()))
        feature_dict['context_char_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_char.flatten()))
        feature_dict['question_word_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=question_dense.flatten()))
        feature_dict['question_char_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=question_char.flatten()))
        feature_dict['answer_word_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=answer_dense.flatten()))
        feature_dict['answer_char_embedding'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=answer_char.flatten()))
            
        feature_dict['word_maxlen'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self.wml]))
        feature_dict['char_maxlen'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=[self.cml]))
            
        feature_dict['context_word_len'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=c_len))
        feature_dict['question_word_len'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=q_len))
        feature_dict['answer_word_len'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=a_len))
        return feature_dict

        

    def _to_np_array(self, t: Tuple):
        return tuple([np.array(i) for i in t])

    def build_db(self, is_train, db, subset) -> Dict:
        record_root = self.train_record_root if is_train else self.val_record_root
        if db is not None:
            return db
        task_dict = {}
        for task in bAbI_20.task_list:
            task_tf_root = os.path.join(record_root, subset, task)

            if not DATA_STORE.is_valid(task_tf_root):
                raise NotImplementedError("{} not built".format(task_tf_root))

            task_dict[task] = tf.data.TFRecordDataset(
                DATA_STORE[task_tf_root], num_parallel_reads=self.num_parallel_reads).map(self._map_fn)

        return task_dict

    @property
    def sample_train_db(self,) -> Dict:
        self._sample_train_db = self.build_db(True, self._sample_train_db, self.subset)
        return self._sample_train_db
    
    @property
    def sample_val_db(self, ) -> Dict:
        self._sample_val_db = self.build_db(False, self._sample_val_db, self.subset + "-valid")
        return self._sample_val_db

    @property
    def train_db(self,) -> Dict:
        self._train_db = self.build_db(True, self._train_db, self.subset + "-10k")
        return self._train_db

    @property
    def val_db(self,) -> Dict:
        self._val_db = self.build_db(False, self._val_db, self.subset + "-10k-valid")
        return self._val_db

    def _map_fn(self, serialized_example):
        feature_dict = {}
        feature_dict['context_word_embedding'] =  tf.FixedLenFeature([UNIT_SIZE, self.wml], tf.int64)
    #     feature_dict['context_char_embedding'] = tf.FixedLenFeature([15, mwl, mcl], tf.int64)
        feature_dict['question_word_embedding'] = tf.FixedLenFeature([QA, self.wml], tf.int64)
    #     feature_dict['question_char_embedding'] = tf.FixedLenFeature([5, mwl, mcl], tf.int64)
        feature_dict['answer_word_embedding'] = tf.FixedLenFeature([QA, self.wml], tf.int64)
    #     feature_dict['answer_char_embedding'] = tf.FixedLenFeature([5, mwl, mcl], tf.int64)

        feature_dict['word_maxlen'] = tf.FixedLenFeature([], tf.int64)
    #     feature_dict['char_maxlen'] = tf.FixedLenFeature([], tf.int64)
        feature_dict['context_word_len'] = tf.FixedLenFeature([UNIT_SIZE],tf.int64)
        feature_dict['question_word_len'] = tf.FixedLenFeature([QA],tf.int64)
        feature_dict['answer_word_len'] = tf.FixedLenFeature([QA],tf.int64)


        features = tf.parse_single_example(\
                                            serialized_example,\
                                        features=feature_dict)
        
        cwe = features["context_word_embedding"]
        qwe = features["question_word_embedding"]
        awe = features["answer_word_embedding"]
        
        wml = features["word_maxlen"]
        cwl = features["context_word_len"]
        qwl = features["question_word_len"]
        awl = features["answer_word_len"]
        return cwe, qwe, awe, wml, cwl, qwl, awl

