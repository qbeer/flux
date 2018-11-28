"""
Utils for parsing the Visual7W dataset
"""

import json
import os
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



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class Visual7W(object):
    """ One type of VQA Dataset
    http://web.stanford.edu/%7Eyukez/visual7w/
    """
    def __init__(self, data_type="pointing", num_parallel_reads: int=1, force_rebuild: bool=False, ignore_hashes=False,
                 image_shape: Sequence[int] = [448, 448], read_codes=False, code_shape: Sequence[int] = [7, 7, 2048],
                 merge_qa=False) -> None:

        log_message("Building Dataset " + data_type)

        self.image_resize_shape = image_shape
        self.read_codes = read_codes
        self.code_shape = code_shape
        self.merge_qa = merge_qa
        # Get all of the necessary data
        self.images_key = maybe_download_and_store_zip('http://vision.stanford.edu/yukezhu/visual7w_images.zip', 'visual7w/data/images', use_subkeys=False)
        # Get all of the necessary data
        self.dataset_key = maybe_download_and_store_zip("http://web.stanford.edu/~yukez/papers/resources/dataset_v7w_{0}.zip".format(data_type), 'visual7w/{0}/data/json'.format(data_type), use_subkeys=True)
        # Get the grounding data
        self.grounding_key = maybe_download_and_store_zip("http://web.stanford.edu/~yukez/papers/resources/dataset_v7w_grounding_annotations.zip", "visual/data/grounding", use_subkeys=True)
        
        self.image_root_path = DATA_STORE[self.images_key[0]]

        # Compute the size of the datasets
        self.num_train_examples = 0
        self.num_val_examples = 0
        self.num_test_examples = 0

        self.max_word_length = 44
        self.max_char_length = 26

        self.data_type = data_type

        root_key = "visual7w/{0}".format(data_type)
        dict_key = os.path.join(root_key, "dictionary")
        # Load the vocab files
        if not ignore_hashes and (force_rebuild or not DATA_STORE.is_valid(dict_key)):
            self.dictionary = NLPDictionary()
        else:
            self.dictionary = NLPDictionary().load(DATA_STORE[dict_key])

        self.train_fpath = os.path.join(root_key, 'tfrecord/train')
        self.val_fpath = os.path.join(root_key, 'tfrecord/val')
        self.test_fpath = os.path.join(root_key, 'tfrecord/test')

        if force_rebuild:
        # Now that we have the data, load and parse the JSON file
            file_ = DATA_STORE[self.dataset_key[0]]
            with open(file_, 'r') as ptr:
                self._json = json.load(ptr)
            self._build_images()
            self._build_dataset()
        else:
            # Compute the size of the datasets
            self.num_train_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[os.path.join(self.train_fpath, "images")]))
            self.num_val_examples = sum(1 for _ in tf.python_io.tf_record_iterator(DATA_STORE[os.path.join(self.val_fpath, "images")]))

        

        # Setup some default options for the dataset
        self._val_db = None
        self._train_db = None
        self._test_db = None
        self.num_parallel_reads = num_parallel_reads




        # Save the vocab
        if force_rebuild:
            self.dictionary.save(DATA_STORE.create_key(dict_key, 'dict.pkl', force=True))
            DATA_STORE.update_hash(dict_key)


        self.word_vocab_size = len(self.dictionary.word_dictionary)
        self.char_vocab_size = len(self.dictionary.char_dictionary)

    def get_boxes(self, box_id, boxes_dict):
        assert type(box_id) == int
        curr_box = boxes_dict[box_id]
        _box_id = curr_box['box_id']
        assert box_id == _box_id
        answer_name = curr_box['name']
        answer_loc = [curr_box['x'], curr_box['y'], curr_box['width'], curr_box['height']]
        answer_dense, answer_len = self.dictionary.dense_parse(answer_name, word_padding=self.max_word_length, char_padding=self.max_char_length)
        return (answer_loc, answer_dense, answer_len)
        

    def _build_images(self, ) -> None:
        # Define the Record Root

        # Open the TFRecordWriter
        train_record_root = os.path.join(self.train_fpath, "images")
        val_record_root = os.path.join(self.val_fpath, "images")
        test_record_root = os.path.join(self.test_fpath, "images")

        # Construct the record reader
        train_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(train_record_root, 'data.tfrecords', force=True))
        val_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(val_record_root, 'data.tfrecords', force=True))
        test_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(test_record_root, 'data.tfrecords', force=True))

        # Loop over the data and parse
        errors = 0
        log_message('Building the image...')

        images = self._json['images']

        total_num_examples = len(images)
        for idx, entry in tqdm.tqdm(enumerate(images), total=total_num_examples):
            # Load the image
            filename = entry['filename']
            image_path = os.path.join(self.image_root_path, "images", filename)
            assert os.path.exists(image_path)
            image = load_image(image_path)
            image_shape = list(image.shape)
            image = encode_jpeg(image)
            if image is None:
                errors += 1
                log_warning('Error loading image: {}. {} Errors so far.'.format(os.path.join(self.image_root_path, "images", filename), errors))
                continue

            # Split the dataset
            split = entry["split"]
            if split == "val":
                tf_record_writer = val_record_writer
            elif split == "test":
                tf_record_writer = test_record_writer
            else:
                tf_record_writer = train_record_writer

            image_id = entry['image_id']


            feature = {
                'image_size': _int64_feature(image_shape),
                'image_id': _int64_feature([image_id]),
                'image': _bytes_feature(tf.compat.as_bytes(image)),
            }    
            # Write the TF-Record
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            tf_record_writer.write(example.SerializeToString())

        val_record_writer.close()
        train_record_writer.close()
        test_record_writer.close()
        DATA_STORE.update_hash(test_record_root)
        DATA_STORE.update_hash(train_record_root)
        DATA_STORE.update_hash(val_record_root)
    
    def _build_dataset(self, ) -> None:
        # Define the Record Root

        # Open the TFRecordWriter
        train_record_root = os.path.join(self.train_fpath, "data")
        val_record_root = os.path.join(self.val_fpath, "data")
        test_record_root = os.path.join(self.test_fpath, "data")

        # Construct the record reader
        train_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(train_record_root, 'data.tfrecords', force=True))
        val_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(val_record_root, 'data.tfrecords', force=True))
        test_record_writer = tf.python_io.TFRecordWriter(DATA_STORE.create_key(test_record_root, 'data.tfrecords', force=True))

        # Loop over the data and parse
        errors = 0
        log_message('Building the dataset...')

        images = self._json['images']
        if self.data_type == "pointing":
            boxes = self._json['boxes']
            boxes_dict = {d["box_id"]:d for d in boxes}

        total_num_examples = len(images)
        for idx, entry in tqdm.tqdm(enumerate(images), total=total_num_examples):
            # Load the image

            # Split the dataset
            split = entry["split"]
            if split == "val":
                tf_record_writer = val_record_writer
                self.num_val_examples += 1
            elif split == "test":
                tf_record_writer = test_record_writer
                self.num_test_examples += 1
            else:
                tf_record_writer = train_record_writer
                self.num_train_examples += 1

            image_id = entry['image_id']
            qa_pairs = entry['qa_pairs']

            for qa in qa_pairs:
                question_raw = qa['question']
                question_type = qa['type']
                qa_id = qa['qa_id']
                mlt_choice = qa["multiple_choices"]
                answer = qa['answer']

                assert len(mlt_choice) == 3
                question_dense, question_len = self.dictionary.dense_parse(question_raw, word_padding=self.max_word_length, char_padding=self.max_char_length)

                if self.data_type == "telling":
                    answer_dense, answer_len = self.dictionary.dense_parse(answer, word_padding=self.max_word_length, char_padding=self.max_char_length)
                    m1_dense, m1_len = self.dictionary.dense_parse(mlt_choice[0], word_padding=self.max_word_length, char_padding=self.max_char_length)
                    m2_dense, m2_len = self.dictionary.dense_parse(mlt_choice[1], word_padding=self.max_word_length, char_padding=self.max_char_length)
                    m3_dense, m3_len = self.dictionary.dense_parse(mlt_choice[2], word_padding=self.max_word_length, char_padding=self.max_char_length)
                    

                    # Add the image data
                    feature = {
                        'question_word_embedding': _int64_feature(np.ravel(question_dense[0]).astype(np.int64)),
                        'question_char_embedding': _int64_feature(np.ravel(question_dense[1]).astype(np.int64)),
                        'question_length': _int64_feature([question_len]),
                        'ans_word_embedding': _int64_feature(np.ravel(answer_dense[0]).astype(np.int64)),
                        'ans_char_embedding': _int64_feature(np.ravel(answer_dense[1]).astype(np.int64)),
                        'ans_length': _int64_feature([answer_len]),
                        'm1_embedding': _int64_feature(np.ravel(m1_dense[0]).astype(np.int64)),
                        'm1_char_embedding': _int64_feature(np.ravel(m1_dense[1]).astype(np.int64)),
                        'm2_embedding': _int64_feature(np.ravel(m2_dense[0]).astype(np.int64)),
                        'm2_char_embedding': _int64_feature(np.ravel(m2_dense[1]).astype(np.int64)),
                        'm3_embedding': _int64_feature(np.ravel(m3_dense[0]).astype(np.int64)),
                        'm3_char_embedding': _int64_feature(np.ravel(m3_dense[1]).astype(np.int64)),
                        'mc_len': _int64_feature([m1_len, m2_len, m3_len]),
                        "q_type": _bytes_feature(tf.compat.as_bytes(question_type)),
                        'qa_id': _int64_feature([qa_id]),
                        'image_id': _int64_feature([image_id]),
                    }
                
                else:
                    answer_loc, answer_dense, answer_len = self.get_boxes(answer, boxes_dict)
                    m1_loc, m1_dense, m1_len = self.get_boxes(mlt_choice[0], boxes_dict)
                    m2_loc, m2_dense, m2_len = self.get_boxes(mlt_choice[1], boxes_dict)
                    m3_loc, m3_dense, m3_len = self.get_boxes(mlt_choice[2], boxes_dict)
                    coord = answer_loc + m1_loc + m2_loc + m3_loc
                    # Add the image data
                    feature = {
                        'question_word_embedding': _int64_feature(np.ravel(question_dense[0]).astype(np.int64)),
                        'question_char_embedding': _int64_feature(np.ravel(question_dense[1]).astype(np.int64)),
                        'question_length': _int64_feature([question_len]),
                        'ans_word_embedding': _int64_feature(np.ravel(answer_dense[0]).astype(np.int64)),
                        'ans_char_embedding': _int64_feature(np.ravel(answer_dense[1]).astype(np.int64)),
                        'ans_length': _int64_feature([answer_len]),
                        "coordinate": _int64_feature(coord),
                        'm1_embedding': _int64_feature(np.ravel(m1_dense[0]).astype(np.int64)),
                        'm1_char_embedding': _int64_feature(np.ravel(m1_dense[1]).astype(np.int64)),
                        'm2_embedding': _int64_feature(np.ravel(m2_dense[0]).astype(np.int64)),
                        'm2_char_embedding': _int64_feature(np.ravel(m2_dense[1]).astype(np.int64)),
                        'm3_embedding': _int64_feature(np.ravel(m3_dense[0]).astype(np.int64)),
                        'm3_char_embedding': _int64_feature(np.ravel(m3_dense[1]).astype(np.int64)),
                        'mc_len': _int64_feature([m1_len, m2_len, m3_len]),
                        'qa_id': _int64_feature([qa_id]),
                        "q_type": _bytes_feature(tf.compat.as_bytes(question_type)),
                        'image_id': _int64_feature([image_id]),
                    }

                example = tf.train.Example(features=tf.train.Features(feature=feature))

                tf_record_writer.write(example.SerializeToString())

        val_record_writer.close()
        train_record_writer.close()
        test_record_writer.close()
        DATA_STORE.update_hash(test_record_root)
        DATA_STORE.update_hash(train_record_root)
        DATA_STORE.update_hash(val_record_root)


    
    

    def _map_image_fn(self, serialized_example):

        feature_dict = {
            'image_size': tf.FixedLenFeature([3], tf.int64),
            'image_id': tf.FixedLenFeature([1], tf.int64),
            'image': tf.FixedLenFeature([], tf.string),
        } 

        # if self.read_codes:
        #     feature_dict['image_code'] = tf.FixedLenFeature([self.code_shape[0] * self.code_shape[1] * self.code_shape[2]], tf.float32)

        features = tf.parse_single_example(
            serialized_example,
            features=feature_dict)

        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        if self.data_type == "telling":
            image = tf.image.resize_images(image, self.image_resize_shape)
            image.set_shape((self.image_resize_shape[0], self.image_resize_shape[1], 3))

        return (image,
                features['image_size'],
                features['image_id'])            

    def _map_dataset_fn(self, serialized_example):
        if self.data_type == "pointing":
            feature_dict = {
            'question_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
            'question_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
            "coordinate": tf.FixedLenFeature([16], tf.int64),
            'question_length': tf.FixedLenFeature([1], tf.int64),
            'ans_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
            'ans_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
            'ans_length': tf.FixedLenFeature([1], tf.int64),
            'm1_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
            'm1_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
            'm2_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
            'm2_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
            'm3_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
            'm3_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
            'mc_len': tf.FixedLenFeature([3], tf.int64),
            'qa_id': tf.FixedLenFeature([1], tf.int64),
            'image_id': tf.FixedLenFeature([1], tf.int64),
            "q_type": tf.FixedLenFeature([], tf.string)
            }
        else:
            feature_dict = {
                'question_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                'question_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                'question_length': tf.FixedLenFeature([1], tf.int64),
                'ans_word_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                'ans_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                'ans_length': tf.FixedLenFeature([1], tf.int64),
                'm1_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                'm1_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                'm2_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                'm2_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                'm3_embedding': tf.FixedLenFeature([self.max_word_length], tf.int64),
                'm3_char_embedding': tf.FixedLenFeature([self.max_word_length, self.max_char_length], tf.int64),
                'mc_len': tf.FixedLenFeature([3], tf.int64),
                'qa_id': tf.FixedLenFeature([1], tf.int64),
                'image_id': tf.FixedLenFeature([1], tf.int64),
                "q_type": tf.FixedLenFeature([], tf.string)
            }


        features = tf.parse_single_example(
            serialized_example,
            features=feature_dict)

        if self.data_type == "pointing":
            return (features['question_word_embedding'],
                    features['question_char_embedding'],
                    features['question_length'],
                    features['ans_word_embedding'],
                    features['ans_char_embedding'],
                    features['ans_length'],
                    features['m1_embedding'],
                    features['m1_char_embedding'],
                    features['m2_embedding'],
                    features['m2_char_embedding'],
                    features['m3_embedding'],
                    features['m3_char_embedding'],
                    features['mc_len'],
                    features['coordinate'],
                    features['qa_id'],
                    features['q_type'],
                    features['image_id'])
        else:
            return (features['question_word_embedding'],
                    features['question_char_embedding'],
                    features['question_length'],
                    features['ans_word_embedding'],
                    features['ans_char_embedding'],
                    features['ans_length'],
                    features['m1_embedding'],
                    features['m1_char_embedding'],
                    features['m2_embedding'],
                    features['m2_char_embedding'],
                    features['m3_embedding'],
                    features['m3_char_embedding'],
                    features['mc_len'],
                    features['qa_id'],
                    features['q_type'],
                    features['image_id'])


    @property
    def train_db(self):
        data_path = DATA_STORE[os.path.join(self.train_fpath, "data")]
        image_path = DATA_STORE[os.path.join(self.train_fpath, "images")]
        if self._train_db is None:
            self._train_db = (tf.data.TFRecordDataset(image_path).map(self._map_image_fn, num_parallel_calls=self.num_parallel_reads), 
            tf.data.TFRecordDataset(data_path).map(self._map_dataset_fn, num_parallel_calls=self.num_parallel_reads))
        return self._train_db

    @property
    def val_db(self):
        data_path = DATA_STORE[os.path.join(self.val_fpath, "data")]
        image_path = DATA_STORE[os.path.join(self.val_fpath, "images")]
        if self._val_db is None:
            self._val_db = (tf.data.TFRecordDataset(image_path).map(self._map_image_fn, num_parallel_calls=self.num_parallel_reads), 
            tf.data.TFRecordDataset(data_path).map(self._map_dataset_fn, num_parallel_calls=self.num_parallel_reads))
        return self._val_db


    @property
    def test_db(self):
        data_path = DATA_STORE[os.path.join(self.test_fpath, "data")]
        image_path = DATA_STORE[os.path.join(self.test_fpath, "images")]
        if self._test_db is None:
            self._test_db = (tf.data.TFRecordDataset(image_path).map(self._map_image_fn, num_parallel_calls=self.num_parallel_reads), 
            tf.data.TFRecordDataset(data_path).map(self._map_dataset_fn, num_parallel_calls=self.num_parallel_reads))
        return self._test_db

    def info(self, ) -> str:
        return(tabulate([['Num Train Examples', self.num_train_examples],
                        ['Num Val Examples', self.num_val_examples],
                        ['Word Vocab Size', self.word_vocab_size],
                        ['Char Vocab Size', self.char_vocab_size]]))

class Visual7W_Telling(Visual7W):
    """ One type of VQA Dataset
    http://web.stanford.edu/%7Eyukez/visual7w/
    """
    def __init__(self, force_rebuild=False, *args, **kwargs) -> None:
        super(Visual7W_Telling, self).__init__(*args, force_rebuild=force_rebuild, data_type="telling", **kwargs)



class Visual7W_Pointing(Visual7W):
    """ One type of VQA Dataset
    http://web.stanford.edu/%7Eyukez/visual7w/

    Download Pointing Dataset
    """
    def __init__(self, *args, **kwargs) -> None:
        super(Visual7W_Pointing, self).__init__(*args, data_type="pointing", **kwargs)
        
