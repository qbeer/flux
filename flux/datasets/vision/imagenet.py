"""
Parsing code for the ImageNet dataset
"""
import pickle
import numpy as np

from flux.util.logging import log_message

class Imagenet(object):

    def __init__(self, data_path: str, image_shape: Sequence[int]=(224,224)) -> None:
        self.data_path = data_path
        self.image_reshape_size = image_shape

        self._train_db = None
        self._val_db = None

    def _map_fn(self, serialized_example):
        features = tf.parse_single_example(serialized_example,
            features = {
                'image/channels': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/format': tf.FixedLenFeature([],tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/class/synset': tf.FixedLenFeature([], tf.string),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/class/text': tf.FixedLenFeature([], tf.string),
                'image/colorspace': tf.FixedLenFeature([], tf.string),
                'image/filename': tf.FixedLenFeature([], tf.string),
            })

        image_shape = tf.concat([features['image/height'],features['images/width'], tf.size(tf.string_split([image/colorspace],""))]) 
        image = tf.reshape(tf.decode_raw(features['image/encoded'], tf.uint8), image_shape)

        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize_images(image, self.image_resize_shape)
        image.set_shape([self.image_resize_shape[0], self.image_resize_shape[1], 3])

        # This tuple is the longest, most terrible thing ever
        return (image, features['image/class/label'])
    
    @property
    def train_db(self):
        if self._train_db is None:
            train_record_files = [os.path.join(self.data_path,"train-{:5}-of-01024").format(i) for i in range(1024)]
            self._train_db = tf.data.TFRecordDataset(
                train_record_files, num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._train_db

    @property
    def val_db(self):
        if self._val_db is None:
            val_record_files = [os.path.join(self.data_path,"validation-{:5}-of-00128").format(i) for i in range(128)]
            self._val_db = tf.data.TFRecordDataset(
                val_record_files, num_parallel_reads=self.num_parallel_reads).map(self._map_fn)
        return self._val_db

    @property
    def info(self, ) -> str:
        return(None)

        

        
