"""
Classes and methods for handling tf-records
"""

# import numpy as np
# from flux.backend.globals import DATA_STORE

# try:
#     import tensorflow as tf
# except Exception as e:
#     print('Error: You need tensorflow to use the tf-record utilities!')
#     raise e


# class TFFeatureGenerator():

#     def get_numpy_feature_list(feature: np.ndarray) -> tf.train.Feature:
#         for dim in feature.shape:


#     def get_int64_feature(feature: int) -> tf.train.Feature:
#         return tf.train.Feature(int64_list=tf.train.Int64List(value=[feature]))


# class TFRecordWriter():
#     def __init__(self, key: str):
#         # Get a file-key from the data store
#         with DATA_STORE.get_key(key) as key_filepath:
#             self.writer = tf.python_io.TFRecordWriter(key_filepath)
    
#     def add_example_from_(*args):
#         feature_dict = {}
#         for data_element in args:
#             if type(data_element[1]) is int:


#         example = tf.train.Example(features=tf.train.Features(feature_dict))