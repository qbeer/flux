"""
Classes and methods for handling tf-records
"""

import numpy as np
from flux.backend.globals import DATA_STORE

from typing import Tuple, Dict

try:
    import tensorflow as tf
except Exception as ex:
    from flux.util.logging import log_warning
    log_warning(
        'TFRecord utilities require Tensorflow! Get it here: https://www.tensorflow.org/ ')
    raise ex


class TFFeature():
    """
    Wrapper class for the TF-Feature which contains some metadata
    """

    def __init__(self, feature: tf.train.Feature, name: str, shape: Tuple, dtype: np.dtype) -> None:
        self.feature = feature
        self.name = name
        self.shape = shape
        self.dtype = dtype


def _feature_from_raw_data(value, name: str) -> TFFeature:
    """
    Takes a raw data value and a name, and computes a TF.train.feature
    automatically from the data.

    Arguments:
        value {[type]} -- The value of the feature
        name {str} -- The name of the feature

    Raises:
        NotImplementedError -- If the type of the feature is not recognized or supported

    Returns:
        TFFeature -- the metadata/feature object
    """

    # Figure out the type of value that has been passed in
    if type(value) == int:
        # We're handling an int, so we want to return a int feature
        # We upsample all ints to int64 when storing in the tf-record
        return TFFeature(feature=tf.train.Feature(int64_list=tf.train.Int64List(value=[value])), name=name, shape=(1,), dtype=np.int64)
    elif type(value) == float:
        # We return a floating point feature
        # We save single float values as a 1 element float list
        return TFFeature(feature=tf.train.Feature(float_list=tf.train.FloatList(value=[value])), name=name, shape=(1,), dtype=np.float32)
    elif type(value) == str:
        # We're handling string data, thus, we need to convert the data
        # to bytes, and store it as a byte list
        converted_bytes = tf.compat.as_bytes(value)
        return TFFeature(feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=converted_bytes)), name=name, shape=(len(value),), dtype=np.dtype(str))
    elif type(value) == np.ndarray:
        # We're handling a numpy array, we should figure out the type
        if value.dtype == np.int:
            return TFFeature(feature=tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten())), name=name, shape=value.shape, dtype=value.dtype)
        elif value.dtype == np.float:
            return TFFeature(feature=tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten())), name=name, shape=value.shape, dtype=value.dtype)
        else:
            # Store the value as bytes
            converted_bytes = value.tobytes(value)
            return TFFeature(feature=tf.train.Feature(bytes_list=tf.train.BytesList(value=converted_bytes)), name=name, shape=value.shape, dtype=value.dtype)
    else:
        raise NotImplementedError(
            "Feature type {} is not yet supported.".format(type(value)))


def _build_example(*kwargs) -> tf.train.Example:
    """
    Takes a set of input tuples (name, value) and creates a tf.train.example from these
    values

    Returns:
        tf.train.Example -- The created example
    """

    features = list(kwargs)
    feature_dictionary = {}
    example_metadata = {}

    # Save the features to the data
    for feature in features:
        f_meta = _feature_from_raw_data(feature[1], feature[0])
        feature_dictionary[feature[0]] = f_meta.feature
        example_metadata[f_meta.name] = (f_meta.dtype, f_meta.shape)

    # Save the meta-data
    for key, value in example_metadata.items():
        feature_dictionary['flux_metadata_type_{}'.format(key)] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=tf.compat.as_bytes(str(value[0]))
            )
        )
        feature_dictionary['flux_metadata_shape_{}'.format(key)] = tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=tf.compat.as_bytes(str(value[1]))
            )
        )
    return tf.train.Example(features=tf.train.Features(feature=feature_dictionary))

def _parse_example(db_meta: Dict, example: tf.train.Example) -> Tuple:
    """
    Parse a tf.train.example generated with flux
    
    Arguments:
        example {tf.train.Example} -- [description]
    
    Returns:
        Tuple -- [description]
    """
    pass



class TFRecordManager():
    """
    Managing class for TF-Record style data. It should expose:
    1) A set of databases
    2) A way to create a database using a list of numpy arrays
    3) A way to manage meta-data for accessing data
    4) A way to quickly and easily sample data from a tf-record file
    5) Hooks for distributed tf-record training
    """

    def __init__(self, ) -> None:
        pass

    def AddDB(self, key: str) -> None:
        pass
