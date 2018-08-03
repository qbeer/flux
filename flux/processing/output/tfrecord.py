"""
Classes and methods for handling tf-records
"""

import numpy as np
from flux.backend.globals import DATA_STORE

try:
    import tensorflow as tf
except Exception as ex:
    from flux.util.logging import log_warning
    log_warning('TFRecord utilities require Tensorflow! Get it here: https://www.tensorflow.org/ ')
    raise ex
