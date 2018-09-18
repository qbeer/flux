"""
General Structure for Dataset Interface
"""

class Dataset(object):
    def __init__(self, *args, **kwargs):
        return

    def data_format(self):
        raise NotImplementedError("Data_format consists of shape and structure of each sample")

    @property
    def train_db(self):
        raise NotImplementedError("Train_db not implemented. Should return a dataset iterator.")
   
    @property
    def val_db(self):
        raise NotImplementedError("val_db not implemented. ")