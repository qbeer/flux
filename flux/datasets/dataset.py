"""
General Structure for Dataset Interface
"""
from flux.util.system import freespace

from flux.util.logging import log_message

from flux.backend.globals import ROOT_FPATH

class Dataset(object):
    def __init__(self, *args, **kwargs):
        return

    @staticmethod
    def has_space(download_space):
        """
            download_space:  Should be amount of bytes downloaded.
        """
        bytes_left = freespace(ROOT_FPATH)
        req_bytes = download_space * 2
        if req_bytes >= bytes_left:
            log_message("Recommended {} bytes of free space in disk.".format(req_bytes))
            return False
        return True

    # def data_format(self):
    #     raise NotImplementedError("Data_format consists of shape and structure of each sample")

    @property
    def train_db(self):
        raise NotImplementedError("Train_db not implemented. Should return a dataset iterator.")
    @property
    def val_db(self):
        raise NotImplementedError("val_db not implemented. ")