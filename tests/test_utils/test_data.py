import unittest
import os

class DataFlowTestCases(unittest.TestCase):
    """ Tests for Download Utilities """

    def __init__(self, *args, **kwargs):
        super(DataFlowTestCases, self).__init__(*args, **kwargs)

    def test_maybe_download_and_store_google_drive(self):
        """Given a path, test_maybe_download_and_store_google_drive
           will return a set of keys
        
        """
        return

    def test_maybe_download_and_store_zip(self):
        """Given a path, test_maybe_download_and_store_zip
           will return a set of keys
           1. If use subkey:  Return the keys for all the files in the folder
           2. Else: return the key to the folder
        """
        return

    def test_maybe_download_and_store_single_file(self):
        """Given a path, test_maybe_download_and_store_single_file
           will return a set of keys
        """
        return

    def test_maybe_download_and_store_tar(self):
        """Given a path, test_maybe_download_and_store_tar
           will return a set of keys
           1. If use subkey:  Return the keys for all the files in the folder
           2. Else: return the key to the folder
        """
        return
        
