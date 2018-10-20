from flux.util.download import maybe_download, maybe_download_text
import unittest
from flux.backend.globals import DATA_STORE
import os
from filecmp import cmp as compare
from urllib.request import urlopen
import urllib
from flux.util.logging import log_message

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def internet_on():
    try:
        urlopen('http://216.58.192.142', timeout=1)
        return True
    except urllib.error.URLError as err: 
        return False

class DownloadTestCases(unittest.TestCase):
    """ Tests for Download Utilities """

    def __init__(self, *args, **kwargs):
        super(DownloadTestCases, self).__init__(*args, **kwargs)
        self.passed = False
        if not internet_on():
            log_message("No internet.  All Download Test Ignored.")
            self.passed = True
        
        self.sample_download_location = "https://www.w3.org/TR/PNG/iso_8859-1.txt"
        self.working_directory = DATA_STORE.working_directory
        self.sample_txt = os.path.join(TEST_ROOT, "test_data", "sample.txt")
    
    def test_maybe_download(self):
        file_path = maybe_download("sample.txt", self.sample_download_location, self.working_directory)
        if not self.passed:
            self.assertTrue(compare(file_path, self.sample_txt))
        return
