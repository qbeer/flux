import os
import urllib
from urllib.request import urlopen

TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(TEST_ROOT, "tmp")
TMP_CONFIG = "_tmp_config.json"
TEST_DATA = os.path.join(TEST_ROOT, "test_data")
def internet_on():
    try:
        urlopen('http://216.58.192.142', timeout=1)
        return True
    except urllib.error.URLError as err: 
        return False


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)