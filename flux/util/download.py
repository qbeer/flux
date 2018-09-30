"""
Utilities for downloading data from the internet
"""

import os
import urllib
import urllib.request
import requests

from typing import Dict

from tqdm import tqdm

from flux.util.logging import log_message
from flux.util.system import mkdir_p

tqdm.monitor_interval = 0

MOCK_BROWSER_HEADER = [('User-agent', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7')]
CHUNK_SIZE = 1024 * 32

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(file_name: str, source_url: str, work_directory: str, postprocess=None, username: str=None, password: str=None):
    """Download a file from source-url to the work directory as file_name if the
       file does not already exist
    Arguments:
        file_name {str} -- The name of the file to save the url download as
        source_url {str} -- The URL to download from
        work_directory {str} -- The directory to download to
    """

    # Create the work directory if it doesn't already exist
    if not os.path.exists(work_directory):
        mkdir_p(work_directory)

    # Check if the file-exists, if not, retrieve it
    filepath = os.path.join(work_directory, file_name)
    if not os.path.exists(filepath):
        log_message('Downloading {} from {}, please wait...'.format(
            file_name, source_url))
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1) as t:
            # Create a mock browser 
            if username is not None:
                if password is None:
                    raise ValueError('If using authentication, provide both a username and password.')
                manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
                manager.add_password(None, source_url, username, password)
                auth = urllib.request.HTTPBasicAuthHandler(manager)
                opener = urllib.request.build_opener(auth)
            else:
                opener = urllib.request.build_opener()
            opener.addheaders = MOCK_BROWSER_HEADER
            urllib.request.install_opener(opener)
            filepath, _ = urllib.request.urlretrieve(source_url, filepath, t.update_to)
        stat_info = os.stat(filepath)
        log_message('Successfully downloaded {} ({} bytes).'.format(
            file_name, stat_info.st_size))
        if postprocess is not None:
            filepath = postprocess(filepath)
    return filepath



    

def maybe_download_text(url:str, charset: str='utf-8') -> str:
    """Get URL contents as a string

    Arguments:
        url {str} -- The URL to download from

    Keyword Arguments:
        charset {str} -- The character-set to use for decoding (default: {'utf-8'})

    Returns:
        str -- The decoded data
    """
    return urllib.request.urlopen(url).read().decode(charset)


# Citation: https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def maybe_download_google_drive(file_id:str, file_destination:str, force_download=False) -> str:
    """Get List of Files from google drive

    Arguments:
        file_id -- url id of the file
        file_destination -- place to download the file
    Returns:
        file_destination
    """
    if os.path.isfile(file_destination) and not force_download:
        print("File already exists")
        return file_destination

    GOOGLE_URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(GOOGLE_URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(GOOGLE_URL, params = params, stream = True)

    save_response_content(response, file_destination, CHUNK_SIZE)
    return file_destination    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    
    return None

def save_response_content(response, destination, chunk_size):
    with open(destination, "wb") as f:
        for chunk in  tqdm(response.iter_content(chunk_size)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)