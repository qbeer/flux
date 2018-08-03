"""
Utilities for downloading data from the internet
"""

import os
import sys
import urllib
import urllib.request

from flux.util.system import mkdir_p
from flux.util.logging import log_message


def maybe_download(file_name:str, source_url:str, work_directory:str, postprocess=None):
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
        filepath, _ = urllib.request.urlretrieve(
            source_url, filepath, progress_bar_report_hook)
        stat_info = os.stat(filepath)
        log_message('Successfully downloaded {} ({} bytes).'.format(
            file_name, stat_info.st_size))
        if postprocess is not None:
            postprocess(filepath)
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


#  reporthook from stackoverflow #13881092
def progress_bar_report_hook(blocknum: int, blocksize: int, totalsize: int) -> None:
    """Gives a progress bar report hook for URLLIB request
    Arguments:
        blocknum {[type]} -- [description]
        blocksize {[type]} -- [description]
        totalsize {[type]} -- [description]
    """

    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))
