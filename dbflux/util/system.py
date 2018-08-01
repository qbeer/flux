"""
System level utilities for manipulating file information
"""

import os
import errno
import hashlib


def mkdir_p(fpath: str) -> None:
    """mkdir -p wrapper in python
    Arguments:
        fpath {string} -- The path to construct
    Raises:
        RuntimeError -- If the directory is not correctly created
    """

    try:
        os.makedirs(fpath)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(fpath):
            pass
        else:
            raise RuntimeError()


def md5(path: str) -> str:
    """Compute the MD5 hash of a file
    Arguments:
        path {string} -- The path to the file
    Returns:
        string -- the MD5 hash
    """

    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
