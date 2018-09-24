"""
System level utilities for manipulating file information
"""

import os
import errno
import hashlib
import zipfile
import zlib
import tarfile
from flux.util.logging import log_message


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

def adler32(path: str) -> str:
    """Compute the Adler-32 checksum of a file
    Arguments:
        path {string} -- The path to the file
    Returns:
        string -- the MD5 hash
    """
    checksum = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            checksum += zlib.adler32(chunk)
    return str(hex(checksum))


def unzip(path: str) -> str:
    """Unzip a file in the current directory

    Arguments:
        path {str} -- The path to the file to unzip
    """
    if (path.endswith('.zip')):
        log_message('Decompressing: {}'.format(path))
        zip_ref = zipfile.ZipFile(path, 'r')
        output_fpath = os.path.join('/'.join(path.split('/')[:-1]),path.split('/')[-1][:-4])
        zip_ref.extractall(path=output_fpath)
        zip_ref.close()
        return output_fpath
    else:
        raise ValueError('Not a .zip file: {}'.format(path))


def untar(path: str) -> str:
    """Untar a file

    Arguments:
        path {string} -- The file to untar
    """
    if (path.endswith("tar.gz")):
        log_message('Decompressing: {}'.format(path))
        tar = tarfile.open(path)
        output_fpath = os.path.join('/'.join(path.split('/')[:-1]),path.split('/')[-1][:-7])
        tar.extractall(path=output_fpath)
        tar.close()
        return output_fpath
    else:
        raise ValueError('Not a .tar.gz file: {}'.format(path))
