"""
Backend data manipulation routines
"""

import os
from typing import List, Dict

from flux.backend.globals import DATA_STORE
from flux.util.download import maybe_download, maybe_download_google_drive
from flux.util.system import untar, unzip, mkdir_p
from flux.util.logging import log_message


def maybe_download_and_store_google_drive(file_pair: Dict[str, str], root_key: str, description: str=None, force_build: bool=False, force_download: bool=False, use_subkeys=True, keep_root=True,**kwargs) -> List[str]:
    """Download a dictionary of files from google drive

    Arguments:
        file_pair {Dict[str, str]} -- key: filename, value: GoogleDrive ID
        root_key {str}
        description {str}
        force_build {boolean} 
        force_download {boolean}
        use_subkeys {boolean}
        keep_root {boolean} - For details, see add_file in flux.backend.datastore

    Returns:
        None
    """
    old_keys: List[str] = []
    if not force_download and DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys

    keys = []
    DATA_STORE.create_key(root_key, '', force=True)

    for file_name in file_pair:
        log_message("Downloading " + file_name)
        file_id = file_pair[file_name]
        file_dest = os.path.join(DATA_STORE.working_directory, file_name)
        data_path = maybe_download_google_drive(file_id, file_dest, force_download=force_download)
        data_path = post_process(data_path)
        log_message("Decompressed " + file_name + " to " + data_path)
        if os.path.isdir(data_path):
            if use_subkeys:
                _keys = register_to_datastore(data_path, root_key, description)
                keys.extend(_keys)
            else:
                data_key = DATA_STORE.add_folder(root_key, data_path, force=True, keep_root=keep_root)
                keys.append(data_key)
        else:
            data_key = DATA_STORE.add_file(root_key, data_path, description, force=True, create_folder=False)
            keys.append(data_key)
        log_message("Completed " + file_name)
    return keys

def post_process(data_path):
    if data_path.endswith(".zip"):
        return unzip(data_path)
    if data_path.endswith(".tar"):
        return untar(data_path)
    return data_path

def maybe_download_and_store_zip(url: str, root_key: str, force_download=False, description: str=None, use_subkeys=True, keep_root=True, **kwargs) -> List[str]:
    old_keys: List[str] = []
    if not force_download and DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys
        # Ensure one layer file structure for zip file? TODO (Karen)
    DATA_STORE.create_key(root_key, '', force=True)
    data_path = maybe_download(file_name=url.split("/")[-1], source_url=url, work_directory=DATA_STORE.working_directory, postprocess=unzip, **kwargs)
    log_message(data_path)
    keys: List[str] = []
    if use_subkeys:
        keys = register_to_datastore(data_path, root_key, description)
    else:
        data_key = DATA_STORE.add_folder(root_key, data_path, force=True, keep_root=keep_root)
        keys.append(data_key)

    return keys 


def maybe_download_and_store_single_file(url: str, key: str, description: str=None, postprocess=None, **kwargs) -> str:
    if not DATA_STORE.is_valid(key):
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        if postprocess is None:
            data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory)
        else:
            data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=postprocess, **kwargs)
        data_key = DATA_STORE.add_file(key, data_path, description, force=True)
    return data_key


def validate_subkeys(root_key, old_keys=[]):
    """Validates the sub-keys in a root key

    Arguments:
        root_key {[type]} -- [description]

    Keyword Arguments:
        old_keys {list} -- [description] (default: {[]})

    Returns:
        [type] -- [description]
    """
    
    for key in DATA_STORE.db.keys():
        if key.startswith(root_key) and key != root_key:
            old_keys.append(key)
            if not DATA_STORE.is_valid(key):
                return False
    if len(old_keys) == 0:
        return False
    old_keys.append(root_key)
    return True

def retrieve_subkeys(root_key):
    keys = []
    for key in DATA_STORE.db.keys():
        if key.startswith(root_key) and key != root_key:
            if DATA_STORE.is_valid(key):
                keys.append(key)
    return keys


def write_csv_file(root_key, filename, description):

    data_path = os.path.join(root_key, filename)
    mkdir_p(root_key)
    open(data_path, 'a+').close()
    key = data_path[: data_path.rfind('.')] if data_path.rfind('.') > 0 else data_path

    DATA_STORE.add_file(key, data_path, description, force=True)

    return data_path


def register_to_datastore(data_path, root_key, description):
    root_length = len(data_path.split('/'))
    new_keys: List[str] = []
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            if not irr_file(filename):
                key = '/'.join(os.path.join(root, filename).split('/')[root_length:])
                key = key[: key.rfind('.')] if key.rfind('.') > 0 else key
                new_key = DATA_STORE.add_file(os.path.join(root_key,key), os.path.join(root, filename), description, force=True, create_folder=False)
                if new_key is not None:
                    new_keys.append(new_key)
    return new_keys

def irr_file(filename):
    case1 = filename.endswith(".zip")
    case2 = filename.endswith('.tar')
    case3 = "README" in filename
    case4 = "LICENSE" in filename
    case5 = "readme" in filename
    return case1 or case2 or case3 or case4 or case5


def maybe_download_and_store_tar(url: str, root_key: str, description: str=None, use_subkeys=True, **kwargs) -> List[str]:
    # Validate the keys in the directory
    # needs_redownload = False
    # Traverse the key dictionary, and check the integrity of each of the files
    old_keys: List[str] = []
    if DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys
    DATA_STORE.create_key(root_key, '', force=True)
    # This is where the hard work happens
    # First, we have to download the file into the working directory
    data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=untar, **kwargs)

    # The data path gives us the root key
    keys: List[str] = []
    if use_subkeys:
        keys = register_to_datastore(data_path, root_key, description)
    else:
        data_key = DATA_STORE.add_folder(root_key, data_path, force=True)
        keys.append(data_key)

    return keys
