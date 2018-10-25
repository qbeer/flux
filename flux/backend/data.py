"""
Backend data manipulation routines
"""

import os
from typing import List, Dict

from flux.backend.globals import DATA_STORE
from flux.util.download import maybe_download, maybe_download_google_drive
from flux.util.system import untar, unzip, mkdir_p
from flux.util.logging import log_message


def maybe_download_and_store_google_drive(file_pair: Dict[str, str], root_key: str, description: str=None, force_download: bool=False, use_subkeys=True, **kwargs) -> List[str]:
    old_keys: List[str] = []
    if not force_download and DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys

    keys = []
    DATA_STORE.create_key(root_key, 'root.key', force=True)

    for file_name in file_pair:
        log_message("Downloading " + file_name)
        file_id = file_pair[file_name]
        file_dest = os.path.join(DATA_STORE.working_directory, file_name)
        data_path = maybe_download_google_drive(file_id, file_dest, force_download=force_download)
        data_path = post_process(data_path)
        log_message("Decompressed " + file_name + "to " + data_path)
        if os.path.isdir(data_path):
            if use_subkeys:
                _keys = register_to_datastore(data_path, root_key, description)
                keys.extend(_keys)
            else:
                data_key = os.path.join(root_key, file_name.split(".zip")[0])
                DATA_STORE.add_folder(data_key, data_path, force=True)
                keys.append(data_key)
        else:
            _key = os.path.join(root_key, file_name.split(".")[0])
            DATA_STORE.add_file(_key, data_path, description, force=True)
            keys.append(_key)
        log_message("Completed " + file_name)
    DATA_STORE.create_key(root_key, 'root.key', force=True)

    return [k for k in keys] + [root_key]

def post_process(data_path):
    if data_path.endswith(".zip"):
        return unzip(data_path)
    if data_path.endswith(".tar"):
        return untar(data_path)
    return data_path

def maybe_download_and_store_zip(url: str, root_key: str, description: str=None, use_subkeys=True, **kwargs) -> List[str]:
    old_keys: List[str] = []
    if DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys
        # Ensure one layer file structure for zip file? TODO (Karen)

    data_path = maybe_download(file_name=url.split("/")[-1], source_url=url, work_directory=DATA_STORE.working_directory, postprocess=unzip, **kwargs)
    keys: List[str] = []
    if use_subkeys:
        keys = register_to_datastore(data_path, root_key, description)
        # DATA_STORE.create_key(root_key, 'root.key', force=True) I removed this because this call removes all the file I have stored with the previous register_to_datastore. (Karen)
    else:
        DATA_STORE.add_folder(root_key, data_path, force=True)


    return [os.path.join(root_key, k) for k in keys]


def maybe_download_and_store_single_file(url: str, key: str, description: str=None, postprocess=None, **kwargs) -> str:
    if not DATA_STORE.is_valid(key):
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        if postprocess is None:
            data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory)
        else:
            data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=postprocess, **kwargs)
        DATA_STORE.add_file(key, data_path, description, force=True)
    return key


def validate_subkeys(root_key, old_keys=None):
    """Validates the sub-keys in a root key

    Arguments:
        root_key {[type]} -- [description]

    Keyword Arguments:
        old_keys {list} -- [description] (default: {[]})

    Returns:
        [type] -- [description]
    """
    if old_keys is None:
        old_keys: List[str] = []

    for key in DATA_STORE.db.keys():
        if key.startswith(root_key) and key != root_key:
            old_keys.append(key)
            if not DATA_STORE.is_valid(key):
                return False
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
    DATA_STORE.create_key(root_key, '', force=True)
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            if not filename.endswith(".zip"):
                key = '/'.join(os.path.join(root, filename).split('/')[root_length:])
                key = key[: key.rfind('.')] if key.rfind('.') > 0 else key
                new_keys.append(key)
                DATA_STORE.add_file(os.path.join(root_key,key), os.path.join(root, filename), description, force=True)
    return new_keys


def maybe_download_and_store_tar(url: str, root_key: str, description: str=None, use_subkeys=True, **kwargs) -> List[str]:
    # Validate the keys in the directory
    # needs_redownload = False
    # Traverse the key dictionary, and check the integrity of each of the files
    old_keys: List[str] = []
    if DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys

    # This is where the hard work happens
    # First, we have to download the file into the working directory
    data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=untar, **kwargs)

    # The data path gives us the root key
    keys: List[str] = []
    if use_subkeys:
        keys = register_to_datastore(data_path, root_key, description)
    else:
        DATA_STORE.create_key(root_key, '', force=True)

    return [os.path.join(root_key, k) for k in keys] + [root_key]
