"""
Backend data manipulation routines
"""

import os
from typing import List

from flux.backend.globals import DATA_STORE
from flux.util.download import maybe_download
from flux.util.system import untar, unzip

def maybe_download_and_store_zip(url: str, root_key: str, description: str=None) -> str:
    old_keys = []
    if DATA_STORE.is_valid(root_key) and validate_subkeys(root_key, old_keys):
        return old_keys
        # Ensure one layer file structure for zip file? TODO (Karen)
            
    data_path = maybe_download(file_name=url.split("/"[-1]), source_url=url, work_directory=DATA_STORE.working_directory, postprocess=unzip)
    
    return register_to_datastore(data_path, root_key)


def maybe_download_and_store_single_file(url: str, key: str, description: str=None) -> str:
    if not DATA_STORE.is_valid(key):
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory)
        DATA_STORE.add_file(key, data_path, description, force=True)
    return key

def validate_subkeys(root_key, old_keys=[]):
    for key in DATA_STORE.db.keys():
        if key.startswith(root_key) and key != root_key:
            old_keys.append(key)
            if not DATA_STORE.is_valid(key):
                return False
    return True

def register_to_datastore(data_path, root_key):
    root_length = len(data_path.split('/'))
    new_keys = []
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            key = '/'.join(os.path.join(root, filename).split('/')[root_length:])
            key = key[: key.rfind('.')] if key.rfind('.') > 0 else key
            new_keys.append(key)
            DATA_STORE.add_file(os.path.join(root_key,key), os.path.join(root, filename), description, force=True)
    DATA_STORE.create_key(root_key, 'root.key', force=True)
    return new_keys

def maybe_download_and_store_tar(url: str, root_key: str, description: str=None) -> List[str]:
    # Validate the keys in the directory
    # needs_redownload = False
    # Traverse the key dictionary, and check the integrity of each of the files
    old_keys = []
    if not (DATA_STORE.is_valid(root_key) and  validate_subkeys(root_key, old_keys)):
        return old_keys

    # This is where the hard work happens
    # First, we have to download the file into the working directory
    data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=untar)

    # The data path gives us the root key
    return register_to_datastore(data_path, root_key)

