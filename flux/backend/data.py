"""
Backend data manipulation routines
"""

import os
from typing import List

from flux.backend.globals import DATA_STORE
from flux.util.download import maybe_download
from flux.util.system import untar


def maybe_download_and_store_single_file(url: str, key: str, description: str=None) -> str:
    if not DATA_STORE.is_valid(key):
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory)
        DATA_STORE.add_file(key, data_path, description, force=True)
    return key


def maybe_download_and_store_tar(url: str, root_key: str, description: str=None) -> List[str]:
    # Validate the keys in the directory
    needs_redownload = False
    # Traverse the key dictionary, and check the integrity of each of the files
    old_keys = []
    if DATA_STORE.is_valid(root_key):
        for key in DATA_STORE.db.keys():
            if key.startswith(root_key) and key != root_key:
                old_keys.append(key)
                if not DATA_STORE.is_valid(key):
                    needs_redownload = True
                    break
    else:
        needs_redownload = True

    if needs_redownload:
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory, postprocess=untar)

        # The data path gives us the root key
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

    else:
        return old_keys
