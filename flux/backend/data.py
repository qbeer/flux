"""
Backend data manipulation routines
"""

from flux.backend.globals import DATA_STORE
from flux.util.download import maybe_download


def maybe_download_and_store_single_file(url: str, key: str, description: str=None) -> str:
    if not DATA_STORE.is_valid(key):
        # This is where the hard work happens
        # First, we have to download the file into the working directory
        data_path = maybe_download(url.split('/')[-1], url, DATA_STORE.working_directory)
        DATA_STORE.add_file(key, data_path, description, force=True)
    return key
