
import os
from pathlib import Path

from flux.backend.datastore import DataStore
from flux.util.logging import log_message

initialized = False

if not initialized:
    initialized = True
    log_message('Initializing...')
    # Get the values from the path

    def get_var(var_name: str, default: str):
        if os.environ.get(var_name) is None:
            return default
        else:
            return str(os.environ.get(var_name))

    ROOT_FPATH = get_var('DBFLUX_ROOT', os.path.join(
        str(Path.home()), '.dbflux'))
    CONFIG_FILE = get_var('DBFLUX_CONFIG', 'dbflux_config.json')
    DATA_STORE = DataStore(root_filepath=ROOT_FPATH, config_file=CONFIG_FILE)
