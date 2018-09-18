
import os
from pathlib import Path

from flux.backend.datastore import DataStore
from flux.util.logging import log_message

try:
    initialized
except NameError:
    initialized = True
    log_message('Initializing...')
    # Get the values from the path

    def get_var(var_name: str, default: str) -> str:
        if os.environ.get(var_name) is None:
            return default
        else:
            return str(os.environ.get(var_name))

    ROOT_FPATH = get_var('FLUX_ROOT', os.path.join(
        str(Path.home()), '.flux'))
    CONFIG_FILE = get_var('FLUX_CONFIG', 'flux_config.json')
    DATA_STORE = DataStore(root_filepath=ROOT_FPATH, config_file=CONFIG_FILE)
    
