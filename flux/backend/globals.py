
import os
from pathlib import Path
import shutil

from flux.backend.datastore import DataStore
from flux.util.logging import log_message
from datetime import datetime

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

    CONFIG_FOLDER = get_var('FLUX_CONFIG_FOLDER', '.flux_config')
    CURR_CONFIG = get_var('FLUX_CONFIG', 'flux_config.json')
    CONFIG_FILE = CURR_CONFIG

    RESET_CONFIG = lambda : retrack_config(True)
    RETRACK_CONFIG = retrack_config
    SAVE_CONFIG = save_config
    DATA_STORE = DataStore(root_filepath=ROOT_FPATH, config_file=CONFIG_FILE)
    
def retrack_config(reset=False):
    nonlocal CURR_CONFIG, DATA_STORE, CONFIG_FILE
    all_configs = os.listdir(CONFIG_FOLDER)
    all_configs.sort(key=lambda x: datetime.strptime(x.split(".")[0], '%Y-%m-%d'), reversed=True)
    if reset:
        CURR_CONFIG = get_var('FLUX_CONFIG', 'flux_config.json')
        CONFIG_FILE = CURR_CONFIG
    else:
        index = all_configs.index(CURR_CONFIG) + 1
        if index < len(all_configs):
            CURR_CONFIG = all_configs[index]
        else:
            log_message("Reached the earliest config file.")
        CONFIG_FILE = os.path.join(CONFIG_FOLDER, CURR_CONFIG)

    DATA_STORE = DataStore(root_filepath=ROOT_FPATH, config_file=CONFIG_FILE)

def save_config():
    # Copy CONFIG_FILE into Current Date file
    current_date = datetime.now().strftime("%Y-%m-%d") + ".json"
    shutil.copyfile(CONFIG_FILE, os.path.join(CONFIG_FOLDER, current_date))
