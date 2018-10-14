"""
Backend data-storage management. The data-store is initialized at load
and provides the management over the flux_ROOT_DIR folder. It is also
responsible for keeping track of tf-record data and making sure that
everything ends up in the right place.
"""

import os
import atexit
import json
import shutil
import tqdm

from typing import Dict, Optional
from flux.util.system import mkdir_p, adler32, mv_r
from flux.util.logging import log_warning


class KeyExistsError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class DataStore():
    """Class for managing back-end data files. This data-file store
    will provide easy ways for the backend to manage downloading files
    and making sure that files exist on the data backend.
    """

    def __init__(self, root_filepath: str, config_file: str='.flux_config.json') -> None:
        """Create a DataStore object

        Arguments:
            root_filepath {str} -- The root file-path of the data store

        Keyword Arguments:
            config_file {str} -- The name of the configuration file (default: {'.flux_config.json'})

        Returns:
            None
        """

        # Register the termination method to be run at-termination time
        atexit.register(DataStore.at_terminate, self)

        # First, check to see if we can load the current DBStore information
        # if not, then we need to initialize a new DBStore
        self.root_filepath = root_filepath
        self.config_file = config_file
        if not os.path.exists(os.path.join(self.root_filepath, self.config_file)):
            if not os.path.exists(self.root_filepath):
                mkdir_p(self.root_filepath)
            self.db: Dict[str, Dict[str, Optional[str]]] = {}

        else:
            # Load the information in the database from the file
            with open(os.path.join(self.root_filepath, self.config_file), 'r') as in_file:
                self.db = json.loads(in_file.read())

        # If there's no working directory, setup the working directory
        self.working_directory = os.path.join(self.root_filepath, 'work')
        if not os.path.exists(self.working_directory):
            mkdir_p(self.working_directory)

        self.flush()

    def add_file(self, key: str, fpath: str, description: str=None, force: bool=False) -> Dict[str, Optional[str]]:

        if key in self.db and not force:
            # The file already exists in our data store
            return self.db[key]

        # We're adding a file to the data-store. Move it to the
        # proper location in the store based on key
        file_root_location = key.split('/')
        file_to_location = os.path.join(
            self.root_filepath, *file_root_location)

        # If the directory doesn't exist in our local file-store create it
        if not os.path.exists(file_to_location):
            mkdir_p(os.path.join(self.root_filepath, *file_root_location))

        # If it's not already where it needs to go, move it
        if not os.path.exists(os.path.join(file_to_location, fpath.split('/')[-1])):
            os.rename(fpath, os.path.join(
                file_to_location, fpath.split('/')[-1]))

        db_entry = {
            'fpath': os.path.join(file_to_location, fpath.split('/')[-1]),
            'hash': adler32(os.path.join(file_to_location, fpath.split('/')[-1])),
            'folder': '0',
            'description': description
        }
        self.db[key] = db_entry
        self.flush()

        return self.db[key]

    def add_folder(self, key: str, folder_path: str, description: str=None, force: bool=False) -> Dict[str, Optional[str]]:
        if key in self.db and not force:
            # The file already exists in our data store
            return self.db[key]

        # We're adding a file to the data-store. Move it to the
        # proper location in the store based on key
        file_root_location = key.split('/')
        file_to_location = os.path.join(self.root_filepath, *file_root_location)

        # If the directory doesn't exist in our local file-store create it
        if not os.path.exists(file_to_location):
            mkdir_p(os.path.join(self.root_filepath, *file_root_location))

        # If it's not already where it needs to go, move it
        fpath = os.path.join(self.root_filepath, *file_root_location)

        mv_r(folder_path, os.path.join(file_to_location, fpath), overwrite=True)
        db_entry = {
            'fpath': os.path.join(file_to_location, fpath, folder_path.split('/')[-1]),
            'hash': None,
            'folder': '1',
            'description': description
        }
        self.db[key] = db_entry
        self.flush()

        return self.db[key]

    def get_file(self, key: str) -> Optional[Dict[str, Optional[str]]]:
        """Get a file from the data-store

        Arguments:
            key {str} -- The key that we want to look up

        Returns:
            Optional[Dict[str, Optional[str]]] -- The dbfile object if the file is in
            the database, otherwise None
        """

        if key in self.db:
            # Check that the hash is OK
            if hash is not None:
                if adler32(str(self.db[key]['fpath'])) == self.db[key]['hash']:
                    return self.db[key]
                else:
                    # We have the file, but the hash isn't ok. We remove
                    # the data from the store, and pretend like it doesn't
                    # exist
                    self.remove_file(key)
                    return None
            else:
                return self.db[key]
        else:
            return None

    def remove_file(self, key: str) -> None:
        """Remove a file from the DB-Store

        Arguments:
            key {str} -- The key to remove from the datastore

        Returns:
            None
        """

        if key not in self.db:
            return
        else:
            if os.path.exists(str(self.db[key]['fpath'])):
                shutil.rmtree(os.path.join(
                    self.root_filepath, *key.split('/')))
                self.db.pop(key, None)
                self.flush()
            return

    def has_key(self, key: str) -> bool:
        """Return if the data-store contains a particular key

        Arguments:
            key {str} -- The key to test

        Returns:
            bool -- If the key is in the data store
        """

        return key in self.db

    def __getitem__(self, key: str) -> str:
        return str(self.db[key]['fpath'])

    def create_key(self, key: str, fname: str, description: str=None, force: bool=False) -> str:
        if key in self.db:
            if not force and self.is_valid(key):
                raise KeyExistsError('Can\'t create key: {}! It already exists!'.format(key))
            else:
                self.remove_file(key)

        # We're adding a file to the data-store. Move it to the
        # proper location in the store based on key
        file_root_location = key.split('/')
        file_to_location = os.path.join(
            self.root_filepath, *file_root_location)

        # If the directory doesn't exist in our local file-store create it
        if not os.path.exists(file_to_location):
            mkdir_p(os.path.join(self.root_filepath, *file_root_location))

        db_entry = {
            'fpath': os.path.join(file_to_location, fname),
            'hash': None,
            'folder': '0',
            'description': description
        }

        self.db[key] = db_entry
        self.flush()
        return str(db_entry['fpath'])

    def update_hash(self, key: str) -> None:
        if key in self.db:
            self.db[key]['hash'] = str(adler32(str(self.db[key]['fpath'])))
            self.flush()

    def rehash_all(self,) -> None:
        pop_keys = []
        for key in tqdm.tqdm(self.db.keys()):
            try:
                if self.db[key]['hash'] is not None:
                    self.update_hash(key)
            except FileNotFoundError:
                pop_keys.append(key)
        for ky in pop_keys:
            self.db.pop(ky)

    def is_valid(self, key: str, nohashcheck=False) -> bool:
        try:
            if key in self.db:
                if not nohashcheck and self.db[key]['hash'] is not None:
                    if str(self.db[key]['hash']) == str(adler32(str(self.db[key]['fpath']))):
                        return True
                    else:
                        return False
                else:
                    return True

        # Where is the trigger for this error (Karen)?
        except FileNotFoundError as ex:
            log_warning('Key ({}) doesn\'t exist/has moved :O'.format(key))
            return False
        except Exception as ex:
            log_warning('Key ({}) may have been corrupted: {}'.format(key, str(ex)))
        return False

    def flush(self,) -> None:
        """Flush the DB data to disk

        Returns:
            None
        """

        with open(os.path.join(self.root_filepath, self.config_file), 'w') as out_file:
            json.dump(self.db, out_file)

    def at_terminate(self,):
        """The code which is run at the termination of the
        program. In our case, this saves the db-store data
        """
        # Flush the database
        self.flush()
        # Clean the working directory
        try:
            shutil.rmtree(self.working_directory)
        except Exception as ex:
            log_warning('Error removing working directory: {}'.format(str(ex)))
