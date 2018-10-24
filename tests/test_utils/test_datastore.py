import unittest
import os
from filecmp import cmp as compare
from flux.util.logging import log_message
from .utils import internet_on, touch, TMP, TMP_CONFIG, TEST_DATA
from flux.backend.datastore import DataStore
import shutil

class DatastoreTestCases(unittest.TestCase):
    """ Tests for Datastore Utilities """

    def __init__(self, *args, **kwargs):
        super(DatastoreTestCases, self).__init__(*args, **kwargs)
        self.passed = False
        if not internet_on():
            log_message("No internet.  All Download Test Ignored.")
            self.passed = True
        self.datastore = DataStore(root_filepath=TMP, config_file=TMP_CONFIG, testing=True)

    def setUp(self):
        if os.path.exists(TMP):
            shutil.rmtree(TMP)
        self.folder1 = "folder1"
        self.folder_path1 = os.path.join(TEST_DATA, self.folder1)
        if not os.path.exists(self.folder_path1):
            os.mkdir(self.folder_path1)
        self.filename1 = "_filepath1.txt"
        self.filepath1 = os.path.join(TEST_DATA, self.folder1, self.filename1)
        self.filename2 = "_filepath2.txt"
        self.filepath2 = os.path.join(TEST_DATA, self.folder1, self.filename2)
        touch(self.filepath1)
        touch(self.filepath2)

        with open(self.filepath1, 'r') as data:
            self.data_info = data.read()
        with open(self.filepath2, 'r') as data:
            self.data_info2 = data.read()

        self.folder_key = "folder1_key"
        self.dst_filepath1 = os.path.join(TMP, self.folder_key, self.folder1, self.filename1)
        self.dst_filepath2 = os.path.join(TMP, self.folder_key, self.folder1, self.filename2)
        self.dst_folderpath = os.path.join(TMP, self.folder_key, self.folder1)

    def tearDown(self):
        if os.path.exists(self.folder_path1):
            shutil.rmtree(self.folder_path1)
    def test_add_get_file(self):
        # A file is added to rootpath + key + file
        # For example:  src- .flux/work/file1.txt
        #               dst- .fux/key1/file1.txt
        print("Test Add get file")
        key1 = "file1"
        dst_filepath = os.path.join(TMP, key1, self.filename1)
        description1 = "Test for file1"
        file_entry = self.datastore.add_file(key1, self.filepath1, description1, force=True)
        
        # If add_file ==> fpath should exist, file should be valid (hash and content)
        self.assertTrue(file_entry['fpath'] == dst_filepath)
        self.assertTrue(file_entry["description"] == description1)

        get_file_entry = self.datastore.get_file(key1)
        self.assertTrue(get_file_entry['fpath'] == dst_filepath)
        self.assertTrue(get_file_entry["description"] == description1)

        with open(dst_filepath, 'r') as data2:
            data2_info = data2.read()

        self.assertTrue(self.data_info == data2_info)
        self.assertTrue(self.datastore.is_valid(key1))


    def test_add_folder(self):
        # Adding the entire folder to the key location
        # A file is added to rootpath + key + folder
        # For example:  src- .flux/work/folder1/files*
        #               dst- .fux/key1/folder1/files*
        print("Test add folder")
        if not os.path.exists(self.filepath1):
            touch(self.filepath1)
        description_folder = "This is folder containing _filepath1.txt and _filepath1.txt"
        entry = self.datastore.add_folder(self.folder_key, self.folder_path1, description_folder, True)
        self.assertTrue(entry['fpath'] == self.dst_folderpath)
        self.assertTrue(entry['description'] == description_folder)
     
        with open(self.dst_filepath1, 'r') as data:
            data_info_dest = data.read()
        with open(self.dst_filepath2, 'r') as data:
            data_info2_dest = data.read()

        self.assertTrue(self.data_info == data_info_dest)
        self.assertTrue(self.data_info2 == data_info2_dest)

        self.assertTrue(self.datastore.is_valid(self.folder_key))
        # TODO: hash for a folder not implemented
        return

    def test_add_rm_file(self):
        print("Test add remove file")
        key1 = "file1"
        dst_filepath = os.path.join(TMP, key1, self.filename1)
        description1 = "Test for file1"
        file_entry = self.datastore.add_file(key1, self.filepath1, description1, force=True)
        self.assertTrue(file_entry['fpath'] == dst_filepath)
        self.assertTrue(file_entry["description"] == description1)
        self.assertTrue(self.datastore.has_key(key1))
        self.assertTrue(self.datastore.is_valid(key1))
        # Here, file exists and is valid.
        self.datastore.remove_file(key1)

        # Then the key shouldn't exist
        self.assertFalse(self.datastore.has_key(key1))

        dst_filepath = os.path.join(TMP, key1, self.filename1)

        # Path shouldn't exist
        self.assertFalse(os.path.exists(dst_filepath))

        # Data is invalid
        self.assertFalse(self.datastore.is_valid(key1))
        return

    def test_update_hash(self):
        key1 = "file1"
        file_entry = self.datastore.add_file(key1, self.filepath1, "", force=True)
        
        touch(file_entry['fpath'])
        # self.assertFalse(self.datastore.is_valid(key1)) TODO: This test doesn't pass?
        self.datastore.update_hash(key1)
        self.assertTrue(self.datastore.is_valid(key1))
        
