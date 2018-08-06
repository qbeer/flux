"""
Loading of the GLoVE embedding vectors
"""

import pickle
import os

import numpy as np
from typing import Dict, Optional

from flux.util.system import unzip
from flux.util.logging import log_message
from flux.util.download import maybe_download
from flux.backend.globals import DATA_STORE


class GloveEmbedding():

    # Common constants
    valid_versions = ['wikipedia', 'common-small', 'common-large', 'twitter']
    wikipedia_dimensions = [50, 100, 200, 300]
    common_small_dimensions = [300]
    common_large_dimensions = [300]
    twitter_dimensions = [25, 50, 100, 200]

    def __init__(self, version: str='wikipedia', dimension: int=300) -> None:

        self.version = version
        self.dimension = dimension
        self.embedding_matrix: Optional[np.ndarray] = None

        if self.version == 'wikipedia':
            # Make sure that the dimension is valid
            if self.dimension not in GloveEmbedding.wikipedia_dimensions:
                raise ValueError('Error: Invalid GLoVe dimension ({}) for Wikipedia dataset. Must be one of {}'.format(self.dimension,
                                                                                                                       GloveEmbedding.wikipedia_dimensions))

            if not DATA_STORE.is_valid('glove/wikipedia/dim{}'.format(self.dimension)):
                # Download the file into the working direcotry
                maybe_download(file_name='glove.6B.zip', source_url='http://nlp.stanford.edu/data/glove.6B.zip',
                               work_directory=DATA_STORE.working_directory, postprocess=unzip)

                # Read the data keys from the file
                log_message('Loading vectors...')
                self.encoder: Dict[str, np.ndarray] = {}
                with open(os.path.join(DATA_STORE.working_directory, 'glove.6B.{}d.txt'.format(self.dimension)), 'r') as glove_file:
                    for line in glove_file:
                        tokens = line.split()
                        self.encoder[tokens[0]] = np.array(
                            [float(x) for x in tokens[1:]])

                # Save the encoder
                with open(DATA_STORE.create_key('glove/wikipedia/dim{}'.format(self.dimension), 'encoder.pkl', force=True), 'wb',) as pkl_file:
                    pickle.dump(self.encoder, pkl_file)
                DATA_STORE.update_hash('glove/wikipedia/dim{}'.format(self.dimension))

            else:
                with open(DATA_STORE['glove/wikipedia/dim{}'.format(self.dimension)], 'rb') as pkl_file:
                    self.encoder = pickle.load(pkl_file)

        elif self.version == 'common-small':
            # Make sure that the dimension is valid
            if self.dimension not in GloveEmbedding.common_small_dimensions:
                raise ValueError('Error: Invalid GLoVe dimension ({}) for Common-Crawl Small dataset. Must be one of {}'.format(self.dimension,
                                                                                                                                GloveEmbedding.common_small_dimensions))

            if not DATA_STORE.is_valid('glove/common-small/dim{}'.format(self.dimension)):
                # Download the file into the working direcotry
                maybe_download(file_name='glove.42B.300d.zip', source_url='http://nlp.stanford.edu/data/glove.42B.300d.zip',
                               work_directory=DATA_STORE.working_directory, postprocess=unzip)

                # Read the data keys from the file
                log_message('Loading vectors...')
                self.encoder: Dict[str, np.ndarray] = {}
                with open(os.path.join(DATA_STORE.working_directory, 'glove.42B.{}d.txt'.format(self.dimension)), 'r') as glove_file:
                    for line in glove_file:
                        tokens = line.split()
                        self.encoder[tokens[0]] = np.array(
                            [float(x) for x in tokens[1:]])

                # Save the encoder
                with open(DATA_STORE.create_key('glove/common-small/dim{}'.format(self.dimension), 'encoder.pkl', force=True), 'wb') as pkl_file:
                    pickle.dump(self.encoder, pkl_file)
                DATA_STORE.update_hash('glove/common-small/dim{}'.format(self.dimension))

            else:
                with open(DATA_STORE['glove/common-small/dim{}'.format(self.dimension)], 'rb') as pkl_file:
                    self.encoder = pickle.load(pkl_file)

        elif self.version == 'common-large':
            # Make sure that the dimension is valid
            if self.dimension not in GloveEmbedding.common_large_dimensions:
                raise ValueError('Error: Invalid GLoVe dimension ({}) for Common-Crawl Large dataset. Must be one of {}'.format(self.dimension,
                                                                                                                                GloveEmbedding.common_large_dimensions))

            if not DATA_STORE.is_valid('glove/common-large/dim{}'.format(self.dimension)):
                # Download the file into the working direcotry
                maybe_download(file_name='glove.840B.300d.zip', source_url='http://nlp.stanford.edu/data/glove.840B.300d.zip',
                               work_directory=DATA_STORE.working_directory, postprocess=unzip)

                # Read the data keys from the file
                log_message('Loading vectors...')
                self.encoder: Dict[str, np.ndarray] = {}
                with open(os.path.join(DATA_STORE.working_directory, 'glove.840B.{}d.txt'.format(self.dimension)), 'r') as glove_file:
                    for line in glove_file:
                        tokens = line.split()
                        self.encoder[tokens[0]] = np.array(
                            [float(x) for x in tokens[1:]])

                # Save the encoder
                with open(DATA_STORE.create_key('glove/common-large/dim{}'.format(self.dimension), 'encoder.pkl', force=True), 'wb') as pkl_file:
                    pickle.dump(self.encoder, pkl_file)
                DATA_STORE.update_hash('glove/common-large/dim{}'.format(self.dimension))

            else:
                with open(DATA_STORE['glove/common-large/dim{}'.format(self.dimension)], 'rb') as pkl_file:
                    self.encoder = pickle.load(pkl_file)

        elif self.version == 'twitter':
            # Make sure that the dimension is valid
            if self.dimension not in GloveEmbedding.twitter_dimensions:
                raise ValueError('Error: Invalid GLoVe dimension ({}) for Common-Crawl Large dataset. Must be one of {}'.format(self.dimension,
                                                                                                                                GloveEmbedding.twitter_dimensions))

            if not DATA_STORE.is_valid('glove/twitter/dim{}'.format(self.dimension)):
                # Download the file into the working direcotry
                maybe_download(file_name='glove.twitter.27B.zip', source_url='http://nlp.stanford.edu/data/glove.twitter.27B.zip',
                               work_directory=DATA_STORE.working_directory, postprocess=unzip)

                # Read the data keys from the file
                log_message('Loading vectors...')
                self.encoder: Dict[str, np.ndarray] = {}
                with open(os.path.join(DATA_STORE.working_directory, 'glove.twitter.27B.{}d.txt'.format(self.dimension)), 'r') as glove_file:
                    for line in glove_file:
                        tokens = line.split()
                        self.encoder[tokens[0]] = np.array(
                            [float(x) for x in tokens[1:]])

                # Save the encoder
                with open(DATA_STORE.create_key('glove/twitter/dim{}'.format(self.dimension), 'encoder.pkl', force=True), 'wb') as pkl_file:
                    pickle.dump(self.encoder, pkl_file)
                DATA_STORE.update_hash('glove/twitter/dim{}'.format(self.dimension))

            else:
                with open(DATA_STORE['glove/twitter/dim{}'.format(self.dimension)], 'rb') as pkl_file:
                    self.encoder = pickle.load(pkl_file)
        else:
            raise ValueError('Error: Invalid GLoVe Version: {}, Must be one of {}'.format(
                version, GloveEmbedding.valid_versions))

    def _get_vec(self, input_string: str) -> np.ndarray:
        if input_string in self.encoder:
            return self.encoder[input_string]
        else:
            return self.encoder['unk']

    def GenerateMatrix(self, dictionary: Dict[str, int]) -> np.ndarray:
        # Determine the length of the embedding matrix
        log_message('Generating Embedding Matrix...')
        vocab_size = len(dictionary)
        self.embedding_matrix = np.zeros(shape=(vocab_size, self.dimension))
        for key in dictionary.keys():
            self.embedding_matrix[dictionary[key]] = self._get_vec(key)

        return self.embedding_matrix

    def get_word_vector(self, input_string: str) -> np.ndarray:
        return self._get_vec(input_string)
