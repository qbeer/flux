"""
Fast-Text embedding vectors
"""
try:
    import fastText
except Exception as ex:
    from flux.util.logging import log_warning
    log_warning('FastText vectors require the fastText python module to be installed. Obtain and install from here: https://github.com/facebookresearch/fastText')
    raise ex

import numpy as np

from typing import Dict


class FastTextEmbedding():

    def __init__(self, model_path:str='model.bin') -> None:

        # Need to download the model from Philippe
        self.model = fastText.load_model(model_path)
        self.dimension = self.model.get_dimension()

    def get_word_vector(self,input_string: str) -> np.ndarray:
        return self.model.get_word_vector(input_string)

    def get_sentence_vector(self, input_sentence: str) -> np.ndarray:
        return self.model.get_sentence_vector(input_sentence)

    def GenerateMatrix(self, dictionary: Dict[str, int]) -> np.ndarray:
        # Determine the length of the embedding matrix
        vocab_size = len(dictionary)
        self.embedding_matrix = np.zeros(shape=(vocab_size, self.dimension))
        for key in dictionary.keys():
            self.embedding_matrix[dictionary[key]] = self.get_word_vector(key)

        return self.embedding_matrix
