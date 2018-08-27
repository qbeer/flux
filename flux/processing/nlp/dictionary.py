"""
Dictionary class for constructing and maintaining
NLP token dictionaries
"""
from typing import Dict, List, Optional, Tuple

import pickle
import numpy as np

from flux.processing.nlp.tokenizer import Tokenizer, PunktTokenizer, DelimiterTokenizer
from flux.util.logging import log_message

class NLPDictionary(object):

    def __init__(self, tokenizer: str='punkt', dtype: np.dtype=np.int64) -> None:

        if tokenizer == 'punkt':
            self.tokenizer: Tokenizer = PunktTokenizer()
        elif tokenizer == 'space':
            self.tokenizer: Tokenizer = DelimiterTokenizer(delimiter=' ')
        else:
            raise ValueError("Invalid tokenizer!")

        self.word_dictionary: Dict[str, int] = {'': 0}
        self.char_dictionary: Dict[str, int] = {'': 0}
        self.word_dictionary_rev: Dict[int, str] = {0: ''}
        self.char_dictionary_rev: Dict[int, str] = {0: ''}
        self.dtype = dtype
        self.tokenizer_string = tokenizer

    def dense_parse_tokens(self, input_tokens: List[str], word_padding: Optional[int]=None, char_padding: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        # Handle the padding of the iterm
        if word_padding is not None:
            input_tokens = input_tokens[:word_padding]
            output_word_array = np.zeros(word_padding, dtype=self.dtype)
        else:
            output_word_array = np.zeros(len(input_tokens), dtype=self.dtype)

        # Build the return array
        if word_padding is not None:
            output_char_array = np.zeros(
                shape=(word_padding, char_padding), dtype=self.dtype)
            for idx, token in enumerate(input_tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                    self.word_dictionary_rev[self.word_dictionary[token]] = token
                output_word_array[idx] = self.word_dictionary[token]
                for cdx, c in enumerate(token[:char_padding]):
                    if c not in self.char_dictionary:
                        self.char_dictionary[c] = len(self.char_dictionary)
                        self.char_dictionary_rev[self.char_dictionary[c]] = c
                    output_char_array[idx, cdx] = self.char_dictionary[c]
        else:
            output_data: List[np.ndarray] = []
            for idx, token in enumerate(input_tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                    self.word_dictionary_rev[self.word_dictionary[token]] = token
                output_word_array[idx] = self.word_dictionary[token]

                tok_enc: List[int] = []
                if char_padding is not None:
                    token = token[:char_padding]
                for cdx, c in enumerate(token):
                    if c not in self.char_dictionary:
                        self.char_dictionary[c] = len(self.char_dictionary)
                        self.char_dictionary_rev[self.char_dictionary[c]] = c
                    tok_enc.append(self.char_dictionary[c])
                output_data.append(np.array(tok_enc))
            output_char_array = np.array(output_data)

        return (output_word_array, output_char_array), len(input_tokens)

    # Same as the token dense parse, but tokenize first
    def dense_parse(self, input_string: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.dense_parse_tokens(self.tokenizer.parse(input_string), **kwargs)

    def decode(self, array: np.ndarray) -> List[str]:
        output_list = []
        for val in array:
            if val in self.word_dictionary_rev:
                output_list.append(self.word_dictionary_rev[int(val)])
            else:
                output_list.append('<UNK>')
        return output_list

    def save(self, fpath) -> None:
        log_message('Saving dictionary to: {}'.format(fpath))
        with open(fpath, 'wb') as out_file:
            # Save the individual nec. elements
            save_dict = {
                'wd': self.word_dictionary,
                'wdr': self.word_dictionary_rev,
                'cd': self.char_dictionary,
                'cdr': self.char_dictionary_rev,
                'dtype': self.dtype,
                'tkn': self.tokenizer_string,
            }
            pickle.dump(save_dict, out_file)

    @staticmethod
    def load(fpath):
        with open(fpath, 'rb') as in_file:
            save_dict = pickle.load(in_file)
            return_dict = NLPDictionary(tokenizer=save_dict['tkn'],dtype=save_dict['dtype'])
            return_dict.char_dictionary = save_dict['cd']
            return_dict.word_dictionary = save_dict['wd']
            return_dict.char_dictionary_rev = save_dict['cdr']
            return_dict.word_dictionary_rev = save_dict['wdr']
        return return_dict
