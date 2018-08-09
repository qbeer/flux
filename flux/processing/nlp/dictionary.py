"""
Dictionary class for constructing and maintaining
NLP token dictionaries
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from flux.processing.nlp.tokenizer import Tokenizer, PunktTokenizer, DelimiterTokenizer


class NLPDictionary():

    def __init__(self, tokenizer: str='punkt', char_maxlen: Optional[int]=None,
                 word_maxlen: Optional[int]=None, pad_output: bool=True,
                 dtype: np.dtype=np.int64) -> None:

        if tokenizer == 'punkt':
            self.tokenizer: Tokenizer = PunktTokenizer()
        elif tokenizer == 'space':
            self.tokenizer: Tokenizer = DelimiterTokenizer(delimiter=' ')
        else:
            raise ValueError("Invalid tokenizer!")

        self.word_dictionary: Dict[str, int] = {'': 0}
        self.char_dictioanary: Dict[str, int] = {'': 0}
        self.dtype = dtype

    def dense_parse_tokens(self, input_tokens: List[str], word_padding: Optional[int]=None, char_padding: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        # Tokenize the sentence
        tokens = input_tokens

        if word_padding is not None:
            tokens = tokens[:word_padding]
            output_word_array = np.zeros(word_padding, dtype=self.dtype)
        else:
            output_word_array = np.zeros(len(tokens), dtype=self.dtype)

        # Build the return array
        if word_padding is not None:
            output_char_array = np.zeros(
                shape=(word_padding, char_padding), dtype=self.dtype)
            for idx, token in enumerate(tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                output_word_array[idx] = self.word_dictionary[token]
                for cdx, c in enumerate(token[:char_padding]):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    output_char_array[idx, cdx] = self.char_dictioanary[c]
        else:
            output_data: List[np.ndarray] = []
            for idx, token in enumerate(tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                output_word_array[idx] = self.word_dictionary[token]

                tok_enc: List[int] = []
                if char_padding is not None:
                    token = token[:char_padding]
                for cdx, c in enumerate(token):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    tok_enc.append(self.char_dictioanary[c])
                output_data.append(np.array(tok_enc))
            output_char_array = np.array(output_data)

        return (output_word_array, output_char_array), len(tokens)

    # Same as the token dense parse, but tokenize first
    def dense_parse(self, input_string: str, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.dense_parse_tokens(self.tokenizer.parse(input_string), **kwargs)
