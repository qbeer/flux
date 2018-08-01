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
        self.word_maxlen = word_maxlen
        self.char_maxlen = char_maxlen
        self.pad_output = pad_output
        self.dtype = dtype

        if self.pad_output and (self.char_maxlen is None or self.word_maxlen is None):
            raise ValueError(
                "Cannot pad output if no maximum character/word length specified.")

    def word_parse(self, input_string: str) -> np.ndarray:
        # Tokenize the sentence
        tokens = self.tokenizer.parse(input_string)

        if self.word_maxlen is not None:
            tokens = tokens[:self.word_maxlen]

        # Build the return array
        if self.pad_output:
            output_array = np.zeros(self.word_maxlen, dtype=self.dtype)
        else:
            output_array = np.zeros(len(tokens), dtype=self.dtype)
        for idx, token in enumerate(tokens):
            if token not in self.word_dictionary:
                self.word_dictionary[token] = len(self.word_dictionary)
            output_array[idx] = self.word_dictionary[token]

        return output_array

    def char_parse(self, input_string: str) -> np.ndarray:
        # Tokenize the sentence
        tokens = self.tokenizer.parse(input_string)

        if self.word_maxlen is not None:
            tokens = tokens[:self.word_maxlen]

        # Build the return array
        if self.pad_output:
            output_array = np.zeros(
                shape=(self.word_maxlen, self.char_maxlen), dtype=self.dtype)
            for idx, token in enumerate(tokens):
                for cdx, c in enumerate(token[:self.char_maxlen]):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    output_array[idx, cdx] = self.char_dictioanary[c]
        else:
            output_data: List[np.ndarray] = []
            for idx, token in enumerate(tokens):
                tok_enc: List[int] = []
                if self.char_maxlen is not None:
                    token = token[:self.char_maxlen]
                for cdx, c in enumerate(token):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    tok_enc.append(self.char_dictioanary[c])
                output_data.append(np.array(tok_enc))
            output_array = np.array(output_data)
        return output_array

    def dense_parse(self, input_string: str) -> Tuple[np.ndarray, np.ndarray]:
        # Tokenize the sentence
        tokens = self.tokenizer.parse(input_string)

        if self.word_maxlen is not None:
            tokens = tokens[:self.word_maxlen]

        if self.pad_output:
            output_word_array = np.zeros(self.word_maxlen, dtype=self.dtype)
        else:
            output_word_array = np.zeros(len(tokens), dtype=self.dtype)

        # Build the return array
        if self.pad_output:
            output_char_array = np.zeros(
                shape=(self.word_maxlen, self.char_maxlen), dtype=self.dtype)
            for idx, token in enumerate(tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                output_word_array[idx] = self.word_dictionary[token]
                for cdx, c in enumerate(token[:self.char_maxlen]):
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
                if self.char_maxlen is not None:
                    token = token[:self.char_maxlen]
                for cdx, c in enumerate(token):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    tok_enc.append(self.char_dictioanary[c])
                output_data.append(np.array(tok_enc))
            output_char_array = np.array(output_data)

        return (output_word_array, output_char_array)

    def dense_parse_tokens(self, input_tokens: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        # Tokenize the sentence
        tokens = input_tokens

        if self.word_maxlen is not None:
            tokens = tokens[:self.word_maxlen]

        if self.pad_output:
            output_word_array = np.zeros(self.word_maxlen, dtype=self.dtype)
        else:
            output_word_array = np.zeros(len(tokens), dtype=self.dtype)

        # Build the return array
        if self.pad_output:
            output_char_array = np.zeros(
                shape=(self.word_maxlen, self.char_maxlen), dtype=self.dtype)
            for idx, token in enumerate(tokens):
                if token not in self.word_dictionary:
                    self.word_dictionary[token] = len(self.word_dictionary)
                output_word_array[idx] = self.word_dictionary[token]
                for cdx, c in enumerate(token[:self.char_maxlen]):
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
                if self.char_maxlen is not None:
                    token = token[:self.char_maxlen]
                for cdx, c in enumerate(token):
                    if c not in self.char_dictioanary:
                        self.char_dictioanary[c] = len(self.char_dictioanary)
                    tok_enc.append(self.char_dictioanary[c])
                output_data.append(np.array(tok_enc))
            output_char_array = np.array(output_data)

        return (output_word_array, output_char_array)
