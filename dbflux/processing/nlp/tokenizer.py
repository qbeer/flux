"""
Tokenizer base class for NLP applications
"""

import os
from abc import ABC, abstractmethod
from typing import List

import nltk

from dbflux.backend.globals import ROOT_FPATH

# Handle NLTK imports
nltk.download('punkt', os.path.join(ROOT_FPATH, 'nltk'), quiet=True)
nltk.data.path.append(os.path.join(ROOT_FPATH, 'nltk'))


class Tokenizer(ABC):
    """Simple tokenizer base class
    """

    def __init__(self,) -> None:
        pass

    @abstractmethod
    def parse(self, input_string: str) -> List[str]:
        pass


class PunktTokenizer(Tokenizer):
    """Simple NLTK-based tokenizer
    """

    def __init__(self, ) -> None:
        pass

    def parse(self, input_string: str) -> List[str]:
        return nltk.word_tokenize(input_string)
