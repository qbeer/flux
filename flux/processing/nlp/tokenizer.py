"""
Tokenizer base class for NLP applications
"""

import os
from abc import ABC, abstractmethod
from typing import List

from flux.backend.globals import ROOT_FPATH

# Handle NLTK imports
try:
    import nltk
    nltk.download('punkt', os.path.join(ROOT_FPATH, 'nltk'), quiet=True)
    nltk.data.path.append(os.path.join(ROOT_FPATH, 'nltk'))
    NLTK_IMPORTED = True
except Exception as ex:
    from flux.util.logging import log_warning
    log_warning('The PunktTokenizer requires NLTK. If you\'re using this, install NLTK using \'pip install nltk\'')
    NLTK_IMPORTED = False


class Tokenizer(ABC):
    """Simple tokenizer base class
    """

    def __init__(self,) -> None:
        pass

    @abstractmethod
    def parse(self, input_string: str) -> List[str]:
        pass


class DelimiterTokenizer(Tokenizer):
    """Tokenization based on split
    """

    def __init__(self, delimiter=' ') -> None:
        self.delimiter = delimiter

    def parse(self, input_string: str) -> List[str]:
        return input_string.split(self.delimiter)


class PunktTokenizer(Tokenizer):
    """Simple NLTK-based tokenizer
    """

    def __init__(self, remove_stopwords=False) -> None:
        if not NLTK_IMPORTED:
            raise NotImplementedError("NLTK Required to use the Punkt Tokenizer")

        self.remove_stopwords = remove_stopwords

        if self.remove_stopwords:
            from nltk.corpus import stopwords
            self.stop_words = set(stopwords.words('english'))

    def parse(self, input_string: str) -> List[str]:
        tokenized = nltk.word_tokenize(input_string)
        if not self.remove_stopwords:
            return tokenized
        else:
            return [w for w in tokenized if tokenized not in self.stop_words]
