
from collections import defaultdict, Counter
from itertools import chain
import pandas as pd

class SequenceEncoder:
    def __init__(self, max_sequence_len=None, vocab_len=None, special_tokens=['<PAD>','<UNK>'], padding=True):
        self.special_tokens = special_tokens
        self.vocab_len = vocab_len
        self.counter = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.max_sequence_len = max_sequence_len
        self.padding = padding

    def fit(self, X: list):
        self.counter = Counter(list(chain(*X)))
        if self.vocab_len is None:
            sorted_words = self.counter.most_common()
        else:
            sorted_words = self.counter.most_common(self.vocab_len)

        self.idx_to_word = {idx: token for idx, token in enumerate(self.special_tokens)}
        self.idx_to_word.update({(idx + len(self.special_tokens)): token for idx, (token, count)\
                                 in enumerate(sorted_words)})
        self.word_to_idx = {value: key for key, value in self.idx_to_word.items()}

    def transform(self, X):
        if isinstance(X,pd.Series):
            return X.apply(self._transform)
        if type(X) == list:
            return [self._transform(sequence) for sequence in X]


    def _transform(self, sequence):

        processed_list = [self.word_to_idx.get(token,0) for token in sequence]
        processed_list = processed_list[:self.max_sequence_len]
        return processed_list + (self.max_sequence_len - len(processed_list))*[self.word_to_idx[self.special_tokens[0]]]

    def reverse(self, X):
        if isinstance(X,pd.Series):
            return X.apply(self._reverse)
        if type(X) == list:
            return [self._reverse(sequence) for sequence in X]

    def _reverse(self, sequence):
        processed_list = [self.idx_to_word.get(token,0) for token in sequence ]
        processed_list = [token for token in processed_list if token != self.special_tokens[0]]
        return processed_list