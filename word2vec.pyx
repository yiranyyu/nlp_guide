import numpy as np
import pickle
from libc.stdlib cimport rand, malloc, free
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

cdef extern from "limits.h":
    int INT_MAX

cdef class VocabItem:
    cdef int count, index
    def __init__(self, word, int count, int index):
        self.word = word
        self.count = 0
        self.index = index

cdef class Vocab:
    def __init__(self, corpus_file: str, int min_count, float f):
        print("Start learning vocab")
        cdef unsigned long long word_count = 0

        word_freq = {'</s>': min_count + 1}
        with open(corpus_file, 'rt', encoding='utf8') as data:
            for line in data:
                tokens = line.strip().split()
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
                    word_count += 1

        cdef int next_index = 1, freq
        self.vocab = {}
        self.word_count = word_count
        for word in word_freq:
            freq = word_freq[word]
            if freq >= min_count:
                self.vocab[word] = VocabItem(word, freq, next_index)
                next_index += 1
        print('Total words in training file: %d' % word_count)
        print('Vocab size: %d' % len(self.vocab))

    def __getitem__(self, word: str):
        return self.vocab[word]

    def __iter__(self):
        return iter(self.vocab)

    def __contains__(self, word: str):
        return word in self.vocab

    cdef save_to_path(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    cdef load_from_path(path: str):
        return pickle.load(path)

if __name__ == '__main__':
    pass
