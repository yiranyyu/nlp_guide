import numpy as np
import pickle
from libc.stdlib cimport rand, malloc, free
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI

cdef extern from "limits.h":
    int INT_MAX

cdef class VocabItem:
    cdef int count, index
    cdef str word
    def __init__(self, word, int count, int index):
        self.word = word
        self.count = 0
        self.index = index

cdef class Vocab:
    cdef dict vocab
    cdef int word_count
    def __init__(self, corpus_file: str, min_count, f):
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

    cpdef save_to_path(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load_from_path(path: str):
        return pickle.load(open(path, 'rb'))

    def __str__(self):
        return 'Vocab size: %d' % len(self.vocab)

cdef int exp_table_size = 1000, MAX_EXP = 6
cdef float exp_table[1000]
for i in range(exp_table_size):
    exp_table[i] = exp((i / float(exp_table_size) * 2 - 1) * MAX_EXP); # Precompute the exp() table
    exp_table[i] = exp_table[i] / (exp_table[i] + 1);                   # Precompute f(x) = x / (x + 1)

cpdef sigmoid(float z):
    if z > MAX_EXP:
        return 1.0
    elif z < -MAX_EXP:
        return 0.0
    else:
        return exp_table[int((z + MAX_EXP) * (exp_table_size / MAX_EXP / 2))]

if __name__ == '__main__':
    pass
