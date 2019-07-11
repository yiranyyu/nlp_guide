import numpy as np
import pickle
import math
import time
import os
import warnings
import struct
from scipy.linalg.blas import saxpy
from model import Word2vecModel
from libc.stdlib cimport rand, malloc, free
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from multiprocessing import Pool, Value, Array

class VocabItem:
    def __init__(self, word: str, int count):
        self.word = word
        self.count = count
        self.code = None
        self.path = None    # it's faster to use path and code as list rather than numpy_array

    def __str__(self):
        return '%s, %d, %s, %s' % (self.word, self.count, self.code, self.path)


class Vocab:
    def __init__(self, corpus_file: str, min_count: int):
        print("Start learning vocab")
        t0 = time.time()
        cdef unsigned long long word_count = 0

        word_freq = {'</s>': min_count + 1}
        with open(corpus_file, 'rt', encoding='utf8') as data:
            for line in data:
                tokens = line.strip().split()
                for token in tokens:
                    word_freq[token] = word_freq.get(token, 0) + 1
                    word_count += 1

        cdef int freq, unk_freq = 0
        self.vocab = []
        self.word_count = word_count
        self.bytes = os.path.getsize(corpus_file)
        for word in word_freq:
            freq = word_freq[word]
            if freq >= min_count:
                self.vocab.append(VocabItem(word, freq))
            else:
                unk_freq += freq
        self.vocab.append(VocabItem('<unk>', unk_freq))
        self.__sort_and_init_index()
        print('Vocab learned in %.3lfm' % ((time.time() - t0) / 60))
        print('Total words in training file: %d' % word_count)
        print('Vocab size: %d' % len(self.vocab))

    def __getitem__(self, index: int):
        return self.vocab[index]

    def __iter__(self):
        return iter(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, word: str):
        return word in self.vocab

    def save_to_path(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load_from_path(path: str):
        return pickle.load(open(path, 'rb'))

    def __str__(self):
        rtn = 'size=%d\n' % len(self)
        rtn += '<str>,<cnt>,<code>,<path>\n'
        for token in self:
            rtn += str(token) + '\n'
        return rtn

    def __sort_and_init_index(self):
        """
        Sort in decreasing order with respect to the frequency of words
        """
        self.vocab.sort(key=lambda x: x.count, reverse=True)
        self.vocab_index = {}
        for idx, token in enumerate(self.vocab):
            self.vocab_index[token.word] = idx

    def indices(self, words):
        return [(self.vocab_index[word] if (word in self.vocab_index) else self.vocab_index['<unk>']) for word in words]

    def encode_huffman(self):
        vocab_size = len(self.vocab)
        count = [w.count for w in self.vocab] + [self.word_count + 1] * (vocab_size - 1)
        parent = [0, 0] * (vocab_size - 1)
        binary = [0, 0] * (vocab_size - 1)

        cdef int i, pos1, pos2
        pos1 = vocab_size - 1  # track words
        pos2 = vocab_size  # track generated nodes

        for i in range(vocab_size - 1):
            if pos1 >= 0 and count[pos1] < count[pos2]:
                min1 = pos1
                pos1 -= 1
            else:
                min1 = pos2
                pos2 += 1

            if pos1 >= 0 and count[pos1] < count[pos2]:
                min2 = pos1
                pos1 -= 1
            else:
                min2 = pos2
                pos2 += 1

            # count of ith generated node
            count[vocab_size + i] = count[min1] + count[min2]
            parent[min1] = parent[min2] = vocab_size + i
            binary[min1] = 0
            binary[min2] = 1

        root_idx = 2 * (vocab_size - 1)
        for idx, token in enumerate(self.vocab):
            path = []  # List of indices of words in the path from root to leaf as word
            code = []  # Binary Huffman code from leaf to the root (leaf exclusive)

            while idx < root_idx:
                if idx >= vocab_size:
                    path.append(idx)
                code.append(binary[idx])
                idx = parent[idx]
            path.append(root_idx)

            # reverse path and code to make it as from root to leaf
            # why subtracting vocab_size: index in path is index of no-leaf node
            # which is in range [vocab_size, 2 * vocab_size - 2), so result is
            # in range[0, vocab - 2)
            token.path = [i - vocab_size for i in path[::-1]]
            token.code = code[::-1]


class UnigramTable:
    """
    List of indices of words in the vocab following the distribution x^(3/4)
    """
    def __init__(self, vocab: Vocab):
        vocab_size = len(vocab)
        power = 3 / 4
        norm = sum([math.pow(x.count, power) for x in vocab])

        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)

        print('Start building uni_gram table')
        p = 0
        i = 0
        for j, uni_gram in enumerate(vocab):
            p += float(math.pow(uni_gram.count, power)) / norm
            while i < table_size and float(i) / table_size < p:
                table[i] = j
                i += 1
        self.table = table
        self.table_size = table_size
        print('End building uni_gram table')

    def sample(self, count):
        indices = np.random.randint(0, self.table_size, size=count)
        return [self.table[i] for i in indices]


cdef int exp_table_size = 1000, MAX_EXP = 6
cdef float exp_table[1000]
for i in range(exp_table_size):
    exp_table[i] = exp((i / float(exp_table_size) * 2 - 1) * MAX_EXP);  # Precompute the exp() table
    exp_table[i] = exp_table[i] / (exp_table[i] + 1);  # Precompute f(x) = x / (x + 1)

cpdef float sigmoid(float z):
    if z > MAX_EXP:
        return 1.0
    elif z < -MAX_EXP:
        return 0.0
    else:
        return exp_table[int((z + MAX_EXP) * (exp_table_size / MAX_EXP / 2))]

def init_embedding(dim, vocab_size):
    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)
    return syn0, syn1

def train_epoch(file, start, end, vocab, float lr, start_lr, table, neg, dim, syn0, syn1, current_epoch, epoch, win, cbow, float sample, global_word_count, batch_size_in_character):
    cdef int word_count = 0, last_word_count = 0, target, word, context_word
    cdef long long train_words = vocab.word_count, token_count = 0
    cdef float prob_to_keep, threshold = sample * train_words, p, label, g
    file.seek(start)
    theta = np.zeros(dim)
    while file.tell() < end:
        line = file.read(batch_size_in_character)
        if not line: continue

        sen = vocab.indices(line.split())
        for idx, token in enumerate(sen):
            word_count += 1

            # under sample frequent words
            if sample:
                token_count = vocab[token].count
                p = threshold / token_count
                prob_to_keep = (p ** 0.5) + p
                if prob_to_keep < np.random.rand():
                    continue

            if word_count % 10000 == 0:
                global_word_count.value += (word_count - last_word_count)
                last_word_count = word_count

                # theta lr
                lr = start_lr * (1 - float(global_word_count.value) / (epoch * vocab.word_count + 1))
                lr = max(lr, start_lr * 0.0001)
            current_win = np.random.randint(1, win + 1)
            context_beg = max(idx - current_win, 0)
            context_end = min(idx + current_win + 1, len(sen))
            context = sen[context_beg:idx] + sen[idx + 1:context_end]

            # CBOW
            if cbow:
                context_vector = np.mean(np.asarray([syn0[x] for x in context]), axis=0)
                theta.fill(0)
                if neg:
                    # pos sample
                    x = syn1[token]
                    p = sigmoid(np.dot(context_vector, x))
                    g = lr * (1 - p)
                    theta = saxpy(x, theta, a=g)
                    syn1[token] = saxpy(context_vector, x, a=g)

                    # neg sample
                    neg_samples = table.sample(neg)
                    for target in neg_samples:
                        x = syn1[target]
                        p = sigmoid(np.dot(context_vector, x))
                        g = lr * (-p)
                        theta = saxpy(x, theta, a=g)
                        syn1[target] = saxpy(context_vector, x, a=g)
                else:
                    targets = zip(vocab[token].path, vocab[token].code)
                    for target, label in targets:
                        x = syn1[target]
                        p = sigmoid(np.dot(context_vector, x))
                        g = lr * (label - p)
                        theta = saxpy(x, theta, a=g)
                        syn1[target] = saxpy(context_vector, x, a=g)
                for word in context:
                    syn0[word] = saxpy(theta, syn0[word], a=1)

            # Skip-gram
            else:
                for context_word in context:
                    theta.fill(0)
                    if neg:
                        # pos sample
                        x = syn1[token]
                        p = sigmoid(np.dot(syn0[context_word], x))
                        g = lr * (1 - p)
                        theta = saxpy(x, theta, a=g)
                        syn1[token] = saxpy(syn0[context_word], x, a=g)

                        # neg sample
                        neg_samples = table.sample(neg)
                        for target in neg_samples:
                            x = syn1[target]
                            p = sigmoid(np.dot(syn0[context_word], x))
                            g = lr * (-p)
                            theta = saxpy(x, theta, a=g)
                            syn1[target] = saxpy(syn0[context_word], x, a=g)
                    else:
                        targets = zip(vocab[token].path, vocab[token].code)
                        for target, label in targets:
                            x = syn1[target]
                            p = sigmoid(np.dot(syn0[context_word], x))
                            g = lr * (label - p)
                            theta = saxpy(x, theta, a=g)
                            syn1[target] = saxpy(syn0[context_word], x, a=g)
                    syn0[context_word] = saxpy(theta, syn0[context_word], a=1)
    return (current_epoch + 1), lr
