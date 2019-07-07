import numpy as np
import pickle
import math
import time
import struct
from .model import Word2vecModel
from libc.stdlib cimport rand, malloc, free
from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI
from multiprocessing import Pool, Value, Array

#cython: language_level=3, boundscheck=False
cdef extern from "limits.h":
    int INT_MAX

cdef class VocabItem:
    cdef int count
    cdef str word, code
    cdef list path
    def __init__(self, word: str, int count):
        self.word = word
        self.count = 0
        self.code = None
        self.path = None

cdef class Vocab:
    cdef list vocab
    cdef dict vocab_index
    cdef int word_count
    def __init__(self, corpus_file: str, min_count):
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
        self.vocab = [VocabItem('<unk>', 0)]
        self.vocab_index = {'<unk>': 0}
        self.word_count = word_count
        for word in word_freq:
            freq = word_freq[word]
            if freq >= min_count:
                self.vocab_index[word] = next_index
                self.vocab.append(VocabItem(word, freq))
                next_index += 1
        print('Total words in training file: %d' % word_count)
        print('Vocab size: %d' % len(self.vocab))

    def __getitem__(self, word: str):
        return self.vocab[word]

    def __iter__(self):
        return iter(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, word: str):
        return word in self.vocab

    cpdef save_to_path(self, path: str):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load_from_path(path: str):
        return pickle.load(open(path, 'rb'))

    def __str__(self):
        return 'Vocab size: %d' % len(self.vocab)

    def __sort_by_freq(self):
        """
        Sort in decreasing order with respect to the frequency of words
        """
        self.vocab.sort(key=lambda x: x.count, reverse=True)
        self.vocab_index = {}
        for idx, token in enumerate(self.vocab):
            self.vocab_index[token.word] = idx

    def indices(self, words):
        return [(self.vocab_index[word] if (word in self.vocab_index) else self.vocab_index['<unk>']) for word in words]

    cdef encode_huffman(self):
        self.__sort_by_freq()
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
            # TODO: why subtracting vocab_size
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

def __init_process(*args):
    pass

def init_embedding(dim, vocab_size):
    tmp = np.random.uniform(low=-0.5 / dim, high=0.5 / dim, size=(vocab_size, dim))
    syn0 = np.ctypeslib.as_ctypes(tmp)
    syn0 = Array(syn0._type_, syn0, lock=False)

    tmp = np.zeros(shape=(vocab_size, dim))
    syn1 = np.ctypeslib.as_ctypes(tmp)
    syn1 = Array(syn1._type_, syn1, lock=False)
    return syn0, syn1

def train_process(pid, file_size, num_process, file, lr, epoch, vocab, global_word_count, win, cbow):
    start = file_size / num_process * pid
    end = file_size if pid == num_process - 1 else file_size / num_process * (pid + 1)

    start_lr = lr
    word_count = 0
    last_word_count = 0
    current_epoch = 0
    while current_epoch < epoch:
        file.seek(start)
        current_epoch += 1
        print('Start [%d epoch] of process %d, lr: %f' % (current_epoch, pid, lr))
        while file.tell() < end:
            line = file.readline().strip()
            if not line: continue

            sen = vocab.indices(line.split())
            for idx, token in enumerate(sen):
                if word_count % 10000 == 0:
                    global_word_count.value += (word_count - last_word_count)
                    last_word_count = word_count

                    # update lr
                    lr = start_lr * (1 - float(global_word_count.value) / (epoch * vocab.word_count + 1))
                    lr = max(lr, start_lr * 0.0001)
                current_win = np.random.randint(1, win + 1)
                context_beg = max(idx - current_win, 0)
                context_end = min(idx + current_win + 1, len(sen))
                context = sen[context_beg:idx] + sen[idx + 1:context_end]

                # CBOW
                if cbow:
                    pass  # TODO:
                # Skip-gram

def train(corpus_file: str, dim: int, min_count: int, num_processes: int, save_path: str, neg=None):
    vocab = Vocab(corpus_file, min_count)
    syn0, syn1 = init_embedding(dim, len(vocab))
    global_word_count = Value('i', 0)

    table = None
    if neg:
        print('Init Negative Sampling')
        table = UnigramTable(vocab)
    else:
        print('Init Hierarchycal Softmax')
        vocab.encode_huffman()

    print('Begin training')
    t0 = time.time()
    pool = Pool(processes=num_processes, initializer=__init_process,
                initargs=())
    pool.map(train_process, range(num_processes))
    t1 = time.time()
    print('\nCompleted training. Time consumption is %.3lf\n', (t1 - t0) / 60)

    Word2vecModel.save(vocab, syn0, save_path)
    print('Model saved to ' + save_path)

if __name__ == '__main__':
    pass
