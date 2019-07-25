import torch
import collections
import os


def read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def word2id(filename):
    words = read_words(filename)
    counter = collections.Counter(words)

    # sort with decreasing frequency, if frequency is same, sort by the word in lexicon order
    # result is in format like [word_1: freq_1,
    #                           word_2: freq_2,
    #                           ...,
    #                           word_n: freq_n]
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = word2id(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocab_size = len(word_to_id)
    return train_data, valid_data, test_data, vocab_size


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def clip(self, train, batch_size):
        num_batches = train.size(0) // batch_size
        train = train[:num_batches*batch_size]
        return train.view(batch_size, -1)

    def get_data(self, path, batch_size=20):
        print('Using data from ' + path)
        train, eval, test, vocab_size = map(torch.tensor, ptb_raw_data(path))
        return (self.clip(train, batch_size),
                self.clip(eval, batch_size),
                self.clip(test, batch_size),
                vocab_size)
