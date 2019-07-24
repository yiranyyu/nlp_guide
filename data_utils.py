import torch
import os
from reader import ptb_raw_data


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
        train, eval, test, _ = map(torch.tensor, ptb_raw_data(path))
        return (self.clip(train, batch_size),
                self.clip(eval, batch_size),
                self.clip(test, batch_size))
