import numpy as np
import sys


class Word2vecModel(object):
    def __init__(self, path):
        with open(path, 'rt') as file:
            meta = file.readline()
            words, size = meta.split(' ')
            self.words, self.size = int(words), int(size)

            self.embedding = {}
            for line in file:
                word, *vector = line.split(' ')
                vector = [float(item) for item in vector]
                vector = np.asarray(vector, dtype=np.float)
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / np.linalg.norm(vector)
                self.embedding[word] = vector
        print('Model init with %d words, embbing_size=%d' %
              (self.words, self.size))

    def __getitem__(self, word):
        return self.embedding[word]

    def similarity(self, a: str, b: str):
        vec = self[a] - self[b]
        return np.linalg.norm(vec)

    def nearest_word(self, word: str):
        dist = 0
        nearest = ''
        for that in self.embedding.keys():
            if that == word:
                continue
            curr = self.similarity(word, that)
            if curr > dist:
                dist = curr
                nearest = that
        return nearest
