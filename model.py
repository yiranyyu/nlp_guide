import numpy as np
import sys


# TODO: make a map from <word> to <index_in_matrix> and a map from <index_in_matrix> to <word>
# use <matrix> of embedding to caculate nearest neighbor
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

    def __getitem__(self, word: str):
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

    def nearset_word_of_embedding(self, embedding):
        dist = 0
        nearest = ''
        for word in self.embedding.keys():
            curr = self[word].dot(embedding)
            if curr > dist:
                dist = curr
                nearest = word
        return nearest

    def analogy(self, a: str, b: str, c: str):
        #  d= c + b - a
        d = self[c] - (self[a] - self[b])
        dist = 0
        nearest = ''
        for word in self.embedding.keys():
            if word in [a, b, c]:
                continue
            curr = self[word].dot(d)
            if curr > dist:
                dist = curr
                nearest = word
        return nearest
