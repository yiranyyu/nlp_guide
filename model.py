import numpy as np
import sys


class Word2vecModel(object):
    def __init__(self, path):
        with open(path, 'rt') as file:
            meta = file.readline()
            self.words, self.size = map(int, meta.split())

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

    @staticmethod
    def save(vocab, syn0, path: str):
        """
        vocab: type of Vocab in word2vec_inner.pyx
        """
        print('Saving model to ', path)
        dim = len(syn0[0])
        with open(path, 'wt', encoding='utf8') as file:
            file.write('%d %d\n' % (len(syn0), dim))
            for token, vector in zip(vocab, syn0):
                file.write('%s ' % token.word)
                file.write(str(x) for x in vector)
                file.write('\n')

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
