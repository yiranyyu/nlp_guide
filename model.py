import numpy as np
import sys
from sklearn.neighbors import KDTree

# TODO: make a map from <word> to <index_in_matrix> and a map from <index_in_matrix> to <word>
# use <matrix> of embedding to caculate nearest neighbor


class Word2vecModel(object):
    def __init__(self, path):
        with open(path, 'rt') as file:
            meta = file.readline()
            self.words, self.size = map(int, meta.split())
            self.word_index = {}
            self.index_word = []

            embeddings = []
            for line in file:
                word, *vector = line.split(' ')
                vector = [float(item) for item in vector]
                vector = np.asarray(vector, dtype=np.float)
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
                self.word_index[word] = len(embeddings)
                self.index_word.append(word)
                embeddings.append(vector)
        self.embedding = np.asarray(embeddings)
        self.kd = KDTree(self.embedding, leaf_size=10)
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
        return self.embedding[self.word_index[word]]

    def similarity(self, a: str, b: str):
        vec = self[a] - self[b]
        return np.linalg.norm(vec)

    def nearest_word(self, word: str):
        vec = self[word]
        return self.nearset_word_of_embedding(vec)

    def nearset_word_of_embedding(self, embedding):
        idx = self.kd.query([embedding], return_distance=False)
        return self.index_word[idx]

    def analogy(self, a: str, b: str, c: str):
        #  d= c + b - a
        d = self[c] - (self[a] - self[b])
        norm = np.linalg.norm(d)
        if norm:
            d = d / norm
        return self.nearset_word_of_embedding(d)
