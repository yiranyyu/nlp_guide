import numpy as np
import sys
from sklearn.neighbors import KDTree

class Word2vecModel(object):
    def __init__(self, path):
        with open(path, 'rt') as file:
            meta = file.readline()
            self.words, self.size = map(int, meta.split())
            self.word_index = {}

            index_word = []
            embeddings = []
            for line in file:
                word, *vector = line.split(' ')
                vector = [float(item) for item in vector]
                vector = np.asarray(vector, dtype=np.float)
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
                self.word_index[word] = len(embeddings)
                index_word.append(word)
                embeddings.append(vector)
        self.embeddings = np.asarray(embeddings)
        self.index_word = index_word
        self.kd = KDTree(self.embeddings, leaf_size=40)
        print('Embedding shape = %s' % str(self.embeddings.shape))
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
        return self.embeddings[self.word_index[word]]

    def similarity(self, a: str, b: str):
        vec = self[a] - self[b]
        return np.linalg.norm(vec)

    def nearest_word(self, word: str):
        vec = self[word]
        return self.nearset_word_of_embedding(vec)

    def nearset_word_of_embedding(self, embedding):
        idx = self.kd.query([embedding], return_distance=False)[0][0]
        return self.index_word[idx]

    def analogy(self, a: str, b: str, c: str):
        #  d= c + b - a
        d = self[c] - (self[a] - self[b])
        norm = np.linalg.norm(d)
        if norm:
            d = d / norm

        # rtn = ''
        # dis = 0
        # for word, vec in zip(self.index_word, self.embeddings):
        #     if word in [a, b, c]:
        #         continue
        #     cur_dis = np.dot(d, vec)
        #     if cur_dis > dis:
        #         dis = cur_dis
        #         rtn = word
        # return rtn

        indices = self.kd.query([d], return_distance=False, k=4)[0]
        for index in indices:
            word = self.index_word[index]
            if word not in [a, b, c]:
                return word