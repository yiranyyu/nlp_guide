from word2vec_model import Word2vecModel
from scipy.stats import spearmanr
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Model path', dest='model_path', required=True)
    model_path = parser.parse_args().model_path
    data_path = './data/word2vec/词相似度_wordsim-353.txt'
    model = Word2vecModel(model_path)
    a, b, c, d = [], [], [], []
    with open(data_path, 'rt') as file:
        for line in file:
            e, f, g = line.lower().split('\t')
            a.append(e)
            b.append(f)
            c.append(float(g))
            d.append(model.similarity(e, f))
    c = np.asarray(c)
    d = np.asarray(d)
    rho, pvalue = spearmanr(c, d)
    print('rho = ', rho, '\n pvalue =', pvalue)