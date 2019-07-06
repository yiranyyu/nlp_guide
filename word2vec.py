from word2vec_inner import *


vocab_path = './temp/vocab'
data_path = './data/word2vec/训练语料.txt'
# Vocab(data_path, 5, 0.1).save_to_path(vocab_path)
print(Vocab.load_from_path(vocab_path))
