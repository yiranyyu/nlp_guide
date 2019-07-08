import argparse
import math
import struct
import sys
import time
import warnings
import numpy as np
from multiprocessing import Pool, Value, Array
from word2vec_inner import sigmoid, Vocab, UnigramTable, init_embedding, train_epoch
from model import Word2vecModel


def train_process(pid, epoch, vocab, table, cbow, neg, dim, lr, win, num_processes, corpus_file):
    # Set file to point to the right chunk of training file
    file = open(corpus_file, 'r')
    start = vocab.bytes / num_processes * pid
    end = vocab.bytes if pid == num_processes - 1 else vocab.bytes / num_processes * (pid + 1)

    start_lr = lr
    current_epoch = 1
    while current_epoch <= epoch:
        print(('Start [%d epoch] of process %d, lr: %f' % (current_epoch, pid, lr)), flush=True)
        current_epoch, lr = train_epoch(file, start, end, vocab, lr, start_lr, table, neg, dim, syn0, syn1, current_epoch, epoch, win, cbow)

    file.close()

def __init_process(*args):
    global syn0, syn1, global_word_count
    syn0, syn1, global_word_count = args

def train(corpus_file: str, dim: int, min_count: int, num_processes: int, save_path: str, lr: float, win: int,
          epoch: int, neg=None, cbow=None):
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
                initargs=( syn0, syn1, global_word_count))
    pids = [[x] for x in range(num_processes)]
    args = [epoch, vocab, table, cbow, neg, dim, lr, win, num_processes, corpus_file]
    pool.starmap(train_process, [pid + args for pid in pids])
    t1 = time.time()
    print('\nCompleted training. Time consumption is %.3lfm\n' % ((t1 - t0) / 60))

    Word2vecModel.save(vocab, syn0, save_path)
    print('Model saved to ' + save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', help='Training file', dest='training_file', required=True)
    parser.add_argument('-model', help='Output model file', dest='model_save_path', required=True)
    parser.add_argument('-cbow', help='1 for CBOW, 0 for skip-gram', dest='cbow', default=1, type=int)
    parser.add_argument('-negative', help='Number of negative examples (>0) for negative sampling, 0 for hierarchical softmax', dest='neg', default=5, type=int)
    parser.add_argument('-dim', help='Dimensionality of word embeddings', dest='dim', default=300, type=int)
    parser.add_argument('-lr', help='Starting alpha', dest='lr', default=0.025, type=float)
    parser.add_argument('-window', help='Max window length', dest='win', default=5, type=int)
    parser.add_argument('-min-count', help='Min count for words used to learn <unk>', dest='min_count', default=5, type=int)
    parser.add_argument('-processes', help='Number of processes', dest='num_processes', default=1, type=int)
    parser.add_argument('-binary', help='1 for output model in binary format, 0 otherwise', dest='binary', default=0, type=int)
    parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    args = parser.parse_args()

    train(args.training_file, args.dim, args.min_count, args.num_processes, args.model_save_path, args.lr, args.win, args.epoch, args.neg, args.cbow)