import argparse
import math
import struct
import sys
import os
import time
import warnings
import numpy as np
from multiprocessing import Pool, Value, Array
from word2vec_inner import sigmoid, Vocab, UnigramTable, init_embedding, train_epoch
from model import Word2vecModel


def train_process(pid, epoch, cbow, neg, dim, lr, win, num_processes, corpus_file, sample: float, batch_size_in_character: int):
    # Set file to point to the right chunk of training file
    file_size = os.path.getsize(corpus_file)
    start = file_size / num_processes * pid
    end = file_size if pid == num_processes - 1 else file_size / num_processes * (pid + 1)

    start_lr = lr
    current_epoch = 1
    while current_epoch <= epoch:
        print(('Start [%d epoch] of process %d, lr: %f' % (current_epoch, pid, lr)), flush=True)
        current_epoch, lr = train_epoch(file, start, end, vocab, lr, start_lr, table, neg, dim, syn0, syn1, current_epoch, epoch, win, cbow, sample, global_word_count, batch_size_in_character)

    file.close()

def __init_process(*args):
    global syn0, syn1, global_word_count, vocab, file, table
    syn0, syn1, global_word_count, vocab, corpus_file, table = args
    file = open(corpus_file, 'r')
    syn0 = np.ctypeslib.as_array(syn0)
    syn1 = np.ctypeslib.as_array(syn1)


def train(corpus_file: str, dim: int, min_count: int, num_processes: int, save_path: str, lr: float, win: int,
          epoch: int, neg=None, cbow=None, sample=None, batch_size_in_character=1_00_000):
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
                initargs=(syn0, syn1, global_word_count, vocab, corpus_file, table))
    pids = [[x] for x in range(num_processes)]
    args = [epoch, cbow, neg, dim, lr, win, num_processes, corpus_file, sample, batch_size_in_character]
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
    parser.add_argument('-sample',
                        help='Set threshold for occurrence of words. Those that paper with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n',
                        dest='sample', default=1e-3, type=float)
    parser.add_argument('-epoch', help='Number of training epochs', dest='epoch', default=1, type=int)
    parser.add_argument('-batch_size_in_character', help='Batch size to read, larger size requires larger memory but also less IO operation', dest='batch_size_in_character', default=1_00_000, type=int)
    args = parser.parse_args()
    if args.batch_size_in_character < 1_00_000:
        args.batch_size_in_character = 1_00_000

    train(args.training_file, args.dim, args.min_count, args.num_processes, args.model_save_path, args.lr, args.win, args.epoch, args.neg, args.cbow, args.sample, args.batch_size_in_character)