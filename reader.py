import collections
import os
import tensorflow as tf


def read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def word2id(filename):
    words = read_words(filename)
    counter = collections.Counter(words)

    # sort with decreasing frequency, if frequency is same, sort by the word in lexicon order
    # result is in format like [word_1: freq_1,
    #                           word_2: freq_2,
    #                           ...,
    #                           word_n: freq_n]
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id = word2id(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocab_size = len(word_to_id)
    return train_data, valid_data, test_data, vocab_size


def make_batch(raw_data: list, batch_size: int, seq_len: int):
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

    data_len = tf.size(raw_data)
    n_batches = data_len // batch_size
    data = tf.reshape(raw_data[0: batch_size * n_batches],
                      [batch_size, n_batches])

    # number of steps in one epoch
    epoch_size = (n_batches - 1) // seq_len

    # create queue like [0, 1, ...., epoch_size - 1,
    #                    0, 1, ...., epoch_size - 1,
    #                    ..... ]
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

    # using element in x to predict element in y
    # i.e. using the current word (index) to predict the next word (index)
    x = tf.strided_slice(data, [0, i * seq_len],
                         [batch_size, (i + 1) * seq_len])
    x.set_shape([batch_size, seq_len])
    y = tf.strided_slice(data, [0, i * seq_len + 1],
                         [batch_size, (i + 1) * seq_len + 1])
    y.set_shape([batch_size, seq_len])
    return x, y


class Dataset(object):
    def __init__(self, config, data):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.data, self.targets = make_batch(
            data, batch_size, num_steps)
