import tensorflow as tf


def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        # print('Export %s to %s' % (state_tuple, name))
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
    restored = []
    for i in range(len(state_tuples) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
    return tuple(restored)


def with_prefix(prefix, name):
    """Adds prefix to name."""
    return "/".join((prefix, name))


def get_args():
    # Get command line arguments
    flags = tf.flags

    # first argument is the dest variable
    # second argument is default value
    # third argument is description
    flags.DEFINE_string(
        "model", "small",
        "A type of model. Possible options are: small, medium, large.")
    flags.DEFINE_string("data_path", None,
                        "Where the training/test data is stored.")
    flags.DEFINE_string("save_path", None,
                        "Model output directory.")
    return flags.FLAGS


def data_type():
    return tf.float32


class Config(object):
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200

    # nth_epoch_to_dacay_lr = 4
    # epoch = 13
    nth_epoch_to_dacay_lr = 1
    epoch = 2

    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    float = tf.float32

    def __init__(self, args, eval=False):
        if eval:
            self.batch_size = 1
            self.num_steps = 1
