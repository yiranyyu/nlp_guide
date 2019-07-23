import time
import numpy as np
import tensorflow as tf
import reader
import util
from tensorflow.python.client import device_lib

tf.compat.v1.enable_eager_execution()
print("Eager mode: %s" % tf.executing_eagerly())
logging = tf.logging

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

# get configuration
FLAGS = flags.FLAGS
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"


def data_type():
    return tf.float32


class PTBInput(object):
    def __init__(self, config, data):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.ptb_producer(
            data, batch_size, num_steps)


class PTBModel(object):
    def __init__(self, is_training, config, input_):
        self._is_training = is_training
        self.input = input_
        self._rnn_params = None
        self._cell = None
        self.batch_size = input_.batch_size
        self.num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        with tf.device("/cpu:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(
            logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self.cost = tf.reduce_sum(loss)
        self.final_state = state

        if not is_training:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell."""
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units=config.hidden_size,
            input_size=config.hidden_size,
            dropout=1 - config.keep_prob if is_training else 0)
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
            "lstm_params",
            initializer=tf.random_uniform(
                [params_size_t], -config.init_scale, config.init_scale),
            validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                     tf.float32)
        self.initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config):
        return tf.contrib.rnn.LSTMBlockCell(
            config.hidden_size, forget_bias=0.0)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        def make_cell():
            cell = self._get_lstm_cell(config)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(
                    cell, output_keep_prob=config.keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(config.batch_size, data_type())
        state = self.initial_state
        # Simplified version of tf.nn.static_rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use tf.nn.static_rnn() or tf.nn.static_state_saving_rnn().
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        # outputs, state = tf.nn.static_rnn(cell, inputs,
        #                                   initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self.cost}
        if self._is_training:
            ops.update(lr=self.lr, new_lr=self._new_lr,
                       lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self.initial_state_name = util.with_prefix(self._name, "initial")
        self.final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self.initial_state, self.initial_state_name)
        util.export_state_tuples(self.final_state, self.final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self.train_op = tf.get_collection_ref("train_op")[0]
            self.lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(
                    tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self.cost = tf.get_collection_ref(
            util.with_prefix(self._name, "cost"))[0]
        num_replicas = 1
        self.initial_state = util.import_state_tuples(
            self.initial_state, self.initial_state_name, num_replicas)
        self.final_state = util.import_state_tuples(
            self.final_state, self.final_state_name, num_replicas)


class SmallConfig(object):
    """Small config."""
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
    rnn_mode = BLOCK


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    nth_epoch_to_dacay_lr = 1
    epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000
    rnn_mode = BLOCK


def run_epoch(session, model, eval_op=None, verbose=False):
    start_time = time.time()
    total_cost = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        total_cost += cost
        iters += model.input.num_steps

        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / model.input.epoch_size, np.exp(total_cost / iters),
                   iters * model.input.batch_size * 1 /
                   (time.time() - start_time)))

    return np.exp(total_cost / iters)


def get_config():
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        config = SmallConfig()
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    return config


# program entry
def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = PTBInput(
                config=config, data=train_data)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = PTBModel(is_training=True, config=config,
                                       input_=train_input)
            tf.summary.scalar("Training Loss", train_model.cost)
            tf.summary.scalar("Learning Rate", train_model.lr)

        with tf.name_scope("Valid"):
            valid_input = PTBInput(
                config=config, data=valid_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = PTBModel(is_training=False,
                                       config=config, input_=valid_input)
            tf.summary.scalar("Validation Loss", valid_model.cost)

        with tf.name_scope("Test"):
            test_input = PTBInput(
                config=eval_config, data=test_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = PTBModel(is_training=False, config=eval_config,
                                      input_=test_input)

        models = {"Train": train_model,
                  "Valid": valid_model, "Test": test_model}
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session(config=tf.ConfigProto()) as session:
            for i in range(config.epoch):
                lr_decay = config.lr_decay ** max(i +
                                                  1 - config.nth_epoch_to_dacay_lr, 0)
                train_model.assign_lr(session, config.learning_rate * lr_decay)

                print("Epoch: %d Learning rate: %.3f" %
                      (i + 1, session.run(train_model.lr)))
                train_perplexity = run_epoch(session, train_model, eval_op=train_model.train_op,
                                             verbose=True)
                print("Epoch: %d Train Perplexity: %.3f" %
                      (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, valid_model)
                print("Epoch: %d Valid Perplexity: %.3f" %
                      (i + 1, valid_perplexity))

            print('Tesing')
            test_perplexity = run_epoch(session, test_model)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                sv.saver.save(session, FLAGS.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    # start with main function call
    tf.app.run()
