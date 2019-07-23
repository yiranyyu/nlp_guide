import time
import numpy as np
import tensorflow as tf
import reader
import util

args = util.get_args()
tf.compat.v1.enable_eager_execution()
print("Eager mode: %s" % tf.executing_eagerly())


class LSTMModel(object):
    def __init__(self, is_training, config, input_data):
        self.is_training = is_training
        self.input = input_data
        self.batch_size = input_data.batch_size
        self.num_steps = input_data.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size

        embedding = tf.get_variable(
            "embedding", [vocab_size, hidden_size], dtype=config.float)
        inputs = tf.nn.embedding_lookup(embedding, input_data.data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, vocab_size], dtype=config.float)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=config.float)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(
            logits, [self.batch_size, self.num_steps, vocab_size])

        # Use the contrib sequence loss and average over the batches
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            input_data.targets,
            tf.ones([self.batch_size, self.num_steps], dtype=config.float),
            average_across_timesteps=False,
            average_across_batch=True)

        self.cost = tf.reduce_sum(loss)
        self.final_state = state

        if is_training:
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
        def make_cell():
            rtn = tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                rtn = tf.contrib.rnn.DropoutWrapper(
                    rtn, output_keep_prob=config.keep_prob)
            return rtn

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self.initial_state = cell.zero_state(
            config.batch_size, config.float)
        state = self.initial_state

        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs,
                                          initial_state=self.initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        self._name = name
        ops = {util.with_prefix(self._name, "cost"): self.cost}
        if self.is_training:
            ops.update(lr=self.lr, new_lr=self._new_lr,
                       lr_update=self._lr_update)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self.initial_state_name = util.with_prefix(self._name, "initial")
        self.final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self.initial_state, self.initial_state_name)
        util.export_state_tuples(self.final_state, self.final_state_name)

    def import_ops(self):
        if self.is_training:
            self.train_op = tf.get_collection_ref("train_op")[0]
            self.lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
        self.cost = tf.get_collection_ref(
            util.with_prefix(self._name, "cost"))[0]
        num_replicas = 1
        self.initial_state = util.import_state_tuples(
            self.initial_state, self.initial_state_name, num_replicas)
        self.final_state = util.import_state_tuples(
            self.final_state, self.final_state_name, num_replicas)


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


# program entry
def main(_):
    if not args.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    raw_data = reader.ptb_raw_data(args.data_path)
    train_data, valid_data, test_data, _ = raw_data

    config = util.get_config()
    eval_config = util.get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = reader.Dataset(
                config=config, data=train_data)
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                train_model = LSTMModel(is_training=True, config=config,
                                        input_data=train_input)

        with tf.name_scope("Valid"):
            valid_input = reader.Dataset(
                config=config, data=valid_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                valid_model = LSTMModel(is_training=False,
                                        config=config, input_data=valid_input)

        with tf.name_scope("Test"):
            test_input = reader.Dataset(
                config=eval_config, data=test_data)
            with tf.variable_scope("Model", reuse=True, initializer=initializer):
                test_model = LSTMModel(is_training=False, config=eval_config,
                                       input_data=test_input)

        models = {"Train": train_model,
                  "Valid": valid_model, "Test": test_model}

        # Train model and Valid model and Test model share same network state
        # but have different learning rate and other parameter states
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()

    with tf.Graph().as_default():
        tf.train.import_meta_graph(metagraph)
        for model in models.values():
            model.import_ops()
        sv = tf.train.Supervisor(logdir=args.save_path)
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

            if args.save_path:
                print("Saving model to %s." % args.save_path)
                sv.saver.save(session, args.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    # start with main function call
    tf.app.run()
