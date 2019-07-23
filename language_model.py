import tensorflow as tf
import util


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

        output, state = self.rnn_graph(inputs, config, is_training)

        softmax_w = tf.get_variable(
            "softmax_w", [hidden_size, vocab_size], dtype=config.float)
        softmax_b = tf.get_variable(
            "softmax_b", [vocab_size], dtype=config.float)
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(
            logits, [self.batch_size, self.num_steps, vocab_size])
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

    def rnn_graph(self, inputs, config, is_training):
        def make_cell():
            rtn = tf.contrib.rnn.LSTMBlockCell(
                config.hidden_size, forget_bias=0.0)
            if is_training and config.keep_prob < 1:
                rtn = tf.contrib.rnn.DropoutWrapper(
                    rtn, output_keep_prob=config.keep_prob)
            return rtn

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        state = self.initial_state = cell.zero_state(config.batch_size, config.float)

        inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
        outputs, state = tf.nn.static_rnn(cell, inputs,
                                          initial_state=self.initial_state)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        self._name = name
        ops = {util.join(self._name, "cost"): self.cost}
        if self.is_training:
            ops.update(lr=self.lr, new_lr=self._new_lr,
                       lr_update=self._lr_update)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self.initial_state_name = util.join(self._name, "initial")
        self.final_state_name = util.join(self._name, "final")
        util.export_state_tuples(self.initial_state, self.initial_state_name)
        util.export_state_tuples(self.final_state, self.final_state_name)

    def import_ops(self):
        if self.is_training:
            self.train_op = tf.get_collection_ref("train_op")[0]
            self.lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
        self.cost = tf.get_collection_ref(
            util.join(self._name, "cost"))[0]
        num_replicas = 1
        self.initial_state = util.import_state_tuples(
            self.initial_state, self.initial_state_name, num_replicas)
        self.final_state = util.import_state_tuples(
            self.final_state, self.final_state_name, num_replicas)
