

# def make_batch(raw_data: list, batch_size: int, seq_len: int):
#     raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)

#     data_len = tf.size(raw_data)
#     n_batches = data_len // batch_size
#     data = tf.reshape(raw_data[0: batch_size * n_batches],
#                       [batch_size, n_batches])

#     # number of steps in one epoch
#     epoch_size = (n_batches - 1) // seq_len

#     # create queue like [0, 1, ...., epoch_size - 1,
#     #                    0, 1, ...., epoch_size - 1,
#     #                    ..... ]
#     i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

#     # using element in x to  element in y
#     # i.e. using the current word (index) to predict the next word (index)
#     x = tf.strided_slice(data, [0, i * seq_len],
#                          [batch_size, (i + 1) * seq_len])
#     x.set_shape([batch_size, seq_len])
#     y = tf.strided_slice(data, [0, i * seq_len + 1],
#                          [batch_size, (i + 1) * seq_len + 1])
#     y.set_shape([batch_size, seq_len])
#     return x, y


class Dataset(object):
    def __init__(self, config, data):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        self.data, self.targets = make_batch(
            data, batch_size, num_steps)
