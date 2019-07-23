import time
import numpy as np
import tensorflow as tf
import reader
import util
from language_model import LSTMModel


def run_epoch(session, model, operation=None, verbose=False):
    start_time = time.time()
    total_cost = 0.0
    iters = 0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if operation is not None:
        fetches["eval_op"] = operation

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
            print("%.3f ppl: %.3f" %
                  (step * 1.0 / model.input.epoch_size, np.exp(total_cost / iters)))

    return np.exp(total_cost / iters)


def init_models(args, config):
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(args.data_path)
    eval_config = util.Config(args, eval=True)
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
        return metagraph, models


def main(args, verbose=False):
    config = util.Config(args)
    metagraph, models = init_models(args, config)
    train_model, valid_model, test_model = models['Train'], models['Valid'], models['Test']

    # reload graph after export
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

                print("[Epoch %d] lr: %.3f" %
                      (i + 1, session.run(train_model.lr)))
                train_perplexity = run_epoch(session, train_model, operation=train_model.train_op,
                                             verbose=verbose)
                print("[Epoch %d] Train ppl: %.3f" %
                      (i + 1, train_perplexity))
                valid_perplexity = run_epoch(session, valid_model)
                print("[Epoch %d] Valid ppl: %.3f" %
                      (i + 1, valid_perplexity))

            test_perplexity = run_epoch(session, test_model)
            print("Test ppl: %.3f" % test_perplexity)

            if args.save_path:
                print("Saving model to %s." % args.save_path)
                sv.saver.save(session, args.save_path,
                              global_step=sv.global_step)


if __name__ == "__main__":
    main(util.get_args(), verbose=True)
