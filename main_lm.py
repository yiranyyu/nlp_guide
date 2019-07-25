import sys
import os

if __name__ == "__main__":
    hidden_sizes = [1500]
    lrs = [0.0001]
    max_epoch = 200
    seeds = [233, 777, 42]
    verbose = 0

    data_dir = './data/language_model/'
    model_dir = './model'

    for hidden_size in hidden_sizes:
        for lr in lrs:
            for seed in seeds:
                cmd = ' '.join(['python', './train_torch_lm.py',
                                '-data_dir ', data_dir,
                                '-model_dir', model_dir,
                                '-hidden_size', str(hidden_size),
                                '-max_epoch', str(max_epoch),
                                '-seed', str(seed),
                                '-verbose', str(verbose),
                                '-lr', str(lr)])
                os.system(cmd)
