import sys
import os

if __name__ == "__main__":
    verbose = 0
    max_epoch = 200

    lrs = [0.0001]
    # seeds = [233, 777, 42]
    seeds = [233]
    dropouts = [0.5]
    hidden_sizes = [400]

    data_dir = './data/language_model/sample'
    model_dir = './model'

    for hidden_size in hidden_sizes:
        for lr in lrs:
            for dropout in dropouts:
                for seed in seeds:
                    cmd = ' '.join(['python', './train_torch_lm.py',
                                    '-data_dir ', data_dir,
                                    '-model_dir', model_dir,
                                    '-hidden_size', str(hidden_size),
                                    '-max_epoch', str(max_epoch),
                                    '-seed', str(seed),
                                    '-verbose', str(verbose),
                                    '-dropout', str(dropout),
                                    '-lr', str(lr)])
                    os.system(cmd)
