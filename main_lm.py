import sys
import os

if __name__ == "__main__":
    hidden_sizes = [400]
    lrs = [0.002, 0.004, 0.006, 0.024, 0.072, 0.216, 0.648, 1, 2]
    max_epoch = 30
    seeds = [233, 777, 42]

    data_dir = './data/language_model/sample'
    model_dir = './model'

    for hidden_size in hidden_sizes:
        for lr in lrs:
            for seed in seeds:
                cmd = ' '.join(['python', './train_torch_lm.py',
                                '-data_dir ', data_dir,
                                '-model_dir', model_dir,
                                '-hidden_size', str(hidden_size),
                                '-epoch', str(max_epoch),
                                '-seed', str(seed),
                                '-lr', str(lr)])
                os.system(cmd)
