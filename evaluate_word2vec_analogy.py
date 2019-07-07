import argparse
from os import path
from model import Word2vecModel
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_path = parser.add_argument('-model', help='Model path', dest='model_path', required=True)
    start = time()
    log_dir = './log'
    data_dir = './data/word2vec'
    result_dir = './log'

    data_name = '类比_questions-words.txt'

    log_path = path.join(log_dir, model_name) + '.analogy_log_new'
    data_path = path.join(data_dir, data_name)
    result_path = path.join(result_dir, model_name) + '.analogy_result_new'

    model = Word2vecModel(model_path)

    hit = 0
    miss = 0
    with open(data_path, 'r') as file, open(log_path, 'at', buffering=1) as log_file, open(result_path, 'wt') as result_file:
        log_file.write('Built model in %fm\n' % ((time() - start) / 60))
        log_file.flush()
        for i, line in enumerate(file):
            if line.startswith(':'):
                result_file.write(line)
                continue

            # since the training data is all lowercase, here lower the line first
            line = line.strip().lower()
            a, b, c, d = line.split(' ')
            try:
                e = model.analogy(a, b, c)
                if e == d:
                    hit += 1
                else:
                    miss += 1
                result_file.write(' '.join([a, b, c, d, e]) + '\n')
            except KeyError as e:
                log_file.write('Unkown word <' + str(e) + '>\n')
            if i and i % 100 == 0:
                log_file.write('i = %d, current acc = %.3f\n' %
                               (i, hit / (hit + miss)))
        log_file.write('Finished in %.3fm\n' % ((time() - start) / 60))
        log_file.write('Acc: %f, hit = %d, miss = %d' %
                       (hit / (hit + miss), hit, miss))
