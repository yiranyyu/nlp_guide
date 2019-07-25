import argparse
from os import path
from word2vec_model import Word2vecModel
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', help='Model path', dest='model_path', required=True)
    model_path = parser.parse_args().model_path
    start = time()
    log_dir = './log'
    data_dir = './data/word2vec'
    result_dir = './log'

    data_name = 'analogy_questions_words.txt'
    model_name = model_path.split('/')[-1]

    log_path = path.join(log_dir, model_name) + '.analogy_log_new'
    data_path = path.join(data_dir, data_name)
    result_path = path.join(result_dir, model_name) + '.analogy_result_new'

    model = Word2vecModel(model_path)

    hit = 0
    miss = 0
    print('Evaluating model from ' + model_path)
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
                log_file.write('Unknown word <' + str(e) + '>\n')
            if i and i % 100 == 0:
                log_file.write('i = %d, current acc = %.3f\n' %
                               (i, hit / (hit + miss)))
        result = 'Evaluate finished in %.3fm\n' % ((time() - start) / 60)
        result += 'Acc: %f, hit = %d, miss = %d' % (hit / (hit + miss), hit, miss)
        log_file.write(result)
    print(result)
