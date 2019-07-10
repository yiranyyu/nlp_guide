import sys
import os

if __name__ == "__main__":
    epoch = '5'
    nrof_processes = '11'
    model_dir = './model'
    data_dir = './data/word2vec'
    cbow = 1
    lr = '0.025'
    neg = 5
    dim = '300'
    sample = '1e-5'

    data_file = '250m_words.txt'
    model_file = '%s_%s_%s_%s_%s_%s_lr=%s.%s' % (data_file, epoch, dim, ('cbow' if cbow else 'skip-gram'), (('neg%d' % neg) if neg else 'softmax'), sample, lr, 'model')
    data_path = os.path.join(data_dir, data_file)
    model_path = os.path.join(model_dir, model_file)

    print('\nData path:', os.path.abspath(data_path), flush=True)
    print('Save path:', os.path.abspath(model_path), flush=True)

    cmd0 = ' '.join(['python', './word2vec.py',
                    '-train ', data_path,
                    '-model', model_path,
                    '-processes', nrof_processes,
                    '-epoch', epoch,
                    '-cbow', str(cbow),
					'-dim', dim,
                    '-negative', str(neg),
                    '-sample', sample,
                    '-lr', lr])

    cmd1 = ' '.join(['python', './evaluate_word2vec_analogy.py',
                    '-model', model_path])

    cmd = cmd0 + ' && ' + cmd1
    os.system(cmd)
