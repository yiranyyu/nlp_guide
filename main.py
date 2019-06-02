import sys
import os

if __name__ == "__main__":
    epoch = '5'
    nrof_processes = '8'
    model_dir = './model'
    data_dir = './data/word2vec'
    cbow = 0
    neg = 5

    data_file = '1m_words.txt'
    model_file = '%s_%s_%s_%s.%s' % (data_file, epoch,('cbow' if cbow else 'skip-gram'), (('neg%d' % neg) if neg else 'softmax'),'model')
    data_path = os.path.join(data_dir, data_file)
    model_path = os.path.join(model_dir, model_file)

    print('data path:', os.path.abspath(data_path))
    print('save path:', os.path.abspath(model_path))

    cmd = ' '.join(['python', './word2vec.py',
                    '-train ', data_path,
                    '-model', model_path,
                    '-processes', nrof_processes,
                    '-epoch', epoch,
                    '-cbow', str(cbow),
                    '-negative', str(neg)])
    os.system(cmd)
