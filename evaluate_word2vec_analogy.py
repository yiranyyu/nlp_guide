from os import path
from model import Word2vecModel


if __name__ == "__main__":
    model_dir = './model'
    model_name = '346m_words.txt_5_skip-gram_neg5.model'
    model_path = path.join(model_dir, model_name)
    data_dir = './data/word2vec'
    data_name = '类比_questions-words.txt'
    data_path = path.join(data_dir, data_name)
    result_dir = './data/test_result'
    result_path = path.join(result_dir, model_name) + '.result'
    model = Word2vecModel(model_path)

    hit = 0
    miss = 0
    error_msg = ''
    with open(data_path, 'r') as file:
        for i, line in enumerate(file):
            line = line.strip()
            if line.startswith(':'):
                continue
            a, b, c, d = line.split(' ')
            try:
                e = model.analogy(a, b, c)
                if e == d or e.lower() == d.lower():
                    hit += 1
                else:
                    miss += 1
            except KeyError as e:
                error_msg += 'Unkown word <' + str(e) + '>\n'
                print('Unkown word <' + str(e) + '>\n', flush=True)
            if i and i % 100 == 0:
                print('i = %d, current acc = %.3f' % (i, hit / (hit + miss)), flush=True)
        open(result_path, 'wt').write(error_msg + 'score: %f, hit = %d, miss = %d' % (hit / (hit + miss), hit, miss))