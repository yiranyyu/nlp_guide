from model import Word2vecModel

if __name__ == "__main__":
    model_path = './model/100m_words.txt.model'
    data_path = './data/word2vec/类比_questions-words.txt'
    model = Word2vecModel(model_path)

    hit = 0
    miss = 0
    with open(data_path, 'r') as file:
        lines = file.readlines()
        lines = [line.strip().lower() for line in lines]
        for line in file:
            if line.startswith(':'):
                continue
            a, b, c, d = line.split(' ')
            e = model.analogy(a, b, c)
            if e == d:
                hit += 1
            else:
                miss += 1
        print('score: %f' % (hit / (hit + miss)))