from model import Word2vecModel

if __name__ == "__main__":
    model_path = './model/100m_words.txt.model'
    data_path = './data/word2vec/类比_questions-words.txt'
    model = Word2vecModel(model_path)
    with open(data_path, 'r') as file:
        for line in file:
            if line.startswith(':'):
                continue
            a, b, c, d = line.lower().strip().split(' ')
            print(a, b, c, d, model.analogy(a, b, c))