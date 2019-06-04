with open('../data/word2vec/训练语料.txt', 'r') as file:
    tofind = 'Beijing'
    for line in file:
        if tofind in line:
            index = line.find(tofind)
            print(line)
        else:
            print('#', end='')