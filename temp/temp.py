import os
path = '../log/words_frequency.txt'
with open(path, 'r') as data:
    occurrence = {}
    total = 0
    for line in data:
        word, cnt = line.strip().split(',')
        cnt = int(cnt)
        occurrence[cnt] = occurrence.get(cnt, 0) + 1
        total += 1
    occurrence = [(freq, occurrence[freq]) for freq in occurrence]
    occurrence = sorted(occurrence, key=lambda x:x[0])
    cur_sum = 0
    text = ''
    for freq, occ in occurrence:
        cur_sum += occ
        text += '%d, %.3lf, %.3lf\n' % (freq, cur_sum / total, occ / total)
    open('../log/words_ratio.txt', 'w', encoding='utf8').write(text)