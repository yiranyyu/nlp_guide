## What is this project

This project has two modules, but I mixed the files together.

1. Simple python implementation of word2vec algorithm with simple cython code to boost the speed.
2. Simple language model built with pytorch.

## Performance

### word2vec

Use the configuration in `main.py`, I get the following result

1. Google analogy task accuracy 0.619
2. Wordsim-353 spearman correlation 0.672

### Language model

Use the configuration in `main_lm.py`, I get the following result

1. ptb ppl = 101.17

## Requirements

* scipy >= 1.3.0
* sklearn >= 0.21.2
* pytorch >= 1.1.0
* Cython >= 0.29.4

## Announcement

This whole project is built by [yirany](https://github.com/yiranyyu). The contribution info is incorrect because I used the server of my friend thy to do part of the test work of word2vec model, which has different git info setting.

Thanks [salesforce](https://github.com/salesforce/awd-lstm-lm), [chrisjmccormick](https://github.com/chrisjmccormick/word2vec_commented), reading their work really helped me a lot.