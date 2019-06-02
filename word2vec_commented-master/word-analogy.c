//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

const long long max_size = 2000; // max length of strings
const long long N = 40;          // number of closest words that will be shown
const long long max_w = 50;      // max length of vocabulary entries

int main(int argc, char **argv) {
    FILE *f;
    char input_line[max_size];
    char best_word_of_rank[N][max_size];
    char file_name[max_size], input_words[100][max_size];
    float dist, len, best_dist_of_rank[N], vec[max_size];
    long long a, b, c, d, index_of_input_words[100];
    long long size;  // embedding size
    long long words; // total words number
    long long cn; // number of user input words in command line
    char ch;
    float *embedding; // word embedding
    char *vocab;
    if (argc < 2) {
        printf("Usage: ./word-analogy <FILE>\nwhere FILE contains word "
               "projections in the BINARY FORMAT\n");
        return 0;
    }
    strcpy(file_name, argv[1]);
    f = fopen(file_name, "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    vocab = (char *)malloc((long long)words * max_w * sizeof(char));
    embedding =
        (float *)malloc((long long)words * (long long)size * sizeof(float));
    if (embedding == NULL) {
        printf("Cannot allocate memory: %lld MB    %lld  %lld\n",
               (long long)words * size * sizeof(float) / 1048576, words, size);
        return -1;
    }

    // process input data
    for (b = 0; b < words; b++) {
        // read word
        a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' '))
                break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n'))
                a++;
        }
        vocab[b * max_w + a] = 0;

        // read word vector
        for (a = 0; a < size; a++)
            fread(&embedding[a + b * size], sizeof(float), 1, f);

        // normalize current word vector
        len = 0;
        for (a = 0; a < size; a++)
            len += embedding[a + b * size] * embedding[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++)
            embedding[a + b * size] /= len;
    }
    fclose(f);
    while (1) {
        for (a = 0; a < N; a++)
            best_dist_of_rank[a] = 0;
        for (a = 0; a < N; a++)
            best_word_of_rank[a][0] = 0;
        printf("Enter three words (EXIT to break): ");
        a = 0;
        while (1) {
            input_line[a] = fgetc(stdin);
            if ((input_line[a] == '\n') || (a >= max_size - 1)) {
                input_line[a] = 0;
                break;
            }
            a++;
        }
        if (!strcmp(input_line, "EXIT"))
            break;
        cn = 0;
        b = 0;
        c = 0;
        while (1) {
            input_words[cn][b] = input_line[c];
            b++;
            c++;
            input_words[cn][b] = 0;
            if (input_line[c] == 0)
                break;
            if (input_line[c] == ' ') {
                cn++;
                b = 0;
                c++;
            }
        }
        cn++;
        if (cn < 3) {
            printf("Only %lld words were entered.. three words are needed at "
                   "the input to perform the calculation\n",
                   cn);
            continue;
        }
        for (a = 0; a < cn; a++) {
            for (b = 0; b < words; b++) {
                if (!strcmp(&vocab[b * max_w], input_words[a]))
                    break;
            }
            if (b == words)
                b = 0;
            index_of_input_words[a] = b;
            printf("\nWord: %s  Position in vocabulary: %lld\n", input_words[a],
                   index_of_input_words[a]);
            if (b == 0) {
                printf("Out of dictionary word!\n");
                break;
            }
        }
        if (b == 0)
            continue;
        printf(
            "\n                                              Word              "
            "Distance\n--------------------------------------------------------"
            "----------------\n");

        // v, v', u, u'
        // compute u' = v' - v + u;
        for (a = 0; a < size; a++)
            vec[a] = embedding[a + index_of_input_words[1] * size] -
                     embedding[a + index_of_input_words[0] * size] +
                     embedding[a + index_of_input_words[2] * size];

        // normalize diff_vector
        len = 0;
        for (a = 0; a < size; a++)
            len += vec[a] * vec[a];
        len = sqrt(len);
        for (a = 0; a < size; a++)
            vec[a] /= len;

        // init
        for (a = 0; a < N; a++)
            best_dist_of_rank[a] = 0;
        for (a = 0; a < N; a++)
            best_word_of_rank[a][0] = 0;

        // find best fit word
        for (c = 0; c < words; c++) {
            // omit input 3 words
            if (c == index_of_input_words[0])
                continue;
            if (c == index_of_input_words[1])
                continue;
            if (c == index_of_input_words[2])
                continue;

            // omit other user inputed words (cn can be more than 3)
            a = 0;
            for (b = 0; b < cn; b++)
                if (index_of_input_words[b] == c)
                    a = 1;
            if (a == 1)
                continue;

            // compute dot-product as dist between diff_vector and current word
            // vector
            dist = 0;
            for (a = 0; a < size; a++)
                dist += vec[a] * embedding[a + c * size];

            // update result with current dist & word
            for (a = 0; a < N; a++) {
                // find a better fit, i.e. larger dist
                if (dist > best_dist_of_rank[a]) {
                    for (d = N - 1; d > a; d--) {
                        best_dist_of_rank[d] = best_dist_of_rank[d - 1];
                        strcpy(best_word_of_rank[d], best_word_of_rank[d - 1]);
                    }
                    best_dist_of_rank[a] = dist;
                    strcpy(best_word_of_rank[a], &vocab[c * max_w]);
                    break;
                }
            }
        }
        // print result
        for (a = 0; a < N; a++) {
            printf("%50s\t\t%f\n", best_word_of_rank[a], best_dist_of_rank[a]);
        }
    }
    return 0;
}
