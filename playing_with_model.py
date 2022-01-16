# coding: utf-8
from __future__ import division

import models
import data
import main

import sys
import codecs

import tensorflow as tf
import numpy as np

MAX_SUBSEQUENCE_LEN = 200
model_file = r'Model_ru_punctuator_h256_lr0.02.pcl'

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T

def convert_punctuation_to_readable(punct_token):
    if punct_token == data.SPACE:
        return " "
    else:
        return punct_token[0]

def restore(text, word_vocabulary, reverse_punctuation_vocabulary, model):
    i = 0
    while True:
        string_to_punct = ''
        subsequence = text[i:i+MAX_SUBSEQUENCE_LEN]

        if len(subsequence) == 0:
            break

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[data.UNK]) for w in subsequence]

        y = predict(to_array(converted_subsequence), model)

        string_to_punct += subsequence[0]

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(tf.reshape(y_t, [-1]))
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in data.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        if subsequence[-1] == data.END:
            step = len(subsequence) - 1
        elif last_eos_idx != 0:
            step = last_eos_idx
        else:
            step = len(subsequence) - 1

        for j in range(step):
            string_to_punct += (punctuations[j] + " " if punctuations[j] != data.SPACE else " ")
            if j < step - 1:
                string_to_punct += subsequence[1+j]

        if subsequence[-1] == data.END:
            break

        i += step
    return(string_to_punct)

def predict(x, model):
    return tf.nn.softmax(net(x))

if __name__ == "__main__":

    vocab_len = len(data.read_vocabulary(data.WORD_VOCAB_FILE))
    x_len = vocab_len if vocab_len < data.MAX_WORD_VOCABULARY_SIZE else data.MAX_WORD_VOCABULARY_SIZE + data.MIN_WORD_COUNT_IN_VOCAB
    x = np.ones((x_len, main.MINIBATCH_SIZE)).astype(int)

    print("Loading model parameters...")
    net, _ = models.load(model_file, x)

    print("Building model...")

    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    reverse_word_vocabulary = {v:k for k,v in word_vocabulary.items()}
    reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}
    for key, value in reverse_punctuation_vocabulary.items():
        if value == '.PERIOD':
            reverse_punctuation_vocabulary[key] = '.'
        if value == ',COMMA':
            reverse_punctuation_vocabulary[key] = ','
        if value == '?QUESTIONMARK':
            reverse_punctuation_vocabulary[key] = '?'
    print('Нажмите Ctrl + C, чтобы выйти.')
    while True:
        input_text = input('Введите текст без знаков пунктуации: ')

        if len(input_text) == 0:
            sys.exit("Input file empty.")

        text = [w for w in input_text.split() if w not in punctuation_vocabulary and w not in data.PUNCTUATION_MAPPING and not w.startswith(data.PAUSE_PREFIX)] + [data.END]
        pauses = [float(s.replace(data.PAUSE_PREFIX,"").replace(">","")) for s in input_text.split() if s.startswith(data.PAUSE_PREFIX)]

        text_with_punct = restore(text, word_vocabulary, reverse_punctuation_vocabulary, net)
        import nltk.data
        punkt_tokenizer = nltk.data.load('tokenizers/punkt/russian.pickle')
        sentences = punkt_tokenizer.tokenize(text_with_punct)
        sentences = [sent.capitalize() for sent in sentences]
        uppercase_text = ' '.join(sentences)
        print("Текст успешно откорректирован:", uppercase_text)