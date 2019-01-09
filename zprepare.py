#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright 2019 Guenter Bartsch
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#

from __future__ import print_function

import os
import sys
import codecs
import time
import json
import logging
import random

import numpy as np

from optparse import OptionParser
from nltools  import misc

PROC_TITLE        = 'zprepare'

CONFIG_FN         = 'config-%s.json'
EMBEDDINGS_FN     = 'data/fasttext/word_embeddings_%s.vec'
P_TEST            = 10 # use 10% of our training data for testing

TRAIN_INP_FN      = '%s/train_inp'
TRAIN_INPL_FN     = '%s/train_inpl'
TRAIN_Q_FN        = '%s/train_q'
TRAIN_QL_FN       = '%s/train_ql'
TRAIN_A_FN        = '%s/train_a'
TEST_INP_FN       = '%s/test_inp'
TEST_INPL_FN      = '%s/test_inpl'
TEST_Q_FN         = '%s/test_q'
TEST_QL_FN        = '%s/test_ql'
TEST_A_FN         = '%s/test_a'
EMBEDDING_FN      = '%s/embedding'

DEFAULT_CASEFN    = 'data/training'

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] foo.txt [bar.txt ...]")

parser.add_option("-o", "--output-case", dest="casefn", type = "str", default=DEFAULT_CASEFN,
                  help="output case path, default: %s" % DEFAULT_CASEFN)

parser.add_option("-l", "--lang", dest="lang", type = "str", default='en',
                  help="language, default: en")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if len(args)<1:
    parser.print_usage()
    sys.exit(1)

#
# load config
#

with open(CONFIG_FN % options.lang, 'r') as configf:
    config = json.loads(configf.read())

logging.debug("config: %s" % repr(config))

#
# load datasets
#

# adapted from https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano/
def read_babi(fname, tasks):
    
    logging.info("reading babi tasks from %s" % fname)
    task = None
    for i, line in enumerate(codecs.open(fname, 'r', 'utf8')):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {"C": "", "Q": "", "A": "", "S": ""} 
            counter = 0
            id_map = {}
            
        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ')+1:]
        # if not a question
        if line.find('?') == -1:
            task["C"] += line
            id_map[id] = counter
            counter += 1
            
        else:
            idx = line.find('?')
            tmp = line[idx+1:].split('\t')
            task["Q"] = line[:idx]
            task["A"] = tmp[1].strip()
            task["S"] = []
            for num in tmp[2].split():
                task["S"].append(id_map[int(num.strip())])
            tasks.append(task.copy())

tasks = []

for infn in args:
    read_babi(infn, tasks)

random.shuffle(tasks)

logging.info ('%d tasks total.' % len(tasks))

#
# load fasttext word embeddings
#

word2vec = {}

embdfn = EMBEDDINGS_FN % options.lang
logging.info('loading word embeddings from %s ...' % embdfn)

word2vec   = {}
embed_dim  = 0

with codecs.open(embdfn, encoding='utf-8') as embdf:
    first_line = True
    for line in embdf:
        if first_line:
            first_line = False
            continue
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word2vec[word] = coefs
        if not embed_dim:
            embed_dim = coefs.shape[0]
nb_words = len(word2vec)
logging.info('found %s word vectors of dimension %d.' % (nb_words, embed_dim))

assert config['embed_size'] == embed_dim
        
#
# produce datasets
#

def process_word(word):

    global word2vec, vocab, ivocab, config

    if not word in word2vec:
        # if the word is missing, create some fake vector and store in word2vec!
        vector = np.random.uniform(0.0,1.0,(config['embed_size'],))
        word2vec[word] = vector
        logging.debug("word2vec: %s was missing, created random vector" % word)

    if not word in vocab: 
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word
    
    return vocab[word]

vocab = {}
ivocab = {}

# set word at index zero to be end of sentence token so padding with zeros is consistent
process_word(word = "<eos>")

# train_data = process_input(babi_train_raw, config['floatX'], word2vec, vocab, ivocab, config['embed_size'], split_sentences)

questions = []
inputs = []
answers = []
for task in tasks:

    inp = task["C"].lower().split(' . ') 
    inp = [w for w in inp if len(w) > 0]
    inp = [i.split() for i in inp] # FIXME: use tokenizer

    q = task["Q"].lower().split(' ')
    q = [w for w in q if len(w) > 0]

    inp_vector = [[process_word(word = w) for w in s] for s in inp]
                                
    q_vector = [process_word(word = w) for w in q]
    
    inputs.append(inp_vector)
    questions.append(np.vstack(q_vector).astype(np.float32))
    answers.append(process_word(word = task["A"])) # FIXME: here we assume the answer is one word! 

logging.info('%d questions, vocab size: %d' % (len(questions), len(vocab)))

# create embedding matrix

embedding = np.zeros((len(ivocab), config['embed_size']))
for i in range(len(ivocab)):
    word = ivocab[i]
    embedding[i] = word2vec[word]

logging.info ('created embedding matrix %s' % repr(embedding.shape))

# compute input lengths

input_lens  = np.zeros((len(inputs)), dtype=int)
sen_lens    = []
max_sen_len = 0
max_inp_len = 0
for i, t in enumerate(inputs):
    sentence_lens = np.zeros((len(t)), dtype=int)
    for j, s in enumerate(t):
        l = len(s)
        sentence_lens[j] = l
        if l > max_sen_len:
            max_sen_len = l
    l = len(t)
    input_lens[i] = l
    if l > max_inp_len:
        max_inp_len = l
    sen_lens.append(sentence_lens)

logging.info ('max_sen_len: %d, max_inp_len: %d' % (max_sen_len, max_inp_len))

if max_sen_len > config['max_sen_len']:
    logging.error('max_sen_len too low: is %d, should be at least %d.' % (config['max_sen_len'], max_sen_len))
    sys.exit(2)
max_sen_len = config['max_sen_len']

if max_inp_len > config['max_inp_len']:
    logging.error('max_inp_len too low: is %d, should be at least %d.' % (config['max_inp_len'], max_inp_len))
    sys.exit(2)
max_inp_len = config['max_inp_len']

# compute question lens:

q_lens = np.zeros((len(questions)), dtype=int)
for i, t in enumerate(questions):
    q_lens[i] = t.shape[0]
max_q_len = np.max(q_lens)
if max_q_len > config['max_q_len']:
    logging.error('max_q_len too low: is %d, should be at least %d.' % (config['max_q_len'], max_q_len))
    sys.exit(2)
max_q_len = config['max_q_len']

# pad out to max, create numpy arrays

inp_padded = np.zeros((len(inputs), max_inp_len, max_sen_len))
for i, inp in enumerate(inputs):
    padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in enumerate(inp)]
    padded_sentences = np.vstack(padded_sentences)
    padded_sentences = np.pad(padded_sentences, ((0, max_inp_len - input_lens[i]),(0,0)), 'constant', constant_values=0)
    inp_padded[i] = padded_sentences

logging.debug('inp_padded[0] = %s %s' % (repr(inp_padded[0].shape), inp_padded[0]))

# questions = pad_inputs(questions, q_lens, max_q_len)
q_padded = [np.pad(np.squeeze(q, axis=1), (0, max_q_len - q_lens[i]), 'constant', constant_values=0) for i, q in enumerate(questions)]
q_padded = np.vstack(q_padded)

logging.debug('q_padded[0] = %s %s' % (repr(q_padded[0].shape), q_padded[0]))

a_padded = np.stack(answers)

#
# split training + test dataset, save to disk
#

num_total = len(a_padded)
num_test  = num_total * P_TEST / 100
num_train = num_total - num_test

logging.info('stats: %d test samples, %d training samples, %d samples total.' % (num_test, num_train, num_total))

def save_ds(arr, fn):

    global options

    dsfn = fn % options.casefn
    np.save(dsfn, arr)
    logging.info('%s written, %d samples.' % (dsfn, len(arr)))

misc.mkdirs(options.casefn)

save_ds (inp_padded[:num_train], TRAIN_INP_FN )
save_ds (input_lens[:num_train], TRAIN_INPL_FN)
save_ds (q_padded[:num_train],   TRAIN_Q_FN   )
save_ds (q_lens[:num_train],     TRAIN_QL_FN  )
save_ds (a_padded[:num_train],   TRAIN_A_FN   )

save_ds (inp_padded[num_train:], TEST_INP_FN )
save_ds (input_lens[num_train:], TEST_INPL_FN )
save_ds (q_padded[num_train:],   TEST_Q_FN   )
save_ds (q_lens[num_train:],     TEST_QL_FN   )
save_ds (a_padded[num_train:],   TEST_A_FN   )

save_ds (embedding,              EMBEDDING_FN)

cmd = 'cp %s %s/config.json' % (CONFIG_FN % options.lang, options.casefn)
logging.info(cmd)
os.system(cmd)


