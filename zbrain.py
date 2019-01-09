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
import logging
import json

import tensorflow as tf
from attention_gru_cell import AttentionGRUCell
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops

import numpy as np

from optparse import OptionParser
from nltools  import misc

import babi_input

PROC_TITLE        = 'zbrain'

TRAIN_INP_FN      = '%s/train_inp.npy'
TRAIN_INPL_FN     = '%s/train_inpl.npy'
TRAIN_Q_FN        = '%s/train_q.npy'
TRAIN_QL_FN       = '%s/train_ql.npy'
TRAIN_A_FN        = '%s/train_a.npy'
TEST_INP_FN       = '%s/test_inp.npy'
TEST_INPL_FN      = '%s/test_inpl.npy'
TEST_Q_FN         = '%s/test_q.npy'
TEST_QL_FN        = '%s/test_ql.npy'
TEST_A_FN         = '%s/test_a.npy'
EMBEDDING_FN      = '%s/embedding.npy'
CONFIG_FN         = '%s/config.json'

# config = {
#     'batch_size'            : 100,
#     'max_sentences'         :  88,
#     'max_sen_len'           :   6,
#     'word2vec_init'         : False,
#     'embedding_init'        : np.sqrt(3),
#     'embed_size'            :  80,
#     'hidden_size'           :  80,
#     'num_hops'              :   3,
#     'dropout'               : 0.9,
#     'l2'                    : 0.001,
#     'lr'                    : 0.001,
#     'anneal_threshold'      : 1000,
#     'anneal_by'             : 1.5,
#     'early_stopping'        :  20,
#     'cap_grads'             : False,
#     'noisy_grads'           : False,
#     'floatX'                : np.float32,
#     'max_allowed_input_len' : 130,
# 
#     'num_train'             : 9000, # FIXME: -> validation set
#     'max_epochs'            :    1, # FIXME
#     }

class DMN_PLUS(object):

    def get_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):

            features = [fact_vec*q_vec,
                        fact_vec*prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)

            attention = tf.contrib.layers.fully_connected(feature_vec,
                            self.config['embed_size'],
                            activation_fn=tf.nn.tanh,
                            reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                            1,
                            activation_fn=None,
                            reuse=reuse, scope="fc2")

        return attention

    def __init__(self, config, word_embedding):

        self.config = config
        self.variables_to_save = {}
        # self.load_data(debug=False)

        ivocab = {0: '<eos>', 1: 'mary', 2: 'moved', 3: 'to', 4: 'the', 5: 'bathroom', 6: 'sandra', 
                  7: 'journeyed', 8: 'bedroom', 9: 'got', 10: 'football', 11: 'there', 12: 'john', 
                  13: 'went', 14: 'kitchen', 15: 'back', 16: 'garden', 17: 'where', 18: 'is', 
                  19: 'office', 20: 'hallway', 21: 'daniel', 22: 'dropped', 23: 'milk', 24: 'took', 
                  25: 'picked', 26: 'up', 27: 'apple', 28: 'travelled', 29: 'left', 30: 'grabbed', 31: 'discarded', 32: 'put', 33: 'down'}


        self.vocab_size = len(ivocab)

        #
        # placeholders
        #

        self.question_placeholder     = tf.placeholder(tf.int32, shape=(self.config['batch_size'], self.config['max_q_len']), name='qtn_ph')
        logging.debug('self.question_placeholder=%s' % self.question_placeholder)

        self.input_placeholder        = tf.placeholder(tf.int32, shape=(self.config['batch_size'], self.config['max_inp_len'], self.config['max_sen_len']), name='input_ph')
        logging.debug('self.input_placeholder=%s' % self.input_placeholder)

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config['batch_size'],), name='qtn_len_ph')
        logging.debug('self.question_len_placeholder=%s' % self.question_len_placeholder)

        self.input_len_placeholder    = tf.placeholder(tf.int32, shape=(self.config['batch_size'],), name='inp_len_ph')
        logging.debug('self.input_len_placeholder=%s' % self.input_len_placeholder)

        self.answer_placeholder       = tf.placeholder(tf.int64, shape=(self.config['batch_size'],), name='answ_ph')
        logging.debug('self.answer_placeholder=%s' % self.answer_placeholder)

        self.dropout_placeholder      = tf.placeholder(tf.float32, name='dropout_ph')
        logging.debug('self.dropout_placeholder=%s' % self.dropout_placeholder)

        self.word_embedding = word_embedding

        # set up embedding
        self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="embeddings")

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):

            logging.debug('==> get question representation (q_vec)')

            questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)
            logging.debug('questions=%s' % questions)
        
            gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
            logging.debug('gru_cell=%s' % gru_cell)
            _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                         questions,
                                         dtype=np.float32,
                                         sequence_length=self.question_len_placeholder)
            logging.debug('q_vec=%s' % q_vec)

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            logging.debug('==> get input representation (fact_vecs)')

            inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
            logging.debug('inputs=%s' % inputs)
       
            # We could have used RNN for parsing sentence but that tends to overfit.
            # The simpler choice would be to take sum of embedding but we loose loose positional information.
            # Position encoding is described in section 4.1 in "End to End Memory Networks" in more detail (http://arxiv.org/pdf/1503.08895v5.pdf)
            # from https://github.com/domluna/memn2n
            encoding = np.ones((self.config['embed_size'], self.config['max_sen_len']), dtype=np.float32)
            ls = self.config['max_sen_len']+1
            le = self.config['embed_size']+1
            for i in range(1, le):
                for j in range(1, ls):
                    encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
            encoding = 1 + 4 * encoding / self.config['embed_size'] / self.config['max_sen_len']
            self.encoding = np.transpose(encoding)
            logging.debug('self.encoding=%s' % self.encoding)
 
            # use encoding to get sentence representation
            inputs = tf.reduce_sum(inputs * self.encoding, 2)
            logging.debug('inputs=%s' % inputs)
        
            forward_gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
            logging.debug('forward_gru_cell=%s' % forward_gru_cell)
            backward_gru_cell = tf.contrib.rnn.GRUCell(self.config['hidden_size'])
            logging.debug('backward_gru_cell=%s' % backward_gru_cell)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn( forward_gru_cell,
                                                          backward_gru_cell,
                                                          inputs,
                                                          dtype=np.float32,
                                                          sequence_length=self.input_len_placeholder )
            logging.debug('outputs=%s' % repr(outputs))
    
            # sum forward and backward output vectors
            fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
            logging.debug('fact_vecs=%s' % fact_vecs)
    
            # apply dropout
            fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)
            logging.debug('fact_vecs=%s' % fact_vecs)

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            logging.debug('==> build episodic memory')

            # generate n_hops episodes
            prev_memory = q_vec

            for hop_index in range(self.config['num_hops']):
                # get a new episode
                logging.debug('==> generating episode %d' % hop_index)

                # Generate episode by applying attention to current fact vectors through a modified GRU

                attentions = [ tf.squeeze( self.get_attention(q_vec, prev_memory, fv, bool(hop_index) or bool(i)), axis=1)
                               for i, fv in enumerate(tf.unstack(fact_vecs, axis=1)) ]

                with tf.variable_scope("ep_%d" % hop_index):
                    attentions = tf.transpose(tf.stack(attentions))
                    self.attentions.append(attentions)
                    attentions = tf.nn.softmax(attentions)
                    attentions = tf.expand_dims(attentions, axis=-1)

                    # concatenate fact vectors and attentions for input into attGRU
                    gru_inputs = tf.concat([fact_vecs, attentions], 2)

                reuse = True if hop_index > 0 else False
                with tf.variable_scope('attention_gru', reuse=reuse):
                    _, episode = tf.nn.dynamic_rnn(AttentionGRUCell(self.config['hidden_size']),
                                                   gru_inputs,
                                                   dtype=np.float32,
                                                   sequence_length=self.input_len_placeholder)

                # untied weights for memory update
                with tf.variable_scope("hop_%d" % hop_index):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.config['hidden_size'],
                                                  activation=tf.nn.relu)

            rnn_output = prev_memory
            logging.debug('rnn_output=%s' % rnn_output)

        # pass memory module output through linear answer module
        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

            self.output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                          self.vocab_size,
                                          activation=None)
            logging.debug('self.output=%s' % self.output)

            preds = tf.nn.softmax(self.output)
            self.pred = tf.argmax(preds, 1)

        with tf.variable_scope("training", initializer=tf.contrib.layers.xavier_initializer()):
            # self.calculate_loss = self.add_loss_op(self.output)
            """Calculate loss"""
            self.calculate_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.answer_placeholder))

            # add l2 regularization for all variables except biases
            for v in tf.trainable_variables():
                if not 'bias' in v.name.lower():
                    self.calculate_loss += self.config['l2']*tf.nn.l2_loss(v)

            tf.summary.scalar('loss', self.calculate_loss)

            logging.debug('self.calculate_loss=%s' % self.calculate_loss)

            # add training op
            # Calculate and apply gradients
            opt = tf.train.AdamOptimizer(learning_rate=self.config['lr'])
            gvs = opt.compute_gradients(self.calculate_loss)

            # optionally cap and noise gradients to regularize
            if self.config['cap_grads']:
                gvs = [(tf.clip_by_norm(grad, self.config['max_grad_val']), var) for grad, var in gvs]
            if self.config['noisy_grads']:
                gvs = [(_add_gradient_noise(grad), var) for grad, var in gvs]

            self.train_step = opt.apply_gradients(gvs)
            logging.debug('self.train_step=%s' % self.train_step)

            self.merged = tf.summary.merge_all()

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False):
        config = self.config
        dp = config['dropout']
        if train_op is None:
            # train_op = tf.no_op()
            dp = 1
        total_steps = len(data[0]) // config['batch_size']
        total_loss = []
        accuracy = 0

        # shuffle data
        p = np.random.permutation(len(data[0]))
        ip, il, qp, ql, a = data
        qp, ip, ql, il, a = qp[p], ip[p], ql[p], il[p], a[p]

        for step in range(total_steps):
            index = range(step*config['batch_size'], (step+1)*config['batch_size'])
            feed = {self.question_placeholder: qp[index],
                    self.input_placeholder: ip[index],
                    self.question_len_placeholder: ql[index],
                    self.input_len_placeholder: il[index],
                    self.answer_placeholder: a[index],
                    self.dropout_placeholder: dp}

            # print ('self.question_placeholder    : %s' % repr(self.question_placeholder))
            # print ('self.input_placeholder       : %s' % repr(self.input_placeholder))
            # print ('self.question_len_placeholder: %s' % repr(self.question_len_placeholder))
            # print ('self.input_len_placeholder   : %s' % repr(self.input_len_placeholder))
            # print ('self.answer_placeholder      : %s' % repr(self.answer_placeholder))
            # print ('self.dropout_placeholder     : %s' % repr(self.dropout_placeholder))

            if train_op is None:
                loss, pred, summary,  = session.run( [self.calculate_loss, self.pred, self.merged], feed_dict=feed)
            else:
                loss, pred, summary, _ = session.run( [self.calculate_loss, self.pred, self.merged, train_op], feed_dict=feed)

            if train_writer is not None:
                train_writer.add_summary(summary, num_epoch*total_steps + step)

            answers = a[step*config['batch_size']:(step+1)*config['batch_size']]
            accuracy += np.sum(pred == answers)/float(len(answers))


            total_loss.append(loss)
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()


        if verbose:
            sys.stdout.write('\r')

        return np.mean(total_loss), accuracy/float(total_steps)


#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] case_fn")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if len(args) != 1:
    parser.print_usage()
    sys.exit(0)

casefn = args[0]

#
# load dataset
#

# train_mode = True

with open(CONFIG_FN % casefn) as configf:
    config = json.loads(configf.read())

def load_ds(fn):

    global casefn

    dsfn = fn % casefn
    arr = np.load(dsfn)
    logging.info('%s read, %d samples.' % (dsfn, len(arr)))

    return arr

train_inp  = load_ds (TRAIN_INP_FN )
train_inpl = load_ds (TRAIN_INPL_FN)
train_q    = load_ds (TRAIN_Q_FN   )
train_ql   = load_ds (TRAIN_QL_FN  )
train_a    = load_ds (TRAIN_A_FN   )

train = ( train_inp, train_inpl, train_q, train_ql, train_a )

test_inp   = load_ds (TEST_INP_FN  )
test_inpl  = load_ds (TEST_INPL_FN )
test_q     = load_ds (TEST_Q_FN    )
test_ql    = load_ds (TEST_QL_FN   )
test_a     = load_ds (TEST_A_FN    )

test = ( test_inp, test_inpl, test_q, test_ql, test_a )

word_embedding = load_ds (EMBEDDING_FN )

# if train_mode:
#     train, valid, word_embedding, config['max_q_len'], config['max_sentences'], config['max_sen_len'], vocab_size = babi_input.load_babi(babi_id, '', train_mode, config, split_sentences=True)
# 
# else:
#     test, word_embedding, config['max_q_len'], config['max_sentences'], config['max_sen_len'], vocab_size = babi_input.load_babi(babi_id, '', train_mode, config, split_sentences=True)

vocab_size = len(word_embedding)

#
# main
#

best_overall_val_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    model = DMN_PLUS(config, word_embedding)

num_runs = 1

weights_dir = '%s/weights' % casefn
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

for run in range(num_runs):

    print('Starting run', run)

    print('==> initializing variables')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:

        sum_dir = '%s/summaries/train/' % casefn + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(sum_dir):
            os.makedirs(sum_dir)
        train_writer = tf.summary.FileWriter(sum_dir, session.graph)

        session.run(init)

        best_val_epoch = 0
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_accuracy = 0.0

        # if args.restore:
        #     print('==> restoring weights')
        #     saver.restore(session, 'weights/task' + str(babi_id) + '.weights')

        print('==> starting training')
        for epoch in range(config['max_epochs']):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy = model.run_epoch( session, train, epoch, train_writer,
                                                          train_op=model.train_step, train=True)
            valid_loss, valid_accuracy = model.run_epoch( session, test)
            print('Training loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(valid_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            print('Vaildation accuracy: {}'.format(valid_accuracy))

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = epoch
                if best_val_loss < best_overall_val_loss:
                    print('Saving weights')
                    best_overall_val_loss = best_val_loss
                    best_val_accuracy = valid_accuracy
                    saver.save(session, '%s/task.weights' % weights_dir)

            # anneal
            if train_loss > prev_epoch_loss * config['anneal_threshold']:
                config['lr'] /= config['anneal_by']
                print('annealed lr to %f' % config['lr'])

            prev_epoch_loss = train_loss

            if epoch - best_val_epoch > config['early_stopping']:
                break
            print('Total time: {}'.format(time.time() - start))

        print('Best validation accuracy:', best_val_accuracy)


