# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import logging
import json
import codecs
import os

import tensorflow as tf
import tensorflow_hub as hub

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from hparams import Hparams
from data_load import get_batches
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu

os.environ["TFHUB_CACHE_DIR"] = 'tfhub_cache'
MAX_SEQ_LEN = 128 # up to 512
# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

# [PAD]
# [UNK]
# [CLS]
# [SEP]
# [MASK]

DATASETFN = 'data/qa/CoQA/000006914.json'

# logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO) 

tf.enable_eager_execution()

#
# bert tokenizer setup
#

with tf.Graph().as_default():

    bert_module = hub.Module(BERT_MODEL_HUB)

    # bert_module.get_signature_names()
    # bert_module.get_input_info_dict('mlm')
    # bert_module.get_input_info_dict('tokens')
    # bert_module.get_output_info_dict(signature='tokens')

    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)

    tokenization_info

    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
        vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                              tokenization_info["do_lower_case"]])


vocab_file
do_lower_case 

tokenizer= bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
# tokenizer = create_tokenizer_from_hub_module()

tokens = tokenizer.tokenize("This here's an example of using the BERT tokenizer")

tokens += ['[PAD]']
tokens

#
# dataset experiments
#


with codecs.open(DATASETFN, 'r', 'utf8') as dataf:
    data = json.loads(dataf.read())

data['info']

def ds_gen():

    txt = data['info']
    tokens_a = tokenizer.tokenize(txt)

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > MAX_SEQ_LEN - 2:
        tokens_a = tokens_a[0:(MAX_SEQ_LEN - 2)]

    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)
    tokens.append("[SEP]")
    input_type_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < MAX_SEQ_LEN:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == MAX_SEQ_LEN
    assert len(input_mask) == MAX_SEQ_LEN
    assert len(input_type_ids) == MAX_SEQ_LEN

    logging.info("*** Example ***")
    logging.info("tokens: %s" % " ".join( [tokenization.printable_text(x) for x in tokens]))
    logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    logging.info("input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

    yield(input_ids,input_mask,input_type_ids)
    # yield(23,[42])

ds = tf.data.Dataset.from_generator(ds_gen, (tf.int64, tf.int64, tf.int64), (tf.TensorShape([MAX_SEQ_LEN]), tf.TensorShape([MAX_SEQ_LEN]), tf.TensorShape([MAX_SEQ_LEN])))
# ds = tf.data.Dataset.from_generator(ds_gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

for value in ds.take(1):
    print(value)

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

hp.train1, hp.train2

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batches(hp.train1, hp.train2,
                                             hp.maxlen1, hp.maxlen2,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batches(hp.eval1, hp.eval2,
                                             100000, 100000,
                                             hp.vocab, hp.batch_size,
                                             shuffle=False)

train_batches.output_types  # ((tf.int32, tf.int32, tf.string), (tf.int32, tf.int32, tf.int32, tf.string))
train_batches.output_shapes # ((TensorShape([Dimension(None), Dimension(None)]),
                            #   TensorShape([Dimension(None)]),
                            #   TensorShape([Dimension(None)])),
                            #  (TensorShape([Dimension(None), Dimension(None)])
                            #   TensorShape([Dimension(None), Dimension(None)]),
                            #   TensorShape([Dimension(None)]),
                            #   TensorShape([Dimension(None)])))


# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

xs

# (<tf.Tensor 'IteratorGetNext:0' shape=(?, ?) dtype=int32>,
#  <tf.Tensor 'IteratorGetNext:1' shape=(?,) dtype=int32>,
#  <tf.Tensor 'IteratorGetNext:2' shape=(?,) dtype=string>)


ys

# (<tf.Tensor 'IteratorGetNext:3' shape=(?, ?) dtype=int32>,
#  <tf.Tensor 'IteratorGetNext:4' shape=(?, ?) dtype=int32>,
#  <tf.Tensor 'IteratorGetNext:5' shape=(?,) dtype=int32>,
#  <tf.Tensor 'IteratorGetNext:6' shape=(?,) dtype=string>)

logging.info("# Load model")
# m = Transformer(hp)

with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
    x, seqlens, sents1 = xs

    # embedding
    enc = tf.nn.embedding_lookup(self.embeddings, x) # (N, T1, d_model)
    enc *= self.hp.d_model**0.5 # scale

    enc += positional_encoding(enc, self.hp.maxlen1)
    enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

    ## Blocks
    for i in range(self.hp.num_blocks):
        with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
            # self-attention
            enc = multihead_attention(queries=enc,
                                      keys=enc,
                                      values=enc,
                                      num_heads=self.hp.num_heads,
                                      dropout_rate=self.hp.dropout_rate,
                                      training=training,
                                      causality=False)
            # feed forward
            enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])

memory = enc
