#!/usr/bin/env python3

import argparse
import json
import os
import time
import sys
import logging

from readability import Document
from bs4 import BeautifulSoup

import requests

import numpy      as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from gpt2 import model, sample, encoder

# CHECKPOINT_DIR = 'checkpoint'
CHECKPOINT_DIR     = 'engines/gpt-2/checkpoint'
SAMPLE_DIR         = 'samples'

MAX_TOKENS         = 384
MIN_INFO_TOKENS    = 196
SEPARATOR_ANSWER   = '<|answer|>'

parser = argparse.ArgumentParser( description='generate twitter comments using a GPT-2 model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('tweet_url', metavar='TWEETURL', type=str, help='Tweet URL')

parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')

parser.add_argument('--url', metavar='URL', type=str, default=None, help='article link')

args = parser.parse_args()

logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("readability").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

enc = encoder.get_encoder(args.model_name)
hparams = model.default_hparams()
with open(os.path.join('engines/gpt-2/models', args.model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if args.sample_length > hparams.n_ctx:
    raise ValueError(
        "Can't get samples longer than window size: %s" % hparams.n_ctx)

if args.model_name == '345M':
    args.memory_saving_gradients = True
    args.only_train_transformer_layers = True

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

token_bos      = enc.encode('<|bos|>')
token_eos      = enc.encode('<|eos|>')
token_speaker1 = enc.encode('<|speaker1|>')
token_speaker2 = enc.encode('<|speaker2|>')
token_pad      = enc.encode(' ')

l_token_bos     = len(token_bos)
l_token_eos     = len(token_eos)
l_token_speaker = max(len(token_speaker2), len(token_speaker1))
l_token_pad     = len(token_pad)

def build_input_from_segments2(info, history, response):

    # build up sequence, watch out for length limits

    tokens_left = MAX_TOKENS

    # response

    i_history = len(history)
    if response:
        sequence = (token_speaker2 if i_history % 2 else token_speaker1) + response + (token_speaker1 if i_history % 2 else token_speaker2)
    else:
        sequence = token_speaker2 if i_history % 2 else token_speaker1

    tokens_left = MAX_TOKENS - len(sequence)

    # prepend dialog history

    min_info_tokens = min(MIN_INFO_TOKENS, len(info)+l_token_bos)

    while True:
        i_history -= 1
        if i_history<0:
            break
        if (tokens_left - len(history[i_history]) < (min_info_tokens - l_token_speaker)):
            break

        # sequence.insert(0, [token_speaker2 if i_history % 2 else token_speaker1] + history[i_history])
        sequence = (token_speaker2 if i_history % 2 else token_speaker1) + history[i_history] + sequence
        tokens_left = MAX_TOKENS - len(sequence)

    # prepend info

    logging.debug('prepend info: len(token_bos)=%d, tokens_left=%d, len(sequence)=%d', 
                  len(token_bos), tokens_left, len(sequence))

    sequence = token_bos + info[:tokens_left-1-len(token_bos)] + sequence

    logging.debug('prepended info: len(sequence)=%d', len(sequence))

    # sequence.insert(0, [ token_bos ] + info [:tokens_left-1])
    # tokens_left -= len(sequence[0])

    # while len(sequence)<MAX_TOKENS:
    #     sequence.extend(token_pad)

    return sequence

#
# scrape tweet
#

page = requests.get(args.tweet_url)

soup = BeautifulSoup(page.content, 'html.parser')

ps = soup.find_all('p', class_='js-tweet-text')
tweet_text = ''
info_url = ''
for c in ps[0].children:

    logging.debug('child: %s' % c)
    if isinstance(c, str):
        tweet_text += c
    elif c.has_attr('data-expanded-url'):
        if 'http' in c['data-expanded-url']:
            info_url = c['data-expanded-url']
    elif c.has_attr('href'):
        if (not info_url) and ('http' in c['href']):
            info_url = c['href']
        else:
            tweet_text += ' ' + c.get_text()
    else:
        tweet_text += c.get_text()

logging.info('tweet text: %s, info_url: %s' % (tweet_text, info_url))

#
# retrieve article, if any
#

info_text = None
if info_url:
    
    response = requests.get(info_url, verify=True, timeout=7)
    # response.text
    doc = Document(response.text)
    doc.title()
    html = doc.summary()

    # html

    soup = BeautifulSoup(html, features="lxml")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    info_text = '\n'.join(chunk for chunk in chunks if chunk)

    logging.info('info_text: %s' % info_text)

    if 'Auf deiner Timeline findest du in Echtzeit die Informationen, die dir wichtig sind.' in info_text:
        info_text=None

if info_text:
    question_tokens = build_input_from_segments2(enc.encode(info_text), [], enc.encode(tweet_text))
else:
    question_tokens = build_input_from_segments2(enc.encode(tweet_text), [], None)

logging.info('question_tokens     : %s', question_tokens)
logging.info('question_tokens len : %s', len(question_tokens))
logging.info('sample_length       : %s', args.sample_length)

BATCH_SIZE = 1

with tf.compat.v1.Session(config=config) as sess:
    context = tf.compat.v1.placeholder(tf.int32, [BATCH_SIZE, None])
    output = model.model(hparams=hparams, X=context)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=context[:, 1:], logits=output['logits'][:, :-1]))

    tf_sample = sample.sample_sequence( hparams      = hparams,
                                        length       = args.sample_length,
                                        context      = context,
                                        batch_size   = BATCH_SIZE,
                                        temperature  = 1.0,
                                        top_k        = 40)

    all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]

    saver = tf.compat.v1.train.Saver( var_list    = all_vars )
    sess.run(tf.global_variables_initializer())

    if args.restore_from == 'latest':
        ckpt = tf.train.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name))
        if ckpt is None:
            # Get fresh GPT weights if new run.
            ckpt = tf.train.latest_checkpoint(
                os.path.join('engines/gpt-2/models', args.model_name))
    elif args.restore_from == 'fresh':
        ckpt = tf.train.latest_checkpoint(
            os.path.join('engines/gpt-2/models', args.model_name))
    else:
        ckpt = tf.train.latest_checkpoint(args.restore_from)
    logging.info('Loading checkpoint %s ...', ckpt)
    saver.restore(sess, ckpt)

    logging.info ('question       : %s ', enc.decode(question_tokens))
    logging.info ('question length: %s ', len(question_tokens))

    out = sess.run( tf_sample, feed_dict = { context: [ question_tokens ] })
    answer = enc.decode(out[0])

    for a in answer.split('<|speaker')[1:]:

        print ('answer     : %s' % a)

