#!/usr/bin/env python3

import argparse
import json
import os
import time

import numpy      as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2

from gpt2 import model, sample, encoder

# CHECKPOINT_DIR = 'checkpoint'
CHECKPOINT_DIR = 'engines/gpt-2/checkpoint'
SAMPLE_DIR = 'samples'

SEPARATOR_QUESTION = '<|question|>'
SEPARATOR_ANSWER   = '<|answer|>'

parser = argparse.ArgumentParser( description='Chat with your GPT-2 model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')

parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", or a path to a checkpoint file')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
# parser.add_argument('--sample_every', metavar='N', type=int, default=100, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=1023, help='Sample this many tokens')
# parser.add_argument('--sample_num', metavar='N', type=int, default=1, help='Generate this many samples')
# parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')


args = parser.parse_args()
enc = encoder.get_encoder(args.model_name)
hparams = model.default_hparams()
with open(os.path.join('engines/gpt-2/models', args.model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

# if args.sample_length > hparams.n_ctx:
#     raise ValueError(
#         "Can't get samples longer than window size: %s" % hparams.n_ctx)

if args.model_name == '345M':
    args.memory_saving_gradients = True
    args.only_train_transformer_layers = True

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

BATCH_SIZE = 1

with tf.Session(config=config) as sess:
    context = tf.placeholder(tf.int32, [BATCH_SIZE, None])
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

    saver = tf.train.Saver( var_list    = all_vars )
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
    print('Loading checkpoint', ckpt)
    saver.restore(sess, ckpt)

    print('Chat session starts.')

    question = 'What is your name?'
    # question = 'What about trump?'

    # info = 'Stanford has begun posting lectures from my course Natural Language Understanding on YouTube (a few are still to come): (link: https://www.youtube.com/playlist?list=PLoROMvodv4rObpMCir6rNNUlFAn56Js20) youtube.com/playlist?list=… I\'m actually happiest with the "bake-offs", which don\'t appear much in the videos but can be found here: (link: http://web.stanford.edu/class/cs224u/) web.stanford.edu/class/cs224u/'
    # info = 'Iran made a very big mistake!'
    # info = 'Congratulations to President Lopez Obrador — Mexico voted to ratify the USMCA today by a huge margin. Time for Congress to do the same here!'
    # info = 'Ooooooh Noooooo SHE DIDN\'T!! ￼ @realDonaldTrump ￼ #TrumpsAnInternationalDisgrace'
    # info = 'A lot of bullshit gets patiently slayed here by @FlickRubicon and @Scottludlam ￼￼ #WikiLeaks #Assange (link: https://arena.org.au/may-curious-eyes-never-run-dry-by-felicity-ruby-and-scott-ludlam/) arena.org.au/may-curious-ey…'
    # info = 'Bimbo is an elefant. I am 30 years old.'
    # q = 'Hello my name is Bimbo what\'s yours?'


    info = 'The RetroBeat is a weekly column that looks at gaming’s past, diving'
    q = 'The RetroBeat: Timespinner is a beautiful Symphony of the Night indie tribute'

    question = '<|bos|>' + info + '<|speaker1|>' + q + '<|speaker2|>'

    question_tokens = enc.encode(question)
    # question_tokens = enc.encode(question + ' ' + SEPARATOR_QUESTION)
    # context_tokens = enc.encode('What is the meaning of life?')

    print ('question > %s \n' % enc.decode(question_tokens))

    out = sess.run( tf_sample, feed_dict = { context: [ question_tokens ] })
    answer = enc.decode(out[0]).split(SEPARATOR_ANSWER, 1)[0]

    print ('answer > %s \n' % answer)

