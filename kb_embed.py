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

from optparse import OptionParser
from nltools  import misc

import tensorflow as tf
import tensorflow_hub as hub
# import matplotlib.pyplot as plt
import numpy as np

PROC_TITLE        = 'kb_enwiki_embed'

ABSTRACTS_FN      = 'enwiki-abstracts.txt'
TXT_DIR           = 'data/kb/%s/txt'
EMB_DIR           = 'data/kb/%s/emb'
EMB_FN            = 'data/kb/%s/emb/%s.json'

USE_URL           = "https://tfhub.dev/google/universal-sentence-encoder/2" 
                    # "https://tfhub.dev/google/universal-sentence-encoder-large/3"

DEFAULT_SCORE_FACTOR = 1000.0

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] corpus")

parser.add_option ("-i", "--incremental", action="store_true", dest="incremental",
                   help="incremental run, do not clean embedding dir first.")

parser.add_option ("-s", "--score-factor", dest="scoref", type = "float", default=DEFAULT_SCORE_FACTOR,
                   help="scoring factor, default: %f" % DEFAULT_SCORE_FACTOR)

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

if len(args) != 1:
    parser.print_help()
    sys.exit(1)

corpus = args[0]

#
# cleanup
#

if not options.incremental:

    cmd = 'rm -rf %s' % (EMB_DIR % corpus)
    logging.info(cmd)
    os.system(cmd)

    cmd = 'mkdir -p %s' % (EMB_DIR % corpus)
    logging.info(cmd)
    os.system(cmd)

#
# TF Hub's Universal Sentence Encoder init
#

use_embed = hub.Module(USE_URL)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

#
# iterate collect all txts
#

jobs = []
for f in os.listdir(TXT_DIR % corpus):
    jobs.append(f.replace('.json', ''))

cnt = 0
BATCH_SIZE = 1024

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    for job in chunks(jobs, BATCH_SIZE):

        datas=[]
        messages = []
        all_done = True
        for f in job:
            txtfn = '%s/%s.json' % (TXT_DIR % corpus, f)
            with open(txtfn) as txtf:
                data = json.loads(txtf.read())
                messages.append(data['txt'])
                datas.append(data)
            embfn = EMB_FN % (corpus, f)
            if not os.path.exists(embfn):
                all_done = False

        if all_done:
            continue

        embeddings = session.run(use_embed(messages))

        for i, f in enumerate(job):
            
            datas[i]['emb'] = embeddings[i].tolist()
            datas[i]['score'] = int(datas[i]['rating'] * options.scoref)

            embfn = EMB_FN % (corpus, f)         
            with open(embfn, 'w') as embf:
                embf.write(json.dumps(datas[i]))

            logging.info ('%7d %4d %s -> %s written.' % (cnt, datas[i]['score'], datas[i]['title'], embfn))
            cnt += 1

