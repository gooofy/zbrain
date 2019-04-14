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

from optparse import OptionParser
from nltools  import misc

from annoy import AnnoyIndex

PROC_TITLE        = 'kb_index'

EMB_DIR           = 'data/kb/%s/emb'

NUM_TREES         = 10 # FIXME
SEARCH_K          = 2  # FIXME
DIMENSIONS        = 512
INDEX_FN          = 'data/kb/%s/idx.ann'

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] corpus")

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
# build index
#

logging.info('building index for %s' % (EMB_DIR % corpus))

aidx = AnnoyIndex(DIMENSIONS)  
for f in os.listdir(EMB_DIR % corpus):

    logging.debug('indexing %s' % f)

    with open('%s/%s' % (EMB_DIR % corpus, f)) as embf:
        data = json.loads(embf.read())
        # print(repr(data))

        i = int(f.replace('.json',''))

        aidx.add_item(i, data['emb'])


# for i in xrange(1000):
#     v = [random.gauss(0, 1) for z in xrange(f)]

logging.info('building %d trees' % NUM_TREES)
aidx.build(NUM_TREES) 

aidx.save(INDEX_FN % corpus)
logging.debug('%s written.' % (INDEX_FN % corpus))

# # test index
# 
# u = AnnoyIndex(f)
# u.load(INDEX_FN) # super fast, will just mmap the file
# print(u.get_nns_by_item(0, 1000)) # will find the 1000 nearest neighbors



