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

import os
import sys
import codecs
import time
import json
import logging
import hashlib
import tqdm
import random

from optparse        import OptionParser
from nltools         import misc

PROC_TITLE        = 'qa_extract_heise'

HEISE_CORPUSDIR   = '/home/bofh/projects/ai/data/corpora/de/heise_news'

QASRC_DIRFN       = 'data/qa_src'

# DEBUG_LIMIT       = 10
DEBUG_LIMIT       = 0

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options]")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

#
# cleanup / preparation
#
 
cmd = 'mkdir -p %s/heise' % QASRC_DIRFN
logging.info(cmd)
os.system(cmd)

#
# find files
#

ls_files = []

for ls_a in os.listdir(HEISE_CORPUSDIR):

    if not ls_a.endswith('.json'):
        continue

    ls_files.append('%s/%s' % (HEISE_CORPUSDIR, ls_a))

    if len(ls_files) % 100 == 0:
        logging.info ('%7d files...' % len(ls_files))

    if DEBUG_LIMIT and (len(ls_files)>=DEBUG_LIMIT):
        logging.warning('DEBUG_LIMIT reached at %d files.' % len(ls_files))
        break

logging.info('found %d files.' % len(ls_files))

#
# convert tweets
#

cnt = 0
for jsonfn in ls_files:

    logging.info('%s...' % jsonfn)

    with open(jsonfn, 'r') as jsonf:
        data = json.loads(jsonf.read())

        for article in data:
            
            ds = {'info': article['headline'] + '\n' + article['text'], 'date': article['timestamp'], 'dlg': []}

            outjsonfn = '%s/heise/%08d.json' % (QASRC_DIRFN, cnt)
            cnt += 1

            with open(outjsonfn, 'w') as outjsonf:
                outjsonf.write(json.dumps(ds))

            logging.debug ('%-20s: %s written. %s' % (article['author'], outjsonfn, article['headline'][:100]))

