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
import random

from optparse import OptionParser
from nltools  import misc
from common import alphabet

PROC_TITLE                 = 'qa_export_transformer-lm'

QASRC_DIRFN                = 'data/qa_src'
TLM_DIRFN                  = 'engines/transformer-lm/data'

DEFAULT_OUTPUT_CORPUS_NAME = 'lm_corpus'

# DEBUG_LIMIT              = 10000
DEBUG_LIMIT                = 0

TEST_VALID_PART            = 100 # use 1/TEST_VALID_PART of the dataset for validation and testing

# QA

TOKEN_DIALOG               = '<|dialog|>'

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] corpus [corpus2 ...]")

parser.add_option ("-o", "--output-corpus", dest="output_corpus_name", type = "str", default=DEFAULT_OUTPUT_CORPUS_NAME,
                   help="output corpus name, default: %s" % DEFAULT_OUTPUT_CORPUS_NAME)

parser.add_option ("-q", "--qa", action="store_true", dest="qa",
                   help="output q/a dialog")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

if len(args) < 1:
    parser.print_usage()
    sys.exit(1)

input_corpora      = args
output_corpus_name = options.output_corpus_name

#
# cleanup / preparation
#
 
cmd = 'rm -rf %s/%s' % (TLM_DIRFN, output_corpus_name)
logging.info(cmd)
os.system(cmd)
cmd = 'mkdir -p %s/%s/train' % (TLM_DIRFN, output_corpus_name)
logging.info(cmd)
os.system(cmd)
cmd = 'mkdir -p %s/%s/valid' % (TLM_DIRFN, output_corpus_name)
logging.info(cmd)
os.system(cmd)
cmd = 'mkdir -p %s/%s/test' % (TLM_DIRFN, output_corpus_name)
logging.info(cmd)
os.system(cmd)

#
# find files
#

ls_files = []

for corpus_name in input_corpora:

    for ls_a in os.listdir('%s/%s' % (QASRC_DIRFN, corpus_name)):

        ls_files.append('%s/%s/%s' % (QASRC_DIRFN, corpus_name, ls_a))

        if len(ls_files) % 1000 == 0:
            logging.info ('%s %7d files...' % (corpus_name, len(ls_files)))

        if DEBUG_LIMIT and (len(ls_files)>=DEBUG_LIMIT):
            logging.warning('DEBUG_LIMIT reached at %d files.' % len(ls_files))
            break

logging.info('found %d files.' % len(ls_files))

logging.info('shuffling...')
random.shuffle(ls_files)
logging.info('shuffling...done.')

#
# generate corpus
#

unknown_chars = set()

def cleanup_text(txt):

    res = ''
    for c in txt:
        if c in alphabet:
            res += alphabet[c]
        else:
            if not c in unknown_chars:
                unknown_chars.add(c)
                logging.warning('unknown char: %s', c)

    return res.strip() + '\n'

fncnt=0
for datasrcfn in ls_files:

    fncnt += 1

    try:
        with open(datasrcfn, 'r') as datasrcf:
            data = json.loads(datasrcf.read())

        txt = cleanup_text(data['info'])

        if 'Ihr Browser muss JavaScript' in txt:
            continue

        # for r in data['dlg']:
        #     txt += '\n' + cleanup_text(r)


        if options.qa:

            if (not 'dlg' in data) or (len(data['dlg'])==0):
                continue

            for r in data['dlg']:
                txt += TOKEN_DIALOG
                txt += cleanup_text(r)

            # print (txt)

        if fncnt % TEST_VALID_PART == 1:
            secname = 'test'
        elif fncnt % TEST_VALID_PART == 2:
            secname = 'valid'
        else:
            secname = 'train'

        outfn = '%s/%s/%s/%08d.txt' % (TLM_DIRFN, output_corpus_name, secname, fncnt)
        with open(outfn, 'w') as outf:
            outf.write(txt)

        logging.info ('%7d/%7d %s written. %s' % (fncnt, len(ls_files), outfn, txt[:30].replace('\n', ' ')))

    except:
        logging.exception('exception caught %s' % repr(data))


if options.verbose:
    for c in sorted(unknown_chars):
        print("    %s : '', # %s" % (repr(c), c))

