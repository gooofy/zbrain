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

import xml.etree.ElementTree as ET
from optparse        import OptionParser
from nltools         import misc

PROC_TITLE        = 'qa_extract_europarl'

EUROPARL_CORPUSDIR  = '/home/bofh/projects/ai/data/corpora/de/Europarl'

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
 
cmd = 'mkdir -p %s/europarl' % QASRC_DIRFN
logging.info(cmd)
os.system(cmd)

#
# extract json
#

cnt = 0

def europarl_crawl(path, debug_sgm_limit):
    global cnt

    num_files = 0

    files = os.listdir(path)
    for file in files:
        if debug_sgm_limit > 0 and num_files > debug_sgm_limit:
            return num_files

        p = "%s/%s" % (path, file)

        if os.path.isdir(p):
            num_files += europarl_crawl(p, debug_sgm_limit)
            continue

        if not p.endswith('.xml'):
            continue

        logging.info("%8d: found xml: %s" % (num_files, p))
        num_files += 1

        tree = ET.parse(p)

        root = tree.getroot()
        for chapter in root.findall('CHAPTER'):
            for speaker in chapter.findall('SPEAKER'):
                logging.info('speaker.')
                text = ''
                for p in speaker.findall('P'):
                    for s in p.findall('s'):
                        if text:
                            text += '\n'
                        text += s.text
                # logging.info(text)
                ds = {'info': text, 'dlg': []}

                outjsonfn = '%s/europarl/%08d.json' % (QASRC_DIRFN, cnt)
                cnt += 1

                with open(outjsonfn, 'w') as outjsonf:
                    outjsonf.write(json.dumps(ds))

                logging.info ('%s written. %s' % (outjsonfn, text.replace('\n', ' ').strip()[:100]))

    return num_files

europarl_crawl(EUROPARL_CORPUSDIR, 0)


