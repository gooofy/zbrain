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

from html.entities   import name2codepoint
from html.parser     import HTMLParser
from optparse        import OptionParser
from nltools         import misc

PROC_TITLE        = 'qa_extract_parole'

PAROLE_CORPUSDIR  = '/home/bofh/projects/ai/data/corpora/de/German Parole Corpus/DE_Parole'

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
 
cmd = 'mkdir -p %s/parole' % QASRC_DIRFN
logging.info(cmd)
os.system(cmd)

#
# extract json
#

cnt = 0

class ParoleParser(HTMLParser):

    def __init__(self):

        HTMLParser.__init__(self)

        self.in_par = False
        self.text = ''
        self.buf = ''

    def handle_starttag(self, tag, attrs):
        # print "Encountered a start tag:", tag
        if tag == 'p' or tag == 'head':
            self.in_par = True
            self.buf = ""
        elif tag == 'div1':
            self.text = ''

    def handle_endtag(self, tag):

        global cnt

        if tag == 'p' or tag == 'head':
            self.in_par = False
            # print (u"PAR: %s" % self.buf).encode('UTF8')

            self.text += misc.compress_ws(self.buf.replace('\n', ' ')) + '\n'

        if tag == 'div1':
            # logging.info('text: %s', self.text)

            ds = {'info': self.text, 'dlg': []}

            outjsonfn = '%s/parole/%08d.json' % (QASRC_DIRFN, cnt)
            cnt += 1

            with open(outjsonfn, 'w') as outjsonf:
                outjsonf.write(json.dumps(ds))

            logging.info ('%s written. %s' % (outjsonfn, self.text.replace('\n', ' ').strip()[:100]))


    def handle_data(self, data):
        if self.in_par and len(data) > 0:
            # print "About to add: %s" % repr(data)
            self.buf += data

    def handle_entityref(self, name):
        if self.in_par:
            c = ''
            if name == 'star':
                c = u'*'
            elif name == 'bquot':
                c = u'"'
            elif name == 'equot':
                c = u'"'
            elif name == 'lowbar':
                c = u'_'
            elif name == 'parole.tax':
                c = u''
            else:
                if name in name2codepoint:
                    c = unichr(name2codepoint[name])
                else:
                    logging.warning("unknown entityref: %s" % name)
                    c = ''
            # print "Named ent:", c
            self.buf += c

def parole_crawl(path, debug_sgm_limit):
    num_files = 0

    files = os.listdir(path)
    for file in files:
        if debug_sgm_limit > 0 and num_files > debug_sgm_limit:
            return num_files

        p = "%s/%s" % (path, file)

        if os.path.isdir(p):
            num_files += parole_crawl(p, debug_sgm_limit)
            continue

        if not p.endswith('.sgm'):
            continue

        logging.info("%8d: found sgm: %s" % (num_files, p))
        num_files += 1

        pp = ParoleParser()

        with codecs.open(p, 'r', 'utf8', 'ignore') as inf:
            while True:
                sgmldata = inf.read(1024)
                if not sgmldata:
                    break
                pp.feed(sgmldata)

        pp.close()

    return num_files

parole_crawl(PAROLE_CORPUSDIR, 0)

