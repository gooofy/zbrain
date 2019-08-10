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
import re
import regex
import datetime

from optparse import OptionParser
from nltools  import misc

from xml.sax import make_parser, handler
from bz2file import BZ2File

PROC_TITLE        = 'qa_extract_wikipedia'

WIKIXML           = {
                      'en': '/home/bofh/projects/ai/data/corpora/en/enwiki-20181220-pages-articles-multistream.xml.bz2',
                      'de': '/home/bofh/projects/ai/data/corpora/de/dewiki-latest-pages-articles-multistream.xml.bz2'
                    }

DEFAULT_LANG      = 'en'

QASRC_DIRFN       = 'data/qa_src'

CORPUSNAME        = 'wikipedia_%s'

# DEBUG_LIMIT       = 10
DEBUG_LIMIT       = 0

def unwiki(wiki):
    """
    Remove wiki markup from the text.
    """
    wiki = regex.sub(r'(?i)&nbsp;', ' ', wiki)
    wiki = regex.sub(r'(?i)<br[ \\]*?>', '\n', wiki)
    wiki = regex.sub(r'(?m)<!--.*?--\s*>', '', wiki)
    wiki = regex.sub(r'(?i)<ref[^>]*>[^>]*<\/ ?ref>', '', wiki)
    wiki = regex.sub(r'(?m)<.*?>', '', wiki)
    wiki = regex.sub(r'(?i)&amp;', '&', wiki)

    wiki = regex.sub(r'(?i)\{\{IPA(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
    wiki = regex.sub(r'(?i)\{\{Lang(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
    wiki = regex.sub(r'\{\{[^\{\}]+\}\}', '', wiki)
    wiki = regex.sub(r'(?m)\{\{[^\{\}]+\}\}', '', wiki)
    wiki = regex.sub(r'(?m)\{\|[^\{\}]*?\|\}', '', wiki)
    wiki = regex.sub(r'(?i)\[\[Category:[^\[\]]*?\]\]', '', wiki)
    wiki = regex.sub(r'(?i)\[\[Image:[^\[\]]*?\]\]', '', wiki)
    wiki = regex.sub(r'(?i)\[\[File:[^\[\]]*?\]\]', '', wiki)
    wiki = regex.sub(r'\[\[[^\[\]]*?\|([^\[\]]*?)\]\]', lambda m: m.group(1), wiki)
    wiki = regex.sub(r'\[\[([^\[\]]+?)\]\]', lambda m: m.group(1), wiki)
    wiki = regex.sub(r'\[\[([^\[\]]+?)\]\]', '', wiki)
    wiki = regex.sub(r'(?i)File:[^\[\]]*?', '', wiki)
    wiki = regex.sub(r'\[[^\[\]]*? ([^\[\]]*?)\]', lambda m: m.group(1), wiki)
    wiki = regex.sub(r"({[^}]+}+)", '', wiki)
    wiki = regex.sub(r"''+", '', wiki)
    wiki = regex.sub(r'(?m)^\*$', '', wiki)
    wiki = regex.sub(r'===([^=]+)===', lambda m: m.group(1)+"\n", wiki)
    wiki = regex.sub(r'==([^=]+)==', lambda m: m.group(1)+"\n", wiki)
    wiki = regex.sub(r'=([^=]+)=', lambda m: m.group(1)+"\n", wiki)
   
    wiki = regex.sub(r'\n+', '\n', wiki).strip()
 
    return wiki

# with codecs.open('data/qa_src/wikipedia_en/00243518.txt', 'r', 'utf8') as wikif:
#     txt = wikif.read()
#     unwiki(txt)
# 
# sys.exit()

def mangle_title(title):
    return title.replace('/', '_')

class WikiContentHandler(handler.ContentHandler):
    types_ns = 'http://schemas.microsoft.com/exchange/services/2006/types'

    def __init__(self, filename):
        self.filename = filename
        self.lastpos  = 0

    def startDocument(self):
        self.text        = None
        self.title       = None
        self.in_text     = False
        self.in_title    = False

    def startElement(self, name, attrs):

        global wikibz2f, wikisize

        if name == 'text':
            self.in_text = True
            self.text    = u''
        else:
            self.in_text = False
        if name == 'title':
            self.in_title = True
            self.title    = u''
        else:
            self.in_title = False
            
    def characters(self, content):
        if self.in_title:
            self.title += content
        if self.in_text:
            self.text += content

    def endElement(self, name):
        global article_cnt, toprdict, article_tot
        if name == 'page':
            article_tot += 1
            if self.text and self.title and not self.text.startswith('#REDIRECT'):

                article_cnt += 1
                # outtxtfn = '%s/%s/%08d.txt' % (QASRC_DIRFN, corpusname, article_cnt)
                # with open(outtxtfn, 'w') as outtxtf:
                #     outtxtf.write(self.title)
                #     outtxtf.write(self.text)

                title = mangle_title(self.title.strip())
                txt = unwiki(self.text)

                ds = {'info': title + '\n' + txt, 'date': datetime.datetime.now().isoformat(), 'dlg': []}

                outjsonfn = '%s/%s/%08d.json' % (QASRC_DIRFN, corpusname, article_cnt)
                with open(outjsonfn, 'w') as outjsonf:
                    outjsonf.write(json.dumps(ds))

                if article_tot % 100 == 0:
                    curpos = int(wikibz2f.tell() * 1000 / wikisize)
                    logging.info('[%5.1f%%] %9d (%9d) %-32s -> %s' % (float(curpos)/10.0, article_cnt, article_tot, title, outjsonfn))
                    self.lastpos = curpos

            else:
                if article_tot % 100 == 0:
                    curpos = int(wikibz2f.tell() * 1000 / wikisize)
                    logging.info('[%5.1f%%] %9d (%9d) %-32s -> ---' % (float(curpos)/10.0, article_cnt, article_tot, self.title))
                    self.lastpos = curpos

            self.text     = None
            self.title    = None
        else:
            curpos = int(wikibz2f.tell() * 1000 / wikisize)
            if curpos > self.lastpos:
                logging.info('[%5.1f%%] %9d (%9d) <%s>' % (float(curpos)/10.0, article_cnt, article_tot, name))
                self.lastpos = curpos


#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] foo.txt [bar.txt ...]")

parser.add_option ("-l", "--lang", dest="lang", type = "str", default=DEFAULT_LANG,
                   help="language, default: %s" % DEFAULT_LANG)

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

corpusname = CORPUSNAME % options.lang

#
# cleanup / preparation
#

cmd = 'rm -rf %s/%s' % (QASRC_DIRFN, corpusname)
logging.info(cmd)
os.system(cmd)

cmd = 'mkdir -p %s/%s' % (QASRC_DIRFN, corpusname)
logging.info(cmd)
os.system(cmd)

#
# parse XML
#

parser = make_parser()
parser.setContentHandler(WikiContentHandler(WIKIXML[options.lang]))

article_cnt = 0
article_tot = 0

wikisize = os.path.getsize(WIKIXML[options.lang])

with open(WIKIXML[options.lang], 'rb') as wikibz2f:

    # with codecs.getreader('utf8')(BZ2File(WIKIXML[options.lang], 'r')) as wikif:
    with codecs.getreader('utf8')(BZ2File(wikibz2f)) as wikif:

        parser.parse(wikif)

