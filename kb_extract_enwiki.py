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
import re

from optparse import OptionParser
from nltools  import misc

from xml.sax import make_parser, handler
from bz2file import BZ2File

from mwlib.uparser import simpleparse
from wiki2plain    import Wiki2Plain

PROC_TITLE        = 'kb_extract_enwiki'

ENWIKI            = '/home/bofh/projects/ai/data/corpora/en/enwiki-20181220-pages-articles-multistream.xml.bz2'

RANKINGS          = ['data/kb/enwiki/wikirank/enwiki-2018-h.txt',
                     'data/kb/enwiki/wikirank/enwiki-2018-indegree.txt',
                     'data/kb/enwiki/wikirank/enwiki-2018-pr-3.txt',
                     'data/kb/enwiki/wikirank/enwiki-2018-pv.txt' ]

RANKING_TITLES    = 'data/kb/enwiki/wikirank/enwiki-2018.titles'
                    # data/kb/enwiki/wikirank/enwiki-2018.uris

MAX_ABSTRACT_LEN  = 512
MAX_ABSTRACT_LINES= 5
MAX_ARTICLES      = 100000

DB_DIR            = 'data/kb/enwiki/txt'
DB_FN             = 'data/kb/enwiki/txt/%d.json'

wikitxt = "=h1=\n*item 1\n*item2\n==h2==\nsome [[Link|caption]] there\n"

def unwiki(wiki):
    """
    Remove wiki markup from the text.
    """
    wiki = re.sub(r'(?i)&nbsp;', ' ', wiki)
    wiki = re.sub(r'(?i)<br[ \\]*?>', '\n', wiki)
    wiki = re.sub(r'(?m)<!--.*?--\s*>', '', wiki)
    wiki = re.sub(r'(?i)<ref[^>]*>[^>]*<\/ ?ref>', '', wiki)
    wiki = re.sub(r'(?m)<.*?>', '', wiki)
    wiki = re.sub(r'(?i)&amp;', '&', wiki)

    wiki = re.sub(r'(?i)\{\{IPA(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
    wiki = re.sub(r'(?i)\{\{Lang(\-[^\|\{\}]+)*?\|([^\|\{\}]+)(\|[^\{\}]+)*?\}\}', lambda m: m.group(2), wiki)
    wiki = re.sub(r'\{\{[^\{\}]+\}\}', '', wiki)
    wiki = re.sub(r'(?m)\{\{[^\{\}]+\}\}', '', wiki)
    wiki = re.sub(r'(?m)\{\|[^\{\}]*?\|\}', '', wiki)
    wiki = re.sub(r'(?i)\[\[Category:[^\[\]]*?\]\]', '', wiki)
    wiki = re.sub(r'(?i)\[\[Image:[^\[\]]*?\]\]', '', wiki)
    wiki = re.sub(r'(?i)\[\[File:[^\[\]]*?\]\]', '', wiki)
    wiki = re.sub(r'\[\[[^\[\]]*?\|([^\[\]]*?)\]\]', lambda m: m.group(1), wiki)
    wiki = re.sub(r'\[\[([^\[\]]+?)\]\]', lambda m: m.group(1), wiki)
    wiki = re.sub(r'\[\[([^\[\]]+?)\]\]', '', wiki)
    wiki = re.sub(r'(?i)File:[^\[\]]*?', '', wiki)
    wiki = re.sub(r'\[[^\[\]]*? ([^\[\]]*?)\]', lambda m: m.group(1), wiki)
    wiki = re.sub(r"({[^}]+}+)", '', wiki)
    wiki = re.sub(r"''+", '', wiki)
    wiki = re.sub(r'(?m)^\*$', '', wiki)
    wiki = re.sub(r'===([^=]+)===', lambda m: m.group(1)+"\n", wiki)
    wiki = re.sub(r'==([^=]+)==', lambda m: m.group(1)+"\n", wiki)
    wiki = re.sub(r'=([^=]+)=', lambda m: m.group(1)+"\n", wiki)
    
    return wiki

def mangle_title(title):
    return title.replace('/', '_')

# print (unwiki(wikitxt))

# sys.exit(0)

class WikiContentHandler(handler.ContentHandler):
    types_ns = 'http://schemas.microsoft.com/exchange/services/2006/types'

    def __init__(self, filename):
        self.filename = filename

    def startDocument(self):
        self.text        = None
        self.title       = None
        self.in_text     = False
        self.in_title    = False

    def startElement(self, name, attrs):
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

                title = mangle_title(self.title.strip())

                if title in toprdict:

                    txt = unwiki(self.text)
                    abstract = ""
                    num_abstract = 0
                    for line in txt.split("\n"):
                        if len(line.strip())<5:
                            continue
                        if abstract:
                            abstract += " "
                        abstract += line.strip()
                        num_abstract += 1
                        if (len(abstract) > MAX_ABSTRACT_LEN) or (num_abstract > MAX_ABSTRACT_LINES):
                            break

                    data = {'title' : title,
                            'rating': toprdict[title],
                            'txt'   : abstract}

                    fn = DB_FN % article_cnt
                    with open(fn, 'w') as dbf:
                        dbf.write(json.dumps(data))

                    article_cnt += 1
                    # if article_cnt % 100 == 0:
                    logging.info('%9d (%9d) %s -> %s' % (article_cnt, article_tot, title, fn))

            self.text     = None
            self.title    = None


#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] foo.txt [bar.txt ...]")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

#
# cleanup / preparation
#

cmd = 'rm -rf %s' % DB_DIR
logging.info(cmd)
os.system(cmd)

cmd = 'mkdir -p %s' % DB_DIR
logging.info(cmd)
os.system(cmd)

#
# use page rank data to score articles
#

# step 1 : normalization

ranking_max = {}

for ranking in RANKINGS:
    logging.info('normalizing %s ...' % ranking)
    ranking_max[ranking] = 0.0
    with open(ranking, 'r') as rankingf:
        for line in rankingf:
            r = float(line.strip())
            if r > ranking_max[ranking]:
                ranking_max[ranking] = r
    logging.info('normalizing %s ... done. max is %f' % (ranking, ranking_max[ranking]))

# step 2 : read titles

titles = []
logging.info('reading titles from %s ...' % RANKING_TITLES)
with codecs.open(RANKING_TITLES, 'r', 'utf8') as rf:
    for line in rf:
        titles.append(line.strip())
logging.info('reading titles from %s ... done. %d titles.' % (RANKING_TITLES, len(titles)))

# step 3 : compute compound ranking

rdict = {} # title -> ranking

for ranking in RANKINGS:
    logging.info('reading rankings from %s ...' % ranking)
    with open(ranking, 'r') as rankingf:
        idx = 0
        for line in rankingf:
            r = float(line.strip()) / ranking_max[ranking]
            if idx > len(titles):
                logging.error('extra entries in %s detected!' % ranking)
                break
            title = mangle_title(titles[idx])
            if not title in rdict:
                rdict[title] = r 
            else:
                rdict[title] += r
            idx += 1
    logging.info('reading rankings from %s ... done.' % ranking)

# step 4 : extract top titles

toprdict = {} # title -> ranking

count = 0
for i in sorted(rdict.iteritems(), key=lambda x: x[1], reverse=True):
    logging.debug('%7d %6.3f %s' % (count, i[1], i[0]))

    toprdict[i[0]] = i[1]

    count += 1
    if count > MAX_ARTICLES:
        break

# print (repr(toprdict))

#
# parse XML
#

parser = make_parser()
# parser.setFeature(handler.feature_namespaces, True)
parser.setContentHandler(WikiContentHandler(ENWIKI))

article_cnt = 0
article_tot = 0

with codecs.getreader('utf8')(BZ2File(ENWIKI, 'r')) as wikif:

    # for line in wikif:
    #     print (line)

    parser.parse(wikif)

