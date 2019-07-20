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

import newspaper
from newspaper       import Article
from multiprocessing import Pool
from optparse        import OptionParser
from nltools         import misc

PROC_TITLE        = 'qa_extract_twitter'

TWITTER_CORPUSDIR = '/home/bofh/projects/ai/data/corpora/en/twitter'

QASRC_DIRFN       = 'data/qa_src'

# DEBUG_LIMIT       = 10
DEBUG_LIMIT       = 0

BLOCKLIST         = ['wapo.st', 'washingtonpost.com', 'twitter.com', 'tagesspiegel.de']

NUM_PROCS         = 32

USER_AGENTS = ['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.80 Safari/537.36',
               'Mozilla/5.0 (Linux; Android 8.0.0; SM-G930F Build/R16NW; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/74.0.3729.157 Mobile Safari/537.36',
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
               'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/74.0.3729.157 Safari/537.36',
               'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/6.0)',
               'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)',
               'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; wbx 1.0.0; rv:11.0) like Gecko',
               'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko',
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:67.0) Gecko/20100101 Firefox/67.0',
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0',
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:64.0) Gecko/20100101 Firefox/64.0',
               'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:63.0) Gecko/20100101 Firefox/63.0',
               'Mozilla/5.0 (Windows NT 6.2; rv:63.0) Gecko/20100101 Firefox/63.0',
               'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:62.0) Gecko/20100101 Firefox/62.0'
               'Opera/9.80 (Windows NT 6.1; WOW64) Presto/2.12.388 Version/12.18']
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
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

# readability.readability

# logging.getLogger("readability").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if len(args) != 1:
    parser.print_usage()
    sys.exit(1)

corpus_name = args[0]

#
# cleanup / preparation
#
 
cmd = 'mkdir -p %s/%s' % (QASRC_DIRFN, corpus_name)
logging.info(cmd)
os.system(cmd)

#
# find files
#

ls_files = []

for ls_a in os.listdir('%s/%s' % (TWITTER_CORPUSDIR, corpus_name)):

    ls_files.append('%s/%s/%s' % (TWITTER_CORPUSDIR, corpus_name, ls_a))

    if len(ls_files) % 100 == 0:
        logging.info ('%7d files...' % len(ls_files))

    if DEBUG_LIMIT and (len(ls_files)>=DEBUG_LIMIT):
        logging.warning('DEBUG_LIMIT reached at %d files.' % len(ls_files))
        break

logging.info('found %d files. shuffling...' % len(ls_files))
random.shuffle(ls_files)
logging.info('found %d files. shuffling... done.' % len(ls_files))

#
# convert tweets
#

def convert_tweet(twitter_dumpfn):
    try:
        with open(twitter_dumpfn, 'r') as dumpf:
            data = json.loads(dumpf.read())
        if len(data['comments'])==0:
            return

        jsonfn = '%s/%s/%s.json' % (QASRC_DIRFN, corpus_name, data['id'])
        if os.path.exists(jsonfn):
            return

        url = data['textUrl'] if 'textUrl' in data else ''

        text = ''

        if url:

            skip = False
            for blocked_url in BLOCKLIST:
                if blocked_url in url:
                    skip = True
            if skip:
               return 

            logging.debug ('%-20s: %s ... ' % (data['user'], url))

            config = newspaper.Config()
            config.browser_user_agent = random.choice(USER_AGENTS)

            article = Article(url=url, config=config)
            article.download()
            article.parse()

            text = article.text

            # print (text)

        # text

        if text:
            ds = {'info': text, 'date': data['date'], 'dlg': [data['text']]}
        else:
            ds = {'info': data['text'], 'date': data['date'], 'dlg': []}

        fav = 0
        for c in data['comments']:
            if c['favorites'] == 0:
                continue
            ds['dlg'].append(c['text'])
            fav += 1

        if (not text) and (fav == 0):
            return

        # print(repr(ds))

        with open(jsonfn, 'w') as jsonf:
            jsonf.write(json.dumps(ds))

        logging.debug ('%-20s: %s written. %s' % (data['user'], jsonfn, url[:30]))

    except newspaper.article.ArticleException as ae:
        
        logging.info ('%-20s: %s' % (data['user'], str(ae)))

    except:
        logging.exception('exception caught %s' % repr(data))



with Pool(NUM_PROCS) as p:

    for _ in tqdm.tqdm(p.imap_unordered(convert_tweet, ls_files), total=len(ls_files)):
        pass

