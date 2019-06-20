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
import bz2
import requests
import collections

from datetime import datetime
from pyquery  import PyQuery
from optparse import OptionParser
from nltools  import misc

PROC_TITLE        = 'twitterscrape'

TWITTER_CORPUS    = '/home/bofh/projects/ai/data/corpora/en/twitter/corpus'
USER_STAT_FN      = '/home/bofh/projects/ai/data/corpora/en/twitter/user_stats.json'

DEFAULT_MAX_TWEETS = 0
DEFAULT_LANG       = 'en'
DEFAULT_QUERY      = ''
DEFAULT_SINCE      = ''
DEFAULT_UNTIL      = ''
DEFAULT_NEAR       = ''
DEFAULT_WITHIN     = ''

MAX_DUPES          = 512

#
# init
#

misc.init_app(PROC_TITLE)

#
# commandline
#

parser = OptionParser("usage: %prog [options] user [ user2 ... ]")

parser.add_option ("-m", "--max_tweets", dest="max_tweets", type = "int", default=DEFAULT_MAX_TWEETS,
                   help="maximum number of tweets to scrape, default: %d" % DEFAULT_MAX_TWEETS)

parser.add_option ("-l", "--lang", dest="lang", type = "str", default=DEFAULT_LANG,
                   help="language, default: %s" % DEFAULT_LANG)

parser.add_option ("-n", "--near", dest="near", type = "str", default=DEFAULT_NEAR,
                   help="search tweets near location, default: none")

parser.add_option ("-w", "--within", dest="within", type = "str", default=DEFAULT_WITHIN,
                   help="search tweets within this distance of near location, default: none")

parser.add_option ("-q", "--query", dest="query", type = "str", default=DEFAULT_QUERY,
                   help="search query, default: none")

parser.add_option ("-s", "--since", dest="since", type = "str", default=DEFAULT_SINCE,
                   help="search for tweets since, default: no limit")

parser.add_option ("-u", "--until", dest="until", type = "str", default=DEFAULT_UNTIL,
                   help="search for tweets until, default: no limit")

parser.add_option ("-v", "--verbose", action="store_true", dest="verbose",
                   help="verbose output")

(options, args) = parser.parse_args()

if options.verbose:
    logging.getLogger().setLevel(logging.DEBUG)
else:
    logging.getLogger().setLevel(logging.INFO)

logging.getLogger().setLevel(logging.DEBUG)

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

#
# corpus preparation
#

for i in range(256):
    for j in range(256):
        dirfn = '%s/%02x/%02x' % (TWITTER_CORPUS, i, j)
        misc.mkdirs(dirfn)

user_stats = {}

if os.path.exists(USER_STAT_FN):
    with open(USER_STAT_FN, 'r') as user_stat_f:
        user_stats = json.loads(user_stat_f.read())

#
# twitter scraping
#

def scrapeStatusPage(cursor, sess, lang='en', search_user='', search_query='', search_since='', search_until='', search_near='', search_within=''):

    logging.debug('scrapeStatusPage, user=%s, query=%s, since=%s, until=%s, cursor=%s' % (search_user, search_query, search_since, search_until, cursor))

    parUrl = ''
    if not search_user and not search_query:
        raise ValueError("User and Query, at least one of them must be specified.")
    else:
        if search_query:
            parUrl += " " + search_query
        if search_user:
            parUrl += ' from:' + search_user
        if search_until:
            parUrl += ' until:' + search_until
        if search_since:
            parUrl += ' since:' + search_since
        if search_near:
            if search_within:
                parUrl += " near:\"" + search_near + "\" within:" + search_within

    # logging.debug(parUrl)
    # logging.debug(cursor)

    # english
    headers_en = {
        'User-Agent': 'Opera/12.0(Windows NT 5.1;U;en)Presto/22.9.168 Version/12.00',
        'Accept-Language': "en-CA,en;q=0.8",
        'X-Requested-With': "XMLHttpRequest"
    }

    # french
    headers_fr = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Accept-Language': "fr-CA,fr;q=0.8",
        'X-Requested-With': "XMLHttpRequest"
    }

    # english
    url_en = "https://twitter.com/i/search/timeline?f=tweets&q=%s&src=typd&l=en&max_position=%s"

    # french
    url_fr = u"https://twitter.com/i/search/timeline?f=tweets&q=%s&src=typd&l=fr&max_position=%s"
    if lang == 'fr':
        url = url_fr
        headers = headers_fr
    else:
        url = url_en
        headers = headers_en

    url = url % (parUrl, cursor)

    # logging.debug('scrapeStatusPage: url=%s' % url) 

    try:
        # print unicode(url)
        r = sess.get(url, headers=headers)
        # print r.url
        # print 'TEXT: -------------------------'
        # print r.text
        # print 'JSON: -------------------------'
        # print r.json()
        # print '-------------------------\n\n\n'
        return r.json()
    except requests.exceptions.RequestException as e:
        logging.error(u'ERROR while ripping %s: %s' % (url, e))

def scrapeCommentPage(user_name, tweet_id, cursor, sess, lang='en'):

    url = "https://twitter.com/i/%s/conversation/%s?include_available_features=1&include_entities=1&max_position=%s&reset_error_state=false"
    url = url % (user_name, tweet_id, cursor)
    # english
    headers_en = {
        'User-Agent': 'Opera/12.0(Windows NT 5.1;U;en)Presto/22.9.168 Version/12.00',
        'Accept-Language': "en-CA,en;q=0.8",
        'X-Requested-With': "XMLHttpRequest"
    }

    # french
    headers_fr = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
        'Accept-Language': "fr-CA,fr;q=0.8",
        'X-Requested-With': "XMLHttpRequest"
    }

    if lang == 'fr':
        headers = headers_fr
    else:
        headers = headers_en

    try:
        r = sess.get(url, headers=headers)
        return r.json()
    except requests.exceptions.RequestException as e:
        logging.error( 'UNEXPECTED EXCEPTION in scapeCommentPage: url=%s : %s' % ( url, e))

def scrapeComments(user_name, tweet_id, cnt_replies, session, max_comments=0):
    cnt_c = 0
    cursor = ''
    total = 0
    has_more = True
    comments = []
    lim = cnt_replies
    if max_comments and lim >max_comments:
        lim = max_comments
    while has_more is True and total < lim:
        page = scrapeCommentPage(user_name, tweet_id, cursor, session)
        cnt_cp, has_more, cursor, pageTweets = scrapePage(page, session, isComment=True)
        if len(pageTweets) == 0:
            logging.debug('Weird, no comments!')
            break
        comments.extend(pageTweets)
        total += len(pageTweets)
        cnt_c += cnt_cp
    # cnt_c should be 0
    return cnt_c, comments

def scrapeTweet(tweetq, session, isComment=False):

    global user_stats

    """
    "" Read the document, and parse it with PyQuery
    """
    # Number of Comments needs to be pass back
    # Number of Tweets is 1, don't need to be pass back
    # Will return number of comments, and the tweet itself
    cnt_c = 0
    twe = {}
    twe["user"] = tweetq.attr("data-screen-name")

    # Process attributes of a tweet div
    twe["replies"]   = int(tweetq("span.ProfileTweet-action--reply span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))
    twe["retweets"]  = int(tweetq("span.ProfileTweet-action--retweet span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))
    twe["favorites"] = int(tweetq("span.ProfileTweet-action--favorite span.ProfileTweet-actionCount").attr("data-tweet-stat-count").replace(",", ""))
    twe['timestamp'] = int(tweetq("small.time span.js-short-timestamp").attr("data-time"))
    twe["date"]      = datetime.fromtimestamp(twe['timestamp']).strftime("%Y-%m-%d %H:%M")
    twe["id"]        = tweetq.attr("data-tweet-id")
    twe["permalink"] = "https://twitter.com" + tweetq.attr("data-permalink-path")

    # Process text area of a tweet div
    textdiv = tweetq("p.js-tweet-text")

    # Process links in a tweet div, including url, hashtags, and mentions contained in the tweet
    links = textdiv('a')
    if len(links) > 0:
        hashtags = []
        mentions = []
        for link in links:
            textUrl = PyQuery(link).attr('data-expanded-url')
            textHashtag = PyQuery(link)('a.twitter-hashtag')('b')
            if len(textHashtag) > 0:
                hashtags.append('#' + textHashtag.text())
            textMention = PyQuery(link)('a.twitter-atreply')('b')
            if len(textMention) > 0:
                mentions.append('@' + PyQuery(textMention).text())
        twe['textUrl'] = ''
        if textUrl is not None:
            twe['textUrl'] = textUrl
        twe['hashtags'] = hashtags
        twe['mentions'] = mentions

    # Process Emojis in a tweet Div
    emojis = textdiv('img.Emoji--forText')
    emojilist = []
    if len(emojis) > 0:
        for emo in emojis:
            textEmoji = PyQuery(emo)
            if textEmoji is not None:
                emoji = {}
                emoji['face'] = textEmoji.attr('alt')
                emoji['url'] = textEmoji.attr('src')
                emoji['title'] = textEmoji.attr('title')
                emojilist.append(emoji)
    twe['emojis'] = emojilist

    # Process Text in a tweet Div
    textq = textdiv.remove('a').remove('img')
    if textq is not None:
        twe["text"] = textq.text()
    else:
        twe['text'] = ''

    # Process optional Geo area of a tweet
    twe["geo"] = ''
    geoArea = tweetq('span.Tweet-geo')
    if len(geoArea) > 0:
        twe["geo"] = geoArea.attr('title')

    # Process comments area if any
    if not isComment and twe['replies'] > 0:
        cn, twe['comments'] = scrapeComments(twe['user'], twe['id'], twe['replies'], session)
        cnt_c = len(twe['comments'])
    else:
        twe['comments'] = []

    if not isComment:
        logging.debug ("TWEET: %-18s (%4d likes, %4d comments): %s %s" % (twe["user"], twe["favorites"], len(twe['comments']), twe['date'], twe['text']))
        # logging.debug ("       url: %s" % twe["permalink"])
    else:
        if twe["favorites"]>0:
            # logging.debug ("       %-18s (%4d likes, %4d comments): %s" % (twe["user"], twe["favorites"], len(twe['comments']), twe['text']))

            if not twe["user"] in user_stats:
                user_stats[twe["user"]] = 0
            user_stats[twe["user"]] += twe["favorites"]

    # Finally return a json of a tweet
    return cnt_c, twe

def scrapePage(page, session, isComment=False):
    cursor = ''
    items = []
    # cnt_cp: Number of comments implies by this page
    # if this page is comment page, no 2-order comments will be retured
    # that means cnt_cp = 0
    cnt_cp = 0
    has_more = False
    if 'items_html' in page:
        if len(page['items_html'].strip()) == 0:
            return cnt_cp, has_more, cursor, items
    if 'has_more_items' in page:
        has_more = page['has_more_items']
    if has_more is False and not isComment:
        time.sleep(4)
    if 'min_position' in page:
        cursor = page['min_position']
    tweets = []
    if 'items_html' in page:
        tweets = PyQuery(page['items_html'])('div.js-stream-tweet')
    if len(tweets) == 0:
        return cnt_cp, has_more, cursor, items
    for tweetArea in tweets:
        tweet_pq = PyQuery(tweetArea)
        try:
            cnt_c, twe = scrapeTweet(tweet_pq, session, isComment)
            items.append(twe)
            cnt_cp += cnt_c
        except Exception as e:
            logging.error('UNEXPECTED EXCEPTION: %s', e)

    return cnt_cp, has_more, cursor, items

#
# main
#

for search_user in args:

    total_tweets     = 0
    total_items      = 0

    cursor           = ''
    sess             = requests.Session()
    cnt_blank        = 0
    pre_cursors      = collections.deque(3 * [""], 3)
    empty_cursor_cnt = 0

    dupes            = 0

    while (options.max_tweets == 0) or (total_tweets < options.max_tweets):

        page = scrapeStatusPage(cursor, sess, 
                                options.lang,
                                search_user,
                                options.query,
                                options.since,
                                options.until,
                                options.near,
                                options.within)


        # print page
        cnt_c, has_more, cursor, page_tweets = scrapePage(page, isComment=False, session=sess)

        if len(page_tweets) == 0:
            cnt_blank += 1
        if len(page_tweets) > 0:
            cnt_blank = 0
        if cnt_blank > 3:
            logging.error('Too many blank pages, terminating this search.')
            break
        total_tweets += len(page_tweets)
        total_items  += cnt_c + len(page_tweets)

        for tweet in page_tweets:
            plink    = tweet['permalink']
            parts    = plink.split('/')
            
            tweet_id = '%0x' % (int(parts[len(parts)-1]))

            tweet_id_a = tweet_id[0:2]
            tweet_id_b = tweet_id[2:4]

            jsonfn = '%s/%s/%s/%s.json' % (TWITTER_CORPUS, tweet_id_a, tweet_id_b, tweet_id)

            # logging.debug('writing: %s', jsonfn)

            if os.path.exists(jsonfn):
                dupes += 1
            else:
                dupes = 0

            with open(jsonfn, 'w') as jsonf:
                jsonf.write(json.dumps(tweet))

        logging.info('%6d tweets from this iteration, total tweets: %6d dupes: %d' % (len(page_tweets),         total_tweets, dupes))
        logging.info('%6d items  from this iteration, total items:  %6d          ' % (cnt_c + len(page_tweets), total_items ))

        #
        # print top user stats
        #

        cnt = 0
        for user, likes in sorted(user_stats.items(), key=lambda x: x[1], reverse=True):
            logging.info('%7d likes by %s' % (likes, user))
            cnt += 1
            if cnt >= 10:
                break

        with open(USER_STAT_FN, 'w') as user_stat_f:
            user_stat_f.write(json.dumps(user_stats))

        if len(cursor.strip()) > 0:
            if cursor in pre_cursors:
                logging.info("No more tweets coming back, terminating the search.")
                break
            else:
                pre_cursors.append(cursor)
                empty_cursor_cnt = 0
        else:
            empty_cursor_cnt += 1

        if empty_cursor_cnt > 4:
            logging.error("Too many empty cursors coming back, terminating the search.")
            break

        if dupes > MAX_DUPES:
            logging.warn("too many dupes, terminating the search.")
            break



