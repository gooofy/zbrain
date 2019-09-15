#!/bin/bash

# ./twitterscrape.py -v -s 2019-06-01 twitter201906 /home/bofh/projects/ai/data/corpora/en/twitter/twitter_sources.txt

#   -U USER_STATS, --user-stats=USER_STATS
#                         user stats file, default: /home/bofh/projects/ai/data/corpora/en/twitter/user_stats.json


./twitterscrape.py -l de -s 2019-08-01 twitter_de_201908 -U /home/bofh/projects/ai/data/corpora/en/twitter/user_stats_de.json /home/bofh/projects/ai/data/corpora/en/twitter/twitter_sources_de.txt

