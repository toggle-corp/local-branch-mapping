import pandas as pd

from tweet_parser.tweet import Tweet

from tweet_parser.tweet_parser_errors import NotATweetError

import fileinput

import json

from tqdm import tqdm

import json_lines



country = 'Netherlands'

data = []

df = pd.DataFrame()

df_retweeted = pd.DataFrame()

for line in tqdm(fileinput.FileInput("twitter_data/tweets_"+country+".json")):

    try:

        tweet_dict = json.loads(line)

        tweet = Tweet(tweet_dict)

        

        try:

            user_entered_text = tweet.user_entered_text

        except:

            user_entered_text = ""

        try:

            quote_or_rt_text = tweet.quote_or_rt_text

        except:

            quote_or_rt_text = ""

        try:

            profile_location = tweet.profile_location

        except:

            profile_location = ""

        try:

            in_reply_to = tweet.in_reply_to_user_id

        except:

            in_reply_to = ""

        try:

            embedded_tweet = tweet.embedded_tweet

            rt_id = embedded_tweet.id

            rt_name = embedded_tweet.name

            rt_screen_name = embedded_tweet.screen_name

            rt_created_at = embedded_tweet.created_at

            rt_geo_coordinates = embedded_tweet.geo_coordinates

            try:

                rt_profile_location = embedded_tweet.profile_location

            except:

                rt_profile_location = ""

        except:

            rt_id = ""

            rt_name = ""

            rt_screen_name = ""

            rt_created_at = ""

            rt_geo_coordinates = ""

            rt_profile_location = ""

        

        

        df = df.append({'id': tweet.id,

                        'name': tweet.name,

                        'screen_name': tweet.screen_name,

                        'created_at': tweet.created_at_datetime,

                        'geo_coordinates': tweet.geo_coordinates,

                        'profile_location': profile_location,

                        'tweet_type': tweet.tweet_type,

                        'in_reply_to': in_reply_to,

                        'lang': tweet.lang,

                        'text': tweet.text,

                        'user_entered_text': user_entered_text,

                        'rt_text': quote_or_rt_text,

                        'tweet_links': tweet.most_unrolled_urls,

                        'user_mentions': tweet.user_mentions,

                        'hashtags': tweet.hashtags,

                        'media_urls': tweet.media_urls,

                        'rt_id': rt_id,

                        'rt_name': rt_name,

                        'rt_screen_name': rt_screen_name,

                        'rt_created_at': rt_created_at,

                        'rt_geo_coordinates': rt_geo_coordinates,

                        'rt_profile_location': rt_profile_location},

                        ignore_index=True)

    except (json.JSONDecodeError, NotATweetError):

        print('error')

        pass

    

print(df.info())

print(df.describe())



df.to_csv("twitter_data_processed/tweets_"+country+".csv", sep=',')
pd.set_option('max_colwidth', 400)

df['text']