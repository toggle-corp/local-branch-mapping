# load libraries and define fields

import pandas as pd

import fileinput

import json

from tqdm import tqdm

from ast import literal_eval

import numpy as np

country = 'Malawi'



post_fields = ['id',

               'created_time',

               'from',

               'type',

               'story',

               'story_tags',

               'message',

               'message_tags',

               'link',

               'name',

               'place',

               'coordinates',

               'shares',

               'updated_time',

               'is_popular',

               'permalink_url']
# load facebook posts



data = []

df = pd.DataFrame()

df_retweeted = pd.DataFrame()

for line in tqdm(fileinput.FileInput("fb_data_backup/page_posts_"+country+".jsonl")):

    try:

        fb_page = json.loads(line)

        df = df.append(fb_page, ignore_index=True)

        

    except (json.JSONDecodeError, NotATweetError):

        print('error')

        pass

    

# drop duplicates

# df.drop_duplicates(subset ="id", keep = 'last', inplace = True) 

    

# inspect

print(df.info())
# expand data

data = df['data'].apply(pd.Series)

data = data.rename(columns = lambda x : 'data_' + str(x))

posts = pd.DataFrame()

for x in range(len(data.columns)):

    post = data['data_'+str(x)].apply(pd.Series)

    post.dropna(subset=['id'], inplace = True) # drop nan

    posts = posts.append(post, ignore_index=True)

posts
# expand field 'from'

def expand_from(x):

        xd = literal_eval(str(x))

        return pd.Series({'from_name' : xd['name'],

                          'from_id' : int(xd['id'])})

df_from = posts['from'].apply(expand_from)

posts_exp = pd.concat([posts, df_from], axis=1)

posts_exp = posts_exp.drop(columns=['from'])

posts_exp
# select posts of 'good' facebook pages

len_before = len(posts_exp)

df_pages = pd.read_csv('fb_data_processed/page_info_'+country+'.csv', sep='|')

posts_good = posts_exp[posts_exp['from_id'].isin(df_pages['id'].values.tolist())]

len_after = len(posts_good)

print('good pages:', str(df_pages.id.nunique()))

print('discarded', str(len_before-len_after), 'posts from bad pages (out of '+str(len_before)+')')

print('unique pages left:', posts_good.from_id.nunique())
# convert field 'shares'

def expand_shares(x):

        if not pd.isna(x):

            xd = literal_eval(str(x))

            return pd.Series({'shares' : xd['count']})

        else:

            return pd.Series({'shares': np.nan})

    

posts_good['shares'] = posts_good['shares'].apply(expand_shares)

# posts_good = posts_good.drop(columns=['coordinates', 0])

posts_good
posts_good.message.values
# save posts

posts_good.to_csv('fb_data_processed/page_posts_'+country+'.csv', sep='|')