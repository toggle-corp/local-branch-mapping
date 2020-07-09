import pandas as pd

from geopy import distance as dis

import itertools

import numpy as np

import re

import fuzzywuzzy

from fuzzywuzzy import process

from fuzzywuzzy import fuzz
def calculate_distance(p1,p2):

    return dis.distance(p1,p2).km
sm_gt=pd.read_csv("../data/sm_gt.csv")

all_infor_gt=pd.read_csv('../data/page_info_Guatemala.csv',delimiter='|',index_col=0)

infor_gt=pd.read_csv('../data/pages_Guatemala_validate.csv',delimiter='|',index_col=0)

matched_df_gt=pd.read_csv('../data/address_name_match_gt.csv')
infor_gt=infor_gt[['fb_id','latitude','longitude']].merge(all_infor_gt,left_on='fb_id',how='left',right_on="id")
name_list_gt=list(map(lambda x: x[1]['username'] if type(x[1]['username'])==str else x[1]['name'],infor_gt.iterrows()))
name_list_gt_space=list(map(lambda x :re.sub("([a-z])([A-Z])","\g<1> \g<2>",x),name_list_gt))
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
vectorizer.fit(name_list_gt_space)
def top1(string):

    df=pd.DataFrame([vectorizer.get_feature_names(),vectorizer.transform([string]).todense()[0].tolist()[0]]).T.sort_values(1,ascending=False)

    df=df[df[1]==df[1].max()]

    df.columns=['topic','score']

    df['string']=string

    tmp_list=df.reset_index(drop=True).values.tolist()

    return [' '.join(df['topic']),df['score'].max(),set(df['string']).pop()]

top1('Cruz Roja GTSanto Tomas Castilla')
topic_sm_df_gt=pd.DataFrame(list(map(top1,name_list_gt_space)))
topic_sm_df_gt.columns=['topic','score','name_sm']
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

vectorizer.fit(matched_df_gt['name'])
topic_match_df_gt=pd.DataFrame(list(map(top1,matched_df_gt['name'])))
topic_match_df_gt.columns=['topic','score','name_match']
def fuzzy_match(string):

    match=process.extract(string,topic_match_df_gt['topic'],limit=1,scorer=fuzz.token_sort_ratio)[0]

    return [string,match[0]]
middle_match_df_gt=pd.DataFrame(list(map(fuzzy_match,topic_sm_df_gt['topic'])))
middle_match_df_gt.columns=['sm_topic','name_topic']
tmp_df_gt=topic_sm_df_gt.merge(middle_match_df_gt,left_on='topic',how='left',right_on='sm_topic')[["name_sm","sm_topic","name_topic"]].merge(topic_match_df_gt,how='left',left_on='name_topic',right_on='topic')[['name_sm','name_match']].merge(matched_df_gt,how='left',left_on="name_match",right_on='name').drop(columns=['Unnamed: 0','min_dist','name'])
merged_gt=pd.concat([infor_gt,tmp_df_gt],axis=1)[['fb_id', 'latitude', 'longitude','name', 'phone', 'username', 'website', 'single_line_address','name_sm',

       'name_match', 'address', 'web_lat', 'web_lng', 'weight']]
merged_gt['distance_page_web']=merged_gt.apply(lambda x:calculate_distance((x['latitude'],x['longitude']),(x['web_lat'],x['web_lng'])),axis=1)
Gmaps_gt_df=pd.read_csv('../data/GMaps_Guatemala.csv',index_col=0)
vectorizer = TfidfVectorizer(ngram_range=(1, 1))

vectorizer.fit(Gmaps_gt_df['place_name'])
topic_Gmap_df_gt=pd.DataFrame(list(map(top1,Gmaps_gt_df['place_name'])))
topic_Gmap_df_gt.columns=['topic','score','name_match']
def fuzzy_match(string):

    match=process.extract(string,topic_Gmap_df_gt['topic'],limit=1,scorer=fuzz.token_sort_ratio)[0]

    return [string,match[0]]
middle_match_df_gt=pd.DataFrame(list(map(fuzzy_match,topic_sm_df_gt['topic'])))
middle_match_df_gt.columns=['sm_topic','Gmaps_topic']
tmp_df_gt=topic_sm_df_gt.merge(middle_match_df_gt,left_on='topic',how='left',right_on='sm_topic')[["name_sm","sm_topic","Gmaps_topic"]].merge(topic_Gmap_df_gt,how='left',left_on='Gmaps_topic',right_on='topic')[['name_sm','name_match']].merge(Gmaps_gt_df,how='left',left_on="name_match",right_on='place_name').drop(columns=['place_id','name_match'])[['name_sm','place_name','place_address','place_lat','place_long']].drop_duplicates().reset_index(drop=True)
tmp_df_gt.columns=['name_sm', 'Gmaps_place_name', 'Gmaps_place_address', 'Gmap_lat', 'Gmap_long']
merged_gt=merged_gt.merge(tmp_df_gt,left_on='name_sm',how='left',right_on='name_sm')
merged_gt['distance_page_Gmaps']=merged_gt.apply(lambda x:calculate_distance((x['latitude'],x['longitude']),(x['Gmap_lat'],x['Gmap_long'])),axis=1)
merged_gt.columns=['fb_id', 'fb_latitude', 'fb_longitude', 'name', 'phone', 'username',

       'website', 'single_line_address', 'name_sm', 'name_match', 'web_address',

       'web_lat', 'web_lng', 'weight', 'distance_page_web', 'Gmaps_place_name',

       'Gmaps_place_address', 'Gmap_lat', 'Gmap_long', 'distance_page_Gmaps']
merged_gt.to_csv("../data/validation_set_gt.csv")