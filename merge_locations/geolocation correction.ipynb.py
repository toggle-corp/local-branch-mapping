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
merged_gt=pd.read_csv("../data/validation_set_gt.csv",index_col=0)
def weight_address(stringA,stringB):

    if pd.isna(stringA) or pd.isna(stringB):

        return 1

    else:

        return 1/fuzz.token_set_ratio(stringA,stringB)
def decision_making(x):

    f2g=(calculate_distance((x["fb_latitude"],x["fb_longitude"]),(x["Gmap_lat"],x['Gmap_long']))*weight_address(x['single_line_address'],x['Gmaps_place_address']),x["Gmaps_place_address"],x["Gmap_lat"],x['Gmap_long'])

    w2g=(calculate_distance((x["web_lat"],x["web_lng"]),(x["Gmap_lat"],x['Gmap_long']))*weight_address(x['web_address'],x['Gmaps_place_address']),x['Gmaps_place_address'],x["Gmap_lat"],x['Gmap_long'])

    f2w=(calculate_distance((x["fb_latitude"],x["fb_longitude"]),(x["web_lat"],x["web_lng"]))*weight_address(x['single_line_address'],x['web_address']),x['single_line_address'] if pd.notna(x['single_line_address']) else x['web_address'],x["fb_latitude"],x["fb_longitude"]) 

    min_dis=sorted([f2g,w2g,f2w], key=lambda tup: tup[0])[0]

    return min_dis
pd.concat([merged_gt,pd.DataFrame(merged_gt.apply(lambda x:decision_making(x),axis=1).tolist(),columns=["min_dist",'finall_address',"final_lat","final_lng"])],axis=1).to_csv("../data/final_gt.csv")