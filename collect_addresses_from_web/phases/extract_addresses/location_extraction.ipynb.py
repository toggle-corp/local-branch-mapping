from lxml import etree

from bs4 import BeautifulSoup

from polyglot.text import Text

import re

import pandas as pd

import googlemaps

from geopy import distance as dis

import itertools

from sklearn.feature_extraction.text import CountVectorizer

import requests

import pickle

pd.options.display.max_rows = 4000

import numpy as np
def calculate_distance(p1,p2):

    return dis.distance(p1,p2).km
def parse_response(return_json,query_name):

    return {

        "query_name":query_name,

        'formatted_address':return_json['formatted_address'],

        "lat":return_json['geometry']['location']['lat'],

        "lng":return_json['geometry']['location']['lng'],

    }
def n_gram(string):

    vectorizer = CountVectorizer(ngram_range=(1, 100),token_pattern = r"(?u)\b\w+\b")

    X = vectorizer.fit_transform([string])

    analyze = vectorizer.build_analyzer()

    return vectorizer.get_feature_names()
def query_from_geonames(candidate):

    resp=requests.get("https://www.geonames.org/search.html?q="+candidate+"&country=GT").text

    return "records found for" in resp or "record found for" in resp
def validate_place_name(string):

    n_grams=n_gram(string)

    return sum([query_from_geonames(i) for i in n_grams])
def walker(soup):

    result_list=[]

    if soup.name is not None:

        for child in soup.children:

            #process node

            #print(str(child.string))

#             if None != child.string:

#                 print()

#                 t=Text(str(child.extact()))

#                 print(t.pos_tags)

            result_list.append(str(child.string))

            result_list+=walker(child)

    return result_list
with open('../data/NL_html','r',encoding='UTF-8') as f:

    soup=BeautifulSoup(f.read())

    for script in soup(["script", "style"]):

        script.decompose()

    #print(soup)

    candidate_list=list(set(walker(soup)))
after_filter_nl=[]

name_candidate_nl=[]

for i in candidate_list:

    #print(i)

    print(re.match(r"^(?=.*[A-Za-z])(?=.*,).*$",i.replace('\n',''),re.M))

    i=" ".join(i.split())

    if re.match(r"^(?=.*[A-Za-z])(?=.*,).*$",i.replace('\n',''),re.M) and len(i)<100:

        after_filter_nl+=[i.replace('\n','')]

    elif re.match(r"^(?=.*[A-Za-z]).*$",i.replace('\n',''),re.M) and len(i)<50 and len(i)>3:

        name_candidate_nl.append(i)
after_filter_nl
with open('../data/GT_html','r',encoding='UTF-8') as f:

    soup=BeautifulSoup(f.read())

    for script in soup(["script", "style"]):

        script.decompose()

    #print(soup)

    candidate_list_gt=list(set(walker(soup)))
after_filter_gt=[]

name_candidate_gt=[]

for i in candidate_list_gt:

    #print(i)

    #print(re.match(r"^(?=.*[A-Za-z])(?=.*,).*$",i.replace('\n',''),re.M))

    i=" ".join(i.split())

    #print(i)

    if re.match(r"^(?=.*[A-Za-z])(?=.*,).*$",i.replace('\n',''),re.M) and len(i)<100:

        #print(i)

        after_filter_gt+=[i.replace('\n','')]

    elif re.match(r"^(?=.*[A-Za-z]).*$",i.replace('\n',''),re.M) and len(i)<50 and len(i)>3:

        name_candidate_gt.append(i)
len([i for i in name_candidate_gt if 'Delegación' in i])
client=googlemaps.Client("AIzaSyBnB3duxMNxArGL7wu9nWp9aOtV7lKx1Ao")
client.geocode('Delegación Tecún Umán',region='gt')
client.geocode('1 Avenida 4-38 Zona 1, Tecún Umán, San Marcos ',region='gt')
def query_geocode(name,region):

    response=client.geocode(name,region=region) 

    if len(response)>0:

        return (name,response)
addresses_list_gt_google=[query_geocode(i,region='gt')  for i in after_filter_gt]
name_list_gt_google=[query_geocode(i,region='gt')  for i in name_candidate_gt]
import pickle


with open('../data/name_list_gt_google.pickle', 'wb') as f:

    # Pickle the 'data' dictionary using the highest protocol available.

    pickle.dump(name_list_gt_google, f, pickle.HIGHEST_PROTOCOL)

with open('../data/addresses_list_gt_google.pickle', 'wb') as f:

    # Pickle the 'data' dictionary using the highest protocol available.

    pickle.dump(addresses_list_gt_google, f, pickle.HIGHEST_PROTOCOL)
name_list_gt_google=pickle.load(open('../data/name_list_gt_google.pickle','rb'))

addresses_list_gt_google=pickle.load(open("../data/addresses_list_gt_google.pickle",'rb'))
addresses_list_gt=[i for i in addresses_list_gt_google if i and validate_place_name(i[0]) and not re.search("[\U00010000-\U0010ffff]",i[0],flags=re.UNICODE)]

name_list_gt=[i for i in name_list_gt_google if i and validate_place_name(i[0]) and not re.search("[\U00010000-\U0010ffff]",i[0],flags=re.UNICODE) ]
for i in name_list_gt:

    print(i)
[(i[0],len(i[1])) for i in addresses_list_gt]
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))

vectorizer.fit([name[0] for name in name_list_gt])
def plot(string):

    df=pd.DataFrame([vectorizer.get_feature_names(),vectorizer.transform([string]).todense()[0].tolist()[0]]).T.sort_values(1,ascending=False)

    df=df[df[1]==df[1].max()]

    df.columns=['topic','score']

    df['name']=string

    return df[0:1]

pd.concat(list(map(lambda x: plot(x),[name[0] for name in name_list_gt])))
from fuzzywuzzy import fuzz

import collections

import math



def weight_name(name_a,name_b):

    df=pd.DataFrame([vectorizer.get_feature_names(),vectorizer.transform([name_b]).todense()[0].tolist()[0]]).T.sort_values(1,ascending=False)

    df=df[df[1]==df[1].max()]

    df.columns=['topic','score']

    #print(df['topic'].tolist())

    dup_ind=1/math.exp(sum([name_a.lower().count(i)*len(i.split(' ')) for i in df['topic'].tolist()]))

    similar_ind=1/(1+fuzz.ratio(name_a,name_b))

    print(dup_ind,similar_ind)

    return dup_ind*similar_ind
min_distance_list=[]

for addresses in addresses_list_gt:

    address_name=addresses[0]

    for names in name_list_gt:

        department_name=names[0]

        if len(names[1])>3:

            continue

#         if not validate_place_name(department_name):

#             continue

        #print(address_name,department_name)

        min_dis=999999

        #print(min_dis)

        for add_rep in addresses[1]:

            #print(add_rep)

            address_dict=parse_response(add_rep,address_name)

            for name_rep in names[1]:

                

                name_dict=parse_response(name_rep,department_name)

                print(address_dict,name_dict)

                dist=calculate_distance((address_dict['lat'],address_dict['lng']),(name_dict['lat'],name_dict['lng']))

                #print(dist)

                if dist<min_dis:

                    min_dis=dist*weight_name(address_name,department_name)

        min_distance_list.append({"web_lat":address_dict['lat'],"web_lng":address_dict['lng'],"address":address_name,"name":department_name,"min_dist":min_dis,"weight":weight_name(address_name,department_name)})
min_dist_df_gt=pd.DataFrame(min_distance_list)
matched_list=[]

for row in min_dist_df_gt.sort_values("min_dist",ascending=True).iterrows():

    row_=row[1]

    flag=0

    if len([i for i in matched_list if row_['name'] in i['name']]):

        continue

    for previous in matched_list:

#         if row_['name'] in previous['name']:

#                 continue

        if previous['address']==row_['address']:

            

            flag=1

            if previous['min_dist']==row_["min_dist"]:

                previous['name'].add(row_['name'])

    if flag==0:

        tmp_dict=row_.to_dict()

        tmp_dict['name']=set([tmp_dict['name']])

        print(tmp_dict)

        matched_list.append(tmp_dict)
pd.DataFrame(matched_list)
pd.set_option('display.max_colwidth',100)
min_dist_df_gt.sort_values(['name','min_dist'],ascending=True).groupby('name').head(1)
[i[0] for i in addresses_list_gt]
from elasticsearch import Elasticsearch
es=Elasticsearch()

filtered_candidate_gt=[]

for candidate in name_candidate_gt:

    try:

        result=es.search(index='gt',body={

              "query": {

               "query_string": {

              "query": candidate,

              "fields": [

              "FULL_NAME_RO"

              ]}}})

        if len(result)>0:

            condidates=[condidate['_source']['FULL_NAME_RO']for condidate in result['hits']['hits']]

            print(candidate)

            print(condidates)

            if len(condidates) >0:

                filtered_candidate_gt.append(candidate)

    except:

        pass
filtered_candidate_gt
import requests

import lxml

from lxml import etree as et
text=requests.get("https://www.rodekruis.nl/wp-content/plugins/superstorefinder-wp/ssf-wp-xml.php")
address_et=et.fromstring (text.text)
address_et.find("locator")
address_et.tag
address_et
address_list_nl=[]

for child in address_et[5]:

    address_list_nl.append(child[1].text)
address_list_nl=list(set(address_list_nl))
len(address_list_nl)
count=0

for address in address_list_nl:

    address=' '.join(address.split())

    if sum([1 for i in after_filter_nl if address in i]):

        count+=1

    else:

        print(address)
count
nl_se=pd.Series(after_filter_nl)
nl_se.to_csv("../data/nl_se.csv")
gt_se=pd.Series(after_filter_gt)
gt_se.to_csv("../data/gt_se.csv")
QI2qcmn0