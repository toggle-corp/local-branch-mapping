import pandas as pd

import fuzzywuzzy

from fuzzywuzzy import process

import os

import time
pilot_list_df=pd.read_csv('pilot_info.csv',header=0)
pilot_list_df
corpus_mapping_df=pd.read_csv("pilot_scarping/corpus/mapping",header=None)
corpus_mapping_df[0].map(lambda x: [x[:x.find("http")],x[x.find("http"):]]).to_list()
corpus_mapping_df=pd.DataFrame(corpus_mapping_df[0].map(lambda x: [x[:x.find("http")],x[x.find("http"):]]).to_list())
corpus_mapping_df.columns=['id','url']
def read_zipfile(dir_name,file_name):

    df_list=[]

    county_code=file_name

    path=dir_name+file_name

    for file in [file for file in os.listdir(path) if 'administrative' in file or "populatedplaces" in file or len(file)<7]:

        df_list.append(pd.read_csv(path+"/"+file,delimiter='\\t'))

    concected_df=pd.concat(df_list,axis=0)

    concected_df['code']=county_code

    concected_df.drop(columns=[i for i in concected_df.columns if len(i)>20],inplace =True)

    #print(concected_df)

    concected_df.fillna('',inplace=True)

    concected_df=concected_df[['ADM1', 'CC1', 'CC2', 'DMS_LAT', 'DMS_LONG', 'DSG', 'ELEV',

       'FC', 'FULL_NAME_ND_RG', 'FULL_NAME_ND_RO', 'FULL_NAME_RG',

       'FULL_NAME_RO','LC','SORT_NAME_RO']]

    #print(concected_df.columns)

    return concected_df['FULL_NAME_RO'].to_list()+concected_df['SORT_NAME_RO'].to_list()
location_dir="../locations/geonames_20190701_CntryFiles/"

def fuzzy_match_by_nation(inital_site,code,mapping_list):

    filtered_pages=mapping_list[mapping_list.apply(lambda x:True if inital_site in x.url else False,axis=1)]

    name_list=read_zipfile(location_dir,code.lower())

    count=0

    result_list=[]

    #print(filtered_pages)

    #print(name_list)

    for row in filtered_pages.iterrows():

        #print(row[1])

        start=time.time()

        row=row[1]

#         if count==1:

#             break

        with open("pilot_scarping/corpus/"+str(row['id']),'r',encoding='utf-8') as f:

            #print(f.read())

#             for line in f.readlines():

#                 result=process.extract(line,name_list,limit=10)

#                 print(result)

            text= f.read().lower()

            #print("Lilongwe".lower() in text)

            name_list=pd.Series(name_list)

            #print(name_list.apply)

            #print(name_list.apply(lambda x: x if x.lower() in text else None).dropna())

            matched_list=name_list.apply(lambda x: x if x.lower() in text else None).dropna()

            result_list.append((row['url'],matched_list.shape[0]))

        count+=1

        end=time.time()

        #print("It takes "+str(end-start))

    result_df=pd.DataFrame(result_list)

    print(result_df.sort_values(by=[1],ascending=False))

    return result_df.sort_values(by=[1],ascending=False)[0].to_list()[0]
result_list=[]

for row in pilot_list_df.iterrows():

    country=row[1]

    url=country['url']

    best_url=fuzzy_match_by_nation(url[url.find("www"):].replace("/",'/'),country['code'],corpus_mapping_df)

    result_list.append((country['code'],best_url))
fuzzy_match_by_nation("www.redcross.mw","MI",corpus_mapping_df)
fuzzy_match_by_nation("https://www.rodekruis.nl","NL",corpus_mapping_df)
fuzzy_match_by_nation("www.cruzroja.gt","GT",corpus_mapping_df)
fuzzy_match_by_nation("www.redcross.org.lb","LE",corpus_mapping_df)


corpus_mapping_df[corpus_mapping_df.apply(lambda x:True if "www.redcross.mw" in x.url else False,axis=1)]

#"https://www.redcross.mw/" in corpus_mapping_df.url
for i in corpus_mapping_df['url']:

    print("https://www.redcross.mw/" in i)
test=pd.Series([1,2,3,5])
test.apply(lambda x: x if x in [1,5] else None).dropna().shape
result_list