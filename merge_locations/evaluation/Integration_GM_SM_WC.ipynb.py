import pandas as pd

import googlemaps
client=googlemaps.Client("AIzaSyBnB3duxMNxArGL7wu9nWp9aOtV7lKx1Ao")
test=client.geocode("6 Avenida 1-99 Zona 1 Barrio La Democracia, Jalapa, Jalapa",region='gt')
test
def parse_response(return_json,query_name):

    return {

        "query_name":query_name,

        'formatted_address':return_json['formatted_address'],

        "lat":return_json['geometry']['location']['lat'],

        "lng":return_json['geometry']['location']['lng'],

    }

    
parse_response(test[0],"6 Avenida 1-99 Zona 1 Barrio La Democracia")
GT_se=pd.read_csv("gt_se.csv",header=None)



gt_list=GT_se[1]
result_list_gt_full_name=[]

for address in gt_list.to_list():

    result_query=client.geocode(address,region='gt')

    print(result_query)

    if len(result_query)<1:

        continue

    result_list_gt_full_name.append(parse_response(result_query[0],address))
gt_result_full_name=pd.DataFrame(result_list_gt_full_name)
gt_result_full_name.to_csv("gt_result_full_name.csv")
social_media=pd.read_csv("social_media_Guatemala.csv",delimiter="|")
social_media.to_csv("sm_gt.csv")
NL_se=pd.read_csv("nl_se.csv",header=None)
nl_list=NL_se[1]
result_list_nl_full_name=[]

for address in nl_list.to_list():

    result_query=client.geocode(address,region='nl')

    print(result_query)

    if len(result_query)<1:

        continue

    result_list_nl_full_name.append(parse_response(result_query[0],address))
nl_result_full_name=pd.DataFrame(result_list_nl_full_name)
nl_result_full_name.to_csv("nl_result_full_name.csv")
social_media_nl=pd.read_csv("social_media_Netherlands.csv",delimiter="|")
social_media_nl.to_csv("sm_nl.csv")
import requests

import lxml

from lxml import etree as et
text=requests.get("https://www.rodekruis.nl/wp-content/plugins/superstorefinder-wp/ssf-wp-xml.php")
address_et=et.fromstring (text.text)
address_et.find("locator")
address_list_nl=[]

for child in address_et[5]:

    #print(child[:])

    #print(child[3].text,child[4].text)

    address_list_nl.append({"name":child[0].text,"address":child[1].text,"lat":child[3].text,'lon':child[4].text})
pd.DataFrame(address_list_nl).to_csv('correct_nl.csv')
pd.DataFrame(address_list_nl)