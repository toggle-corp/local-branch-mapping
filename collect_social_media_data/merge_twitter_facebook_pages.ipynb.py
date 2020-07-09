import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import jsonlines

import googlemaps

from fuzzywuzzy import process

from fuzzywuzzy import fuzz

import numpy as np

from ast import literal_eval

import geopy

from geopy.geocoders import Nominatim

from geopy.extra.rate_limiter import RateLimiter

import time



country = 'Malawi'

gmaps = googlemaps.Client(key='AIzaSyANzA_Tm91rRCo9O-dTTE3OLQS9tswA_ic')
# load fb posts and add user screen name

df_fb = pd.read_csv('fb_data_processed/page_posts_'+country+'.csv', sep='|', index_col=0)



df_fb
df_tw = pd.read_csv('twitter_data_processed/tweets_'+country+'.csv', sep=',', index_col=0)

df_tw
# first, identify pages and match

df_page_merged = pd.DataFrame()



# facebook page data

# N.B. loading the pre-processed ones, since posts are cleaned already!!!

df_fb_pages = pd.DataFrame()

with jsonlines.open('fb_data/page_info_'+country+'.jsonl') as reader:

    for obj in reader:

        df_fb_pages = df_fb_pages.append(obj, ignore_index=True)

        

# twitter page data

tw_users = df_tw.screen_name.unique()

# filter only offical pages

df_tw_pages = pd.read_csv('social_media_scraper/pilot_countries_twitter_ids.csv', sep='|', index_col=0)

tw_users_names = df_tw_pages['screen name'].to_list()

tw_users = list(filter(lambda x: x in tw_users_names, tw_users))

df_tw = df_tw[df_tw.screen_name.isin(tw_users)]

# df_fb.to_csv('social_media_data_merged/facebook_'+country+'.csv', sep='|')

# df_tw.to_csv('social_media_data_merged/twitter_'+country+'.csv', sep='|')
df_fb_pages
df_tw_pages = df_tw_pages[df_tw_pages['country']==country]

df_tw_pages
df_page_merged = pd.DataFrame()



# match top by activity

top_user_tw = df_tw.screen_name.value_counts().argmax()

info_top_user_tw = df_tw_pages[df_tw_pages["screen name"]==top_user_tw]

top_user_fb = df_fb.from_id.value_counts().argmax()

info_top_user_fb = df_fb_pages[df_fb_pages.id==str(top_user_fb)]



info_top_user_fb.location.to_dict()



# if location not present, search location of national society (from website)

if "city" not in info_top_user_fb.location.to_dict().keys():

    print('location not present in fb, going to geolocate what we got from website')

    df_contacts_ns = pd.read_csv('ifrc_scraper/contacts_clean.csv', sep=',', index_col=0)

    contact = df_contacts_ns[df_contacts_ns.name == info_top_user_tw["national society (english)"].values[0]]

    address = contact.address.values[0]

    coordinates = gmaps.geocode(address)[0]["geometry"]["location"]

    info_top_user_fb.at[0, 'location'] = {'city': '',

                                        'country': country,

                                        'latitude': coordinates["lat"],

                                        'longitude': coordinates["lng"],

                                        'street': address,

                                        'zip': ''}

    

df_page_merged = df_page_merged.append({'type': 'main',

                                       'tw_name': info_top_user_tw.name.values[0],

                                       'tw_id': int(info_top_user_tw.id.values[0]),

                                       'fb_name': info_top_user_fb.name.values[0],

                                       'fb_id': int(info_top_user_fb.id.values[0]),

                                       'location': info_top_user_fb.location.values[0]}, ignore_index=True)

df_page_merged
# match rest by location mentions



tw_users = list(df_tw.screen_name.unique())

tw_users.remove(top_user_tw)

fb_users_ids = list(df_fb.from_id.unique())

fb_users_ids.remove(top_user_fb)

fb_users_ids = dict(zip(fb_users_ids, [True for x in fb_users_ids]))



print(tw_users)



# for user_tw in tw_users:

#     for user_fb, free in fb_users_ids.items():

#         if free == False:

#             continue

        

#         info_user_tw = df_tw_pages[df_tw_pages["screen name"]==user_tw].squeeze()

#         info_user_fb = df_fb_pages[df_fb_pages.id.astype(int)==int(user_fb)].squeeze()

    

#         # (1) if locations are mentioned, check against fb

#         if (info_user_tw.location and any(info_user_fb.location)):

#             if (info_user_tw.location in info_user_fb.location['city'] or \

#                 info_user_fb.location['city'] in info_user_tw.location):

#                 df_page_merged = df_page_merged.append({'type': 'branch',

#                            'tw_name': info_user_tw['name'],

#                            'tw_id': info_user_tw.id,

#                            'fb_name': info_user_fb['name'],

#                            'fb_id': info_user_fb.id,

#                            'location': info_user_fb.location}, ignore_index=True)

#                 fb_users_ids[user_fb] = False

#                 break

#         # (2) otherwise, do fuzzy string matching on name and location

#         fuzzNameScore = process.extract(info_user_tw['name'].replace(" ",""), [info_user_fb['name'].replace(" ","")], scorer=fuzz.token_set_ratio)[0][1]

#         fuzzLocationScore = process.extract(info_user_tw.location, [info_user_fb.location['city']], scorer=fuzz.token_set_ratio)[0][1]

#         if (fuzzNameScore > 90 or fuzzLocationScore>90):

#             df_page_merged = df_page_merged.append({'type': 'branch',

#                        'tw_name': info_user_tw['name'],

#                        'tw_id': info_user_tw.id,

#                        'fb_name': info_user_fb['name'],

#                        'fb_id': info_user_fb.id,

#                        'location': info_user_fb.location}, ignore_index=True)

#             fb_users_ids[user_fb] = False

#             break  



# df_page_merged   
# append facebook pages left

fb_pages_found = list(df_page_merged.fb_id.unique())

fb_pages_left = [x for x in list(df_fb.from_id.unique()) if x not in fb_pages_found]

for fb_page in fb_pages_left:

    info_user_fb = df_fb_pages[df_fb_pages.id.astype(int)==int(fb_page)].squeeze()

    df_page_merged = df_page_merged.append({'type': 'branch',

                           'fb_name': info_user_fb['name'],

                           'fb_id': info_user_fb.id,

                           'location': info_user_fb.location}, ignore_index=True)

df_page_merged 
df_page_merged.fb_id = df_page_merged.fb_id.astype(int)

# expand field 'location'

def expand_location(x):

    if x:

        try:

            xd = literal_eval(str(x))

            one_line_location = ""

            for key in ['street', 'zip', 'city', 'country']:

                if key in xd.keys():

                    one_line_location += xd[key]+' '

            lat = 0

            lon = 0

            if 'latitude' in xd.keys():

                lat = xd['latitude']

            if 'longitude' in xd.keys():

                lon = xd['longitude']

            return pd.Series({'address' : one_line_location,

                              'latitude' : lat,

                              'longitude' : lon})

        except:

            return pd.Series({'address' : np.NaN,

                              'latitude' : np.NaN,

                              'longitude' : np.NaN}) 

df_page_final = df_page_merged['location'].apply(expand_location)

df_page_final = pd.concat([df_page_merged, df_page_final], axis=1)

df_page_final = df_page_final.drop(columns=['location'])

df_page_final
# correct for random or empty coordinates

# first check if coordinates are in country

def which_country(row):

    if pd.isna(row['latitude']):

        print('no info available')

        print('geolocating', row['fb_name'])

        # country different than expected, geolocate address

        coordinates = gmaps.geocode(row['fb_name'])[0]["geometry"]["location"]

        print('--> ', coordinates)

        return coordinates["lat"], coordinates["lng"]

    else:

        latitude = row['latitude']

        longitude = row['longitude']

        ## finds a country given latitude and longitude

        global geoloc_dict

        geolocator = Nominatim(user_agent="my_app")

        place = geolocator.reverse((str(latitude)+', '+str(longitude)), timeout = 300).raw ## returns a nearby address for a point

        country_found = 'Unknown'

        if not 'error' in place:

            if 'country_code' in place['address']:

                country_code = place['address']['country_code'] 

                country_found = codes[codes['Code']==country_code]['Name'].values[0]  # uses codes distionary to find full country name

        time.sleep(1) # geopy does not really like when you do more than 1 request per second

        if country_found != country:

            print('expected', country, ', found', country_found)

            print('geolocating', row['address'])

            # country different than expected, geolocate address

            coordinates = gmaps.geocode(row['address'])[0]["geometry"]["location"]

            print('--> ', coordinates)

            return coordinates["lat"], coordinates["lng"]

        else:

            print('all good')

            return latitude, longitude

    

#dictionary with codes for each country

codes = pd.read_csv('codes.csv', header = 0)

codes['Code'] = codes['Code'].str.lower()



df_page_final['latitude'], df_page_final['longitude'] = zip(*df_page_final.apply(which_country, axis=1))
# inspect final result

df_page_final
# remove something if needed

# df_page_final = df_page_final[:1]

# df_page_final
# save final result

df_page_final.to_csv('social_media_data_merged/pages_'+country+'.csv', sep='|')
gmaps.geocode('Barrio el porvenir San Benito Petén, Guatemala')