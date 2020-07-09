# load libraries and define fields

import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import fileinput

import json

from tqdm import tqdm

country = 'Netherlands'

page_fields = ['id',

               'name',

               'username',

               'about',

               'description',

               'mission',

               'general_info',

               'personal_info',

               'affiliation',

               'bio',

               'birthday',

               'founded',

               'hometown',

               'hours',

               'contact_address',

               'single_line_address',

               'current_location',

               'store_location_descriptor',

               'website',

               'emails',

               'phone',

               'whatsapp_number',

               'category',

               'category_list',

               'company_overview',

               'country_page_likes',

               'is_permanently_closed',

               'engagement',

               'privacy_info_url',

               'fan_count',

               'link']
# load facebook pages



data = []

df = pd.DataFrame()

df_retweeted = pd.DataFrame()

for line in tqdm(fileinput.FileInput("fb_data/page_info_"+country+".jsonl")):

    try:

        fb_page = json.loads(line)

        

        df = df.append(fb_page,

                        ignore_index=True)

        

    except (json.JSONDecodeError):

        print('error')

        pass

    

# drop duplicates

df.drop_duplicates(subset ="id", keep = 'last', inplace = True) 

    

# inspect

print(df.info())

print(df.head())
# inspect in detail

df
# ad-hoc filter to remove garbage

# df_clean = df[:2]

df_clean = df[df.name.str.contains('Rode Kruis')]

df_clean = df_clean[~df.name.str.contains('Kruisstraat')]

df_clean = df_clean[~df.name.str.contains('Kruislaan')]

df_clean = df_clean[~df.name.str.contains('Kruisgebouw')]

df_clean
# save list of 'good' facebook pages

df_clean.to_csv('fb_data_processed/page_info_'+country+'.csv', sep=',')