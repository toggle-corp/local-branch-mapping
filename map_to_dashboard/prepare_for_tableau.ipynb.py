import pandas as pd

country = 'Malawi'



# load dataframe with social media and/or news

df = pd.read_csv('sm_Malawi.csv')

df.head()
# load gazetteer (from http://geonames.nga.mil/gns/html/namefiles.html)

df_loc = pd.read_csv('locations/'+country+'/'+'loc_administrative_a.txt', sep='\t')

df_loc = df_loc.append(pd.read_csv('locations/'+country+'/'+'loc_localities_l.txt', sep='\t'))

df_loc = df_loc.append(pd.read_csv('locations/'+country+'/'+'loc_populatedplaces_p.txt', sep='\t'))



# create a dictionary locations : coordinates

df_loc = df_loc.sort_values(by=['ADM1'])

df_loc = df_loc.drop_duplicates(subset=['FULL_NAME_ND_RO'], keep='first')

loc_dict = dict(zip(df_loc.FULL_NAME_ND_RO, zip(df_loc.LAT, df_loc.LONG)))

loc_dict
import re, string

from tqdm import tqdm

tqdm.pandas(desc="progress")

from tqdm import tqdm_notebook

from fuzzywuzzy import process

from fuzzywuzzy import fuzz



remove_keywords = ['nan', 'Planes de Emergencia', 'Cruz Roja', 'Media Luna Roja', 'Guatemala', 'Argentina', 'Japon', 'Juventud']

remove_keywords.extend(['Malawi', 'Lebanon', 'لبنان', 'Nederland', 'Netherlands'])



# define function to find locations in text

def find_locations(text, loc_dict):

    

    # skip non-string values

    if type(text) != str:

        return []

    else:

        for k in remove_keywords:

            text = re.sub(k, '', text)

        # remove non-capitalized words from text (reduces ambiguities)

        text = " ".join([x for x in text.split(" ") if not x.islower()])

        # skip is final string is too short (20 char)

        if len(text) < 20:

            return []

        # find locations and append them to list

        ratio_loc = process.extract(text, loc_dict.keys(), scorer = fuzz.token_set_ratio)

        new_loc = []

        for l,v in ratio_loc: 

            if v > 95:

                new_loc.append(l)

        new_loc = list({i for sub in new_loc for i in sub.split(' ')})

        new_loc = list(set([i for i in new_loc if i in loc_dict.keys()]))

        return new_loc

    

df['locations'] = df['message'].progress_apply(lambda x: find_locations(x, loc_dict))

df.head()
# load dataframe with keywords for classification

df_cat = pd.read_csv('classification.csv')

df_cat.head()
import re



# define function to classify text based on keywords

def classify_text(text):

    

    if str(text) == 'nan':

        return []

    

    cat_match = []

    for i in range(len(df_cat)):

        row_cat = df_cat.iloc[i]

        keywords = [x for x in row_cat[2:].values if str(x) != 'nan']

        for keyword in keywords:

            if re.search(str(keyword), text, flags=re.IGNORECASE):

                cat_match.append(row_cat[1])

    return list(set(cat_match))



df_class = df.copy()

df_class['category'] = df_class['message'].apply(classify_text)

df_class.head()
from ast import literal_eval



# expand lists of locations and categories (for Tableau)

def expand_lists(df_):

    new_df = pd.DataFrame(columns=df_.columns)

    

    for i in range(len(df_)):

        row_cat = df_.iloc[i]

        for loc in row_cat['locations']: #literal_eval(row_cat['locations'])

            for cat in row_cat['category']:

                new_df = new_df.append({

                            'medium': row_cat[0],

                            'created_time': row_cat[1],

                            'name': row_cat[2],

                            'message': row_cat[3],

                            'locations': loc,

                            'category': cat              

                        }, ignore_index=True)

    return new_df

        

df_for_tableau = expand_lists(df_class)

df_for_tableau.head()
df_for_tableau.to_csv('sm_Malawi_classified.csv', index=False)
###### ADD POLYGONS ADMIN LEVELS



import pandas as pd

country = 'Malawi'

df = pd.read_csv('sm_Malawi_classified.csv')

df.head()
import geopandas as gpd

from shapely.geometry import Point



# load admin level shapefile (from https://data.humdata.org/ or other)

gdf_adm1 = gpd.read_file('shapefiles/'+country+'/adm1.shp')

gdf_adm2 = gpd.read_file('shapefiles/'+country+'/adm2.shp')

gdf_adm2.head()
# loop over points, if within a polygon add polygon's geometry to dataframe

def get_poly(row):

    location = row['locations']

    if location not in loc_dict.keys():

        print('WARNING:', location, 'not found!!!')

        return pd.Series({'adm1_name': 'nan',

                          'adm1_polygon': 'nan',

                          'adm2_name': 'nan',

                          'adm2_polygon': 'nan'})

    else:

        coord = Point((loc_dict[location][1], loc_dict[location][0]))

        for index, row in gdf_adm2.iterrows():

            if row.geometry.contains(coord):

                

                adm1_name = row['NAME_1']

                adm1_polygon = gdf_adm1[gdf_adm1['NAME_1']==adm1_name].geometry

                if (location == adm1_name or 'region' in location.lower()):

                    adm2_name = 'nan'

                    adm2_polygon = 'nan'

                else:

                    adm2_name = str(row['NAME_2']).replace('TA ', '')

                    adm2_polygon = row.geometry

                         

                return pd.Series({'adm1_name': adm1_name,

                                  'adm1_polygon': adm1_polygon,

                                  'adm2_name': adm2_name,

                                  'adm2_polygon': adm2_polygon})

        

df2 = df.copy()

df2 = df2.merge(df2.apply(get_poly, axis=1), left_index=True, right_index=True)

df2
df2.to_csv('sm_Malawi_classified_polygons.csv', index=False)