import pandas as pd
df = pd.DataFrame(index = pd.MultiIndex(levels=[[],[]],

                                        codes=[[],[]],

                                        names=[u'country', u'page']),

                  columns=['screen_name', 'location', 'location_more', 'description', 'url'])
df
df.loc[('Guatemala', 'CruzRojaGT')] = ['','','','','']
df_meta = pd.read_excel('pilot_countries_metadata.xlsx', index_col=0)
df_meta['red cross name (local language)']
df_ids = pd.read_csv('twitter_app/twitter_ids_pilot.csv', sep='|', index_col=0)

df_ids.columns
def is_name_in_it(row):

    ns_eng_norm = row['national society (english)'].lower().replace(" ", "")

    ns_loc_norm = row['national society (local language)'].lower().replace(" ", "")

    target_norm = row['red cross name (local language)'].lower().replace(" ", "")

    return (ns_eng_norm in target_norm or ns_loc_norm in target_norm)
df_ids['is_good'] = df_ids.apply(is_name_in_it, axis=1)

df_ids = df_ids[df_ids['is_good']].drop(columns=['is_good'])
df_ids.index = df_ids.country

df_ids = df_ids.drop(columns=['country'])
df_ids.index.unique().values