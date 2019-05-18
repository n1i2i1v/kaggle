import pandas as pd
import numpy as np
import ast


def data_featurizing(data):
    data['has_collection'] = data['belongs_to_collection'].apply(lambda x: 0 if type(x) is float else 1)
    data.drop('belongs_to_collection', axis=1, inplace=True)

    data['budget'] = data['budget'].apply(lambda x: 1 if x == 0 else x)
#    data['budget'] = np.log(data['budget'])**2

    data.loc[data["genres"].notnull(), "genres"] = \
        data.loc[data["genres"].notnull(), "genres"].apply(lambda x: ast.literal_eval(x))
    data["genres"] = data["genres"].apply(lambda x: x[0]['name'] if type(x) is not float else None)
    data = pd.get_dummies(data, columns=['genres'], dummy_na=True)
    if 'genres_TV Movie' in data:
        data.drop('genres_TV Movie', axis=1, inplace=True)

    data['has_homepage'] = data['homepage'].apply(lambda x: 0 if type(x) is float else 1)
    data.drop(['homepage', 'imdb_id'], axis=1, inplace=True)

    data['original_language'] = data['original_language'].apply(lambda x: 1 if x == 'en' else 0)
    data['original_title'] = (data['title'] == data['original_title'])

    data.loc[data["production_companies"].notnull(), "production_companies"] = data.loc[
        data["production_companies"].notnull(), "production_companies"].apply(lambda x: ast.literal_eval(x))
    data["production_companies_num"] = data["production_companies"].apply(
        lambda x: len(x) if type(x) is not float else 0)

    data.loc[data["production_countries"].notnull(), "production_countries"] = data.loc[
        data["production_countries"].notnull(), "production_countries"].apply(lambda x: ast.literal_eval(x))
    data["production_countries_num"] = data["production_countries"].apply(
        lambda x: len(x) if type(x) is not float else 0)

    data['release_date'].fillna('05/01/00', inplace=True)
    data['release_year'] = data['release_date'].apply(lambda x: int(x.split('/')[2]))
    data['release_year'] = data['release_year'].apply(lambda x: x + 2000 if x < 19 else x + 1900)
    data['release_month'] = data['release_date'].apply(lambda x: int(x.split('/')[0]))
    releaseDate = pd.to_datetime(data['release_date'])
    data['release_dayofweek'] = releaseDate.dt.dayofweek

    data.drop(['overview', 'poster_path', 'production_companies', 'production_countries', 'release_date'],
              axis=1, inplace=True)

    data['runtime'].fillna(data['runtime'].mean(), inplace=True)

    data.loc[data["spoken_languages"].notnull(), "spoken_languages"] = data.loc[
        data["spoken_languages"].notnull(), "spoken_languages"].apply(lambda x: ast.literal_eval(x))
    data["spoken_languages_num"] = data["spoken_languages"].apply(lambda x: len(x) if type(x) is not float else 0)
    data.drop(['spoken_languages', 'status'], axis=1, inplace=True)

    data['tagline'] = data['tagline'].apply(lambda x: 1 if type(x) is not float else 0)

    data.drop('title', axis=1, inplace=True)

    data.loc[data["Keywords"].notnull(), "Keywords"] = data.loc[data["Keywords"].notnull(), "Keywords"].apply(
        lambda x: ast.literal_eval(x))
    data["Keywords_num"] = data["Keywords"].apply(lambda x: len(x) if type(x) is not float else 0)
    data.drop('Keywords', axis=1, inplace=True)

    data.drop(['cast', 'crew'], axis=1, inplace=True)

    if 'revenue' in data:
        data['revenue'] = np.log(data['revenue'])

    return data
