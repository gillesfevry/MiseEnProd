#This module is used to extract data from TMDB using it's API

import requests
import pandas as pd
from tqdm import tqdm
import time

#headers linked to Gilles' account needed to use the API
headers = {"accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzNjcwNTY1MTA0YzUzY2Q0MjE3N2M3ZWQyMmVmZDk1ZCIsIm5iZiI6MTczMjI5MzAwNy4yMTk0MDM1LCJzdWIiOiI2NzQwYWY5OWRkNTNmYTAwNjAzMzM3YzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.8DPhrUjEakZO4CxeovXkNmSOe01nH3r5kEZlntq547M"
            }

def get_movie_ids_list(nb_pages=1, headers=None):
    """
        Creates a list of the ids of the most rated movies using TMDB API discover function. 

        Parameters:
        nb_pages (int >=1): The number of pages in the Discover option we want to get data from. A page lists 20 movies.
        headers (dictionnary): The headers needed to use the TMDB API. 

        Returns:
        list: list of movie ids
    """
    assert headers is not None, "Headers is not None"
    assert isinstance(nb_pages, int) and nb_pages >= 1, "nb_pages is not an int >=1"

    #initializing of the fixed parts of the url and of the ids list
    url_start="https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=fr-FR&page="
    url_end="&sort_by=vote_count.desc&with_original_language=fr"
    ids=[]

    print("getting movie ids")

    #rethrieving movie ids
    for i in tqdm(range(nb_pages)):
        url=url_start+str(i+1)+url_end
        req=requests.get(url, headers=headers).json()
        time.sleep(0.5) # to prevent overloading
        ids.extend(movie["id"] for movie in req["results"])
    
    return(ids)

def get_movies_info(ids=[], headers=None):
    """
        Creates an uncleaned data frame with all the informations of the movies of which the ids were given.

        Parameters:
        ids (list of ints): A list of the TMDB ids of the movies we want to get information from.
        headers (dictionnary): The headers needed to use the TMDB API. 

        Returns:
        A Pandas data frame
    """
    assert headers is not None, "Headers is not None"
    assert len(ids)>=1, "List is empty"

    #initializing of the fixed parts of the url and of the dataframes list
    url_start= "https://api.themoviedb.org/3/movie/"
    url_end="?language=en-US"
    dataframes=[]

    print("getting movie info")

    #rethrieving information and stocking it in a list of dataframes
    for i in tqdm(ids):
        url=url_start+str(i)+url_end
        req = requests.get(url, headers=headers).json()
        time.sleep(0.1) #si trop lent, réduire
        dataframes.append(pd.json_normalize(req)) #pas très efficace voir si on peut améliorer
    
    df = pd.concat(dataframes, ignore_index=True)
    df=df[df["status_message"].isna()]

    return(df)



def drop_useless_info(df=None):
    assert df is not None, "No data frame was given"

    # Liste des colonnes à supprimer
    columns_to_drop = [
        "adult", "backdrop_path", "belongs_to_collection", "homepage", "imdb_id", 
        "origin_country", "original_language", "original_title", "production_companies", 
        "production_countries", "spoken_languages", "status", "tagline", "video", 
        "belongs_to_collection.id", "belongs_to_collection.name", 
        "belongs_to_collection.poster_path", "belongs_to_collection.backdrop_path","success","status_code", "status_message"

    ]

    # Vérifier quelles colonnes sont présentes avant de les supprimer
    columns_in_df = [col for col in columns_to_drop if col in df.columns]

    # Supprimer seulement les colonnes présentes
    df1 = df.drop(columns=columns_in_df)
    
    return df1

def keep_main_genre(df=None):

    df['main_genre_id'] = df['genres'].map(lambda x: x[0]['id'])
    df['main_genre_name'] = df['genres'].map(lambda x: x[0]['name'])
    df1=df.drop(columns=["genres"])
    return(df1)

def full_poster_path(df=None):

    df['full_poster_path']= "https://image.tmdb.org/t/p/original/"+df['poster_path']
    df1=df.drop(columns=["poster_path"])
    return(df1)