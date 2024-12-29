#This module is used to extract data from TMDB using it's API

import requests
import pandas as pd
from tqdm import tqdm
import time
import ast

#headers linked to Gilles' account needed to use the API
headers = {"accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzNjcwNTY1MTA0YzUzY2Q0MjE3N2M3ZWQyMmVmZDk1ZCIsIm5iZiI6MTczMjI5MzAwNy4yMTk0MDM1LCJzdWIiOiI2NzQwYWY5OWRkNTNmYTAwNjAzMzM3YzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.8DPhrUjEakZO4CxeovXkNmSOe01nH3r5kEZlntq547M"
            }

#The genre dictionnary provided by the API.
genre_dictionnary =requests.get('https://api.themoviedb.org/3/genre/movie/list?language=en', headers=headers).json()['genres']

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
    url_start= "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page="
    url_end="&sort_by=vote_count.desc"
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
    return(df)

def get_balanced_movie_list(nb_pages=1, headers=None):
    """
        Creates a list of the ids of movies with near 0 revenue, where each genre is equally represented

        Parameters:
        nb_pages (int >=1): The number of pages in the Discover option we want to get data from. A page lists 20 movies.
        for each genre, nb_pages will be rethrieved 
        headers (dictionnary): The headers needed to use the TMDB API. 

        Returns:
        list: list of movie ids
    """

    assert headers is not None, "Headers is not None"
    assert isinstance(nb_pages, int) and nb_pages >= 1, "nb_pages is not an int >=1"

    #initializing of the fixed parts of the url and of the ids list
    url_start="https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page="
    url_mid="&sort_by=revenue.asc&vote_count.gte=2&with_genres="
    url_end="&without_genres="
    
    genre_list= [genre['id'] for genre in genre_dictionnary]
    ids=[]

    print("getting movie ids, 19 items (=genres) to get")

    #rethrieving movie ids
    for j, id_unique in tqdm(enumerate(genre_list)):
        rest = genre_list[:j] + genre_list[j+1:]
        for i in range(nb_pages):
            url=url_start+str(i+1)+url_mid+str(id_unique)+url_end+"%2C".join(map(str, rest))
            req=requests.get(url, headers=headers).json()
            time.sleep(0.5) # to prevent overloading
            ids.extend(movie["id"] for movie in req["results"])
    
    return(ids)

def clean_data(df=None):
    """
        Calls a series of functions to clean the data-frame provided with get_movies_info

        Parameters:
        df: a pandas data-frame created with get_movies_info

        Returns:
        df1: a cleaned data-frame
    """

    assert df is not None, "No data frame was given"

    if  "status_message" in df.columns: #get rid of ligns where there was a status_error (=no data could be rethrieved)
        df1=df[df["status_message"].isna()]
    else:
        df1=df.copy()
    df1=df1.dropna(subset=['overview']) #get rid of ligns where there is no overview
    df1=drop_useless_info(df1)
    df1=keep_main_genre(df1)
    df1=full_poster_path(df1)
    df1=count_words(df1)
    df1=transform_date(df1)
    return(df1)

def drop_useless_info(df=None, drop_original_title=True):
    """
        Drops useless info

        Parameters:
        df: a pandas data-frame created with get_movies_info
        drop_original_title: a Boolean

        Returns:
        df1: a cleaned data-frame
    """

    assert df is not None, "No data frame was given"

    # Liste des colonnes à supprimer
    columns_to_drop = [
        "adult", "backdrop_path", "belongs_to_collection", "homepage", "imdb_id", 
        "origin_country", "original_language", "production_companies", "success",
        "production_countries", "spoken_languages", "status", "tagline", "video", 
        "belongs_to_collection.id", "belongs_to_collection.name", "status_code",
        "belongs_to_collection.poster_path", "belongs_to_collection.backdrop_path", "status_message"
    ]
    if drop_original_title:
        columns_to_drop.extend(["original_title"])

    # Vérifier quelles colonnes sont présentes avant de les supprimer
    columns_in_df = [col for col in columns_to_drop if col in df.columns]

    # Supprimer seulement les colonnes présentes
    df1 = df.drop(columns=columns_in_df)
    
    return df1

def keep_main_genre(df=None):
    """
        Changes the way genre is encoded

        Parameters:
        df: a pandas data-frame where 'genre' is a list of genre dictionnaries 

        Returns:
        df1: a cleaned data-frame with no 'genre' but a 'main_genre_id' and a 'main_genre_name' column
    """

    if isinstance(df['genres'].iloc[0], str): #in case we open a csv flight where the dictionnaries were transformed in str.
        df['genres'] = df['genres'].apply(ast.literal_eval)
    df = df.dropna(subset=['genres'])
    df = df[df['genres'].apply(lambda x: len(x) > 0)] #get rid of ligns ith no genre (encoded: "")

    df['main_genre_id'] = df['genres'].map(lambda x: x[0]['id'])
    df['main_genre_name'] = df['genres'].map(lambda x: x[0]['name'])
    df1=df.drop(columns=["genres"])
    return(df1)

def full_poster_path(df=None):
    """
        Changes the way the poster path is encoded

        Parameters:
        df: a pandas data-frame whith a 'poster_path' column

        Returns:
        df1: a pandas data frame with no 'poster_path' column but a 'full_poster_path' column containing the fulle poster url
    """

    df['full_poster_path']= "https://image.tmdb.org/t/p/original/"+df['poster_path']
    df1=df.drop(columns=["poster_path"])
    return(df1)

def count_words(df=None):
    """
        Changes the way the poster path is encoded

        Parameters:
        df: a pandas data-frame whith a 'poster_path' column

        Returns:
        df1: a pandas data frame with no 'poster_path' column but a 'full_poster_path' column containing the fulle poster url
    """

    df1=df.copy()
    df1["overview_count"]=df1["overview"].str.len()
    df1["title_count"]=df1["title"].str.len()
    return(df1)

def transform_date(df=None):
    """
        Changes the way the date is encoded

        Parameters:
        df: a pandas data-frame whith a 'release_date' column

        Returns:
        df1: a pandas data frame with a 'timestamp' column: the number of seconds between 1970 and the movie release date. 
    """

    df['release_date'] = pd.to_datetime(df['release_date'])
    df['timestamp'] = df['release_date'].astype('int64') // 10**9
    return(df)