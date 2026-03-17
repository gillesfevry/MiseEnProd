#This module is used to extract data from TMDB using its API

import requests
import pandas as pd
from tqdm import tqdm
import time
import ast
from datetime import datetime

#headers linked to Gilles' account needed to use the API
headers = {"accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MmM0YzIxMjQyOTMwZDAwYzBkMWJhYmVhY2IwMGZlMyIsIm5iZiI6MTczMjI5MjUwNS43NjMsInN1YiI6IjY3NDBhZjk5ZGQ1M2ZhMDA2MDMzMzdjNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.rNcPPSpBYoqNQ5frYu4B8Hai3cjeF9KsjyIqOSNXCa4"
            }

#The genre dictionnary provided by the API.
genre_dictionnary =requests.get('https://api.themoviedb.org/3/genre/movie/list?language=en', headers=headers).json()['genres']

def get_movie_ids_list(nb_pages=1, headers=None, starting_date=None):
    """
    Creates a list of movie IDs using TMDB discover API.
    """

    assert headers is not None, "Headers must not be None"
    assert isinstance(nb_pages, int) and nb_pages >= 1, "nb_pages must be >= 1"

    if starting_date is None:
        starting_date = datetime.now().strftime("%Y-%m-%d")

    url = "https://api.themoviedb.org/3/discover/movie"

    ids = []

    print("getting movie ids")

    for page in tqdm(range(1, nb_pages + 1)):
        params = {
            "include_adult": "false",
            "include_video": "false",
            "language": "en-US",
            "page": page,
            "primary_release_date.lte": starting_date,
            "sort_by": "primary_release_date.desc",
            "vote_count.gte": 100,
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])

            ids.extend(movie.get("id") for movie in results if "id" in movie)

        except requests.exceptions.RequestException as e:
            print(f"Error on page {page}: {e}")
            continue  # skip page

    return ids

def get_movies_info(ids=None, headers=None):
    """
    Creates an uncleaned data frame with all the informations of the movies.

    Parameters:
    ids (list of ints)
    headers (dict)

    Returns:
    Pandas DataFrame
    """

    assert headers is not None, "Headers must not be None"
    assert ids is not None and len(ids) >= 1, "List is empty"

    url_start = "https://api.themoviedb.org/3/movie/"
    url_end = "?language=en-US"

    movies = []

    print("getting movie info")

    for i in tqdm(ids):
        url = f"{url_start}{i}{url_end}"
        req = requests.get(url, headers=headers).json()
        time.sleep(0.1)

        movies.append(req)  

    df = pd.json_normalize(movies)

    return df


def clean_data(df=None, drop_original_title=True):
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
    df1=drop_useless_info(df1, drop_original_title)
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
        "adult", "backdrop_path", "homepage", "imdb_id", 
        "original_language", "production_companies", "success",
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