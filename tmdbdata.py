#This module is used to extract data from TMDB using it's API

import requests
import pandas as pd
from tqdm import tqdm
import time

#headers linked to Gilles' account needed to use the API
headers = {"accept": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzNjcwNTY1MTA0YzUzY2Q0MjE3N2M3ZWQyMmVmZDk1ZCIsIm5iZiI6MTczMjI5MzAwNy4yMTk0MDM1LCJzdWIiOiI2NzQwYWY5OWRkNTNmYTAwNjAzMzM3YzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.8DPhrUjEakZO4CxeovXkNmSOe01nH3r5kEZlntq547M"
            }

def get_movie_ids_list(n=None, headers=None):
    start="https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=fr-FR&page="
    end="&sort_by=vote_count.desc&with_original_language=fr"
    ids=[]
    print("getting movie ids")
    for i in tqdm(range(n)):
        url=start+str(i+1)+end
        req=requests.get(url, headers=headers).json()
        time.sleep(0.5) # si trop lent, réduire
        ids.extend(movie["id"] for movie in req["results"])
    return(ids)

def get_movies_info(ids=None, headers=None):
    start= "https://api.themoviedb.org/3/movie/"
    end="?language=en-US"
    dataframes=[]
    print("getting movie info")
    for i in tqdm(ids):
        url=start+str(i)+end
        req = requests.get(url, headers=headers).json()
        time.sleep(0.5) #si trop lent, réduire
        dataframes.append(pd.json_normalize(req)) #pas très efficace voir si on peut améliorer
    df = pd.concat(dataframes, ignore_index=True)
    return(df)

