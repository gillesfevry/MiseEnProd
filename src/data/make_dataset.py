"""Extraction et nettoyage des données cinématographiques via l'API TMDB.

Ce module contient les fonctions pour :
- Récupérer les identifiants de films via l'endpoint Discover
- Récupérer les détails de chaque film via l'endpoint Movie Details
- Nettoyer et transformer les données brutes en DataFrame exploitable

Il a pour fonction de remplacer tmdbdata_extraction.py
"""

import ast
import time

import pandas as pd
import requests
from tqdm import tqdm

from src.models.config import get_tmdb_headers


TMDB_DISCOVER_URL = (
    "https://api.themoviedb.org/3/discover/movie"
    "?include_adult=false&include_video=false&language=en-US"
)
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie"
TMDB_GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list?language=en"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/original/"


def get_genre_dictionary() -> list[dict]:
    """Récupère le dictionnaire des genres depuis l'API TMDB.

    Returns:
        Liste de dictionnaires contenant les genres (id, name).
    """
    headers = get_tmdb_headers()
    response = requests.get(TMDB_GENRE_URL, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()["genres"]


def get_movie_ids(nb_pages: int = 1, strategy: str = "top_rated") -> list[int]:
    """Récupère une liste d'IDs de films via l'API TMDB Discover.

    Remplace l'ancien `get_movie_ids_list`, `get_movie_ids_list_map`, `get_balanced_movie_list`

    Arguments:
        nb_pages: Nombre de pages à récupérer. Chaque page contient 20 films.
        strategy : 
            Stratégie de sélection parmi :
            - ``"top_rated"`` : films les plus notés (par nombre de votes)
            - ``"french_recent"`` : films français depuis 2016
            - ``"balanced"`` : échantillon équilibré par genre

    Returns
        Liste des identifiants TMDB des films.
    """
    if strategy == "top_rated":
        return _get_ids_top_rated(nb_pages)
    if strategy == "french_recent":
        return _get_ids_french_recent(nb_pages)
    if strategy == "balanced":
        return _get_ids_balanced(nb_pages)
    raise ValueError(
        f"Stratégie inconnue : '{strategy}'. "
        "Utilisez 'top_rated', 'french_recent' ou 'balanced'."
    )


def get_movies_details(ids: list[int]) -> pd.DataFrame:
    """Récupère les informations détaillées de chaque film.

    Cette fonction remplace l'ancienne `get_movies_info()` 

    Arguments:
        ids: Liste des identifiants TMDB des films.

    Returns:
        DataFrame contenant toutes les informations brutes des films.
    """

    if not ids:
        raise ValueError("La liste d'identifiants est vide.")

    headers = get_tmdb_headers()
    dataframes: list[pd.DataFrame] = []

    #rethrieving information and stocking it in a list of dataframes
    for movie_id in tqdm(ids):
        url = f"{TMDB_MOVIE_URL}/{movie_id}?language=en-US"
        response = requests.get(url, headers=headers, timeout=10)
        time.sleep(0.1) #si trop lent, réduire
        dataframes.append(pd.json_normalize(response.json())) #pas très efficace voir si on peut améliorer
    return pd.concat(dataframes, ignore_index=True)


def clean_dataset(df: pd.DataFrame, drop_original_title: bool = True) -> pd.DataFrame:
    """Nettoie le DataFrame brut issu de ``get_movies_details``.

    Applique successivement :
    1. Suppression des lignes en erreur 
    2. Suppression des lignes sans synopsis
    3. Suppression des colonnes inutiles
    4. Extraction du genre principal
    5. Conservation du premier pays d'origine
    6. Construction de l'URL complète de l'affiche
    7. Ajout du nombre de caractères du synopsis et du titre
    8. Conversion de la date en timestamp Unix

    Arguments:
        df: DataFrame brut issu de ``get_movies_details``.
        drop_original_title: Si True, supprime la colonne ``original_title``.

    Returns:
        DataFrame nettoyé et enrichi.
    """
    result = df.copy()

    # get rid of ligns where there was a status_error (=no data could be rethrieved)
    if "status_message" in result.columns:
        result = result[result["status_message"].isna()]

    # get rid of ligns where there is no overview
    result = result.dropna(subset=["overview"])
    result = _drop_useless_columns(result, drop_original_title)
    result = _extract_main_genre(result)
    result = _keep_first_origin_country(result)
    result = _build_poster_url(result)
    result = _add_text_length_features(result)
    result = _add_timestamp(result)
    return result


# ---------------------------------------------------------------------------
# récupération d'IDs
# ---------------------------------------------------------------------------


def _get_ids_top_rated(nb_pages: int) -> list[int]:
    """Récupère les IDs des films les plus notés (triés par vote_count).
    
    Cette fonction remplace `get_movie_ids_list`

    Arguments: 
        nb_pages (int >=1): The number of pages in the Discover option we want to get data from. A page lists 20 movies.

    Returns:
        list of movie ids

    """
    headers = get_tmdb_headers()
    ids: list[int] = []

    print("Récupération des IDs (top rated)...")
    for page in tqdm(range(1, nb_pages + 1)):
        url = f"{TMDB_DISCOVER_URL}&page={page}&sort_by=vote_count.desc"
        response = requests.get(url, headers=headers, timeout=10)
        time.sleep(0.5) # to prevent overloading
        ids.extend(movie["id"] for movie in response.json()["results"])
    return ids


def _get_ids_french_recent(nb_pages: int) -> list[int]:
    """Récupère les IDs des films français sortis depuis 2016.
    
    Cette fonction remplace `get_movie_ids_list_map()`
    """
    headers = get_tmdb_headers()
    base_url = (
        "https://api.themoviedb.org/3/discover/movie"
        "?include_adult=false&include_video=false"
    )
    ids: list[int] = []

    #Récupération des IDs (films français récents)
    for page in tqdm(range(1, nb_pages + 1)):
        url = (
            f"{base_url}&page={page}"
            "&primary_release_date.gte=2016-01-01"
            "&sort_by=primary_release_date.asc"
            "&with_original_language=fr"
        )
        response = requests.get(url, headers=headers, timeout=10)
        time.sleep(0.5)# to prevent overloading
        ids.extend(movie["id"] for movie in response.json()["results"])
    return ids


def _get_ids_balanced(nb_pages: int) -> list[int]:
    """Récupère un échantillon de films équilibré par genre.

    Cette fonction remplace `get_balanced_movie_list()`

    Pour chaque genre, récupère des films ayant uniquement ce genre
    (les autres genres sont exclus via ``without_genres``).
    """
    headers = get_tmdb_headers()
    genre_list = [genre["id"] for genre in get_genre_dictionary()]
    ids: list[int] = []

    print(f"Récupération des IDs (balanced, {len(genre_list)} genres)...")
    for idx, genre_id in tqdm(enumerate(genre_list), total=len(genre_list)):
        other_genres = genre_list[:idx] + genre_list[idx + 1 :]
        excluded = "%2C".join(str(g) for g in other_genres)
        for page in range(1, nb_pages + 1):
            url = (
                f"{TMDB_DISCOVER_URL}&page={page}"
                f"&sort_by=revenue.asc&vote_count.gte=2"
                f"&with_genres={genre_id}&without_genres={excluded}"
            )
            response = requests.get(url, headers=headers, timeout=10)
            time.sleep(0.5)
            ids.extend(movie["id"] for movie in response.json()["results"])
    return ids


# ---------------------------------------------------------------------------
# Fonctions privées — nettoyage
# ---------------------------------------------------------------------------

_COLUMNS_TO_DROP = [
    "adult",
    "backdrop_path",
    "belongs_to_collection",
    "homepage",
    "imdb_id",
    "original_language",
    "production_companies",
    "success",
    "production_countries",
    "spoken_languages",
    "status",
    "tagline",
    "video",
    "belongs_to_collection.id",
    "belongs_to_collection.name",
    "status_code",
    "belongs_to_collection.poster_path",
    "belongs_to_collection.backdrop_path",
    "status_message",
]


def _drop_useless_columns(df: pd.DataFrame, drop_original_title: bool = True) -> pd.DataFrame:
    """Supprime les colonnes inutiles pour l'analyse.
    
    Cette fonction remplace `drop_useless_info`
    """
    columns = list(_COLUMNS_TO_DROP)
    if drop_original_title:
        columns.append("original_title")
    existing = [col for col in columns if col in df.columns] # Vérifier quelles colonnes sont présentes avant de les supprimer
    return df.drop(columns=existing)


def _extract_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait le genre principal (premier de la liste) de chaque film.

    Remplace la colonne ``genres`` (liste de dicts) par deux colonnes :
    ``main_genre_id`` et ``main_genre_name``.

    Parameters:
        df: a pandas data-frame where 'genre' is a list of genre dictionnaries.

    Returns:
        a cleaned data-frame with no 'genre' but a 'main_genre_id' and a 'main_genre_name' column.
    """
    result = df.copy()

    # Convertir la chaîne en liste si nécessaire (cas CSV)
    if isinstance(result["genres"].iloc[0], str):
        result["genres"] = result["genres"].apply(ast.literal_eval)

    result = result.dropna(subset=["genres"])
    result = result[result["genres"].apply(lambda x: len(x) > 0)]
    result["main_genre_id"] = result["genres"].map(lambda x: x[0]["id"])
    result["main_genre_name"] = result["genres"].map(lambda x: x[0]["name"])
    return result.drop(columns=["genres"])


def _keep_first_origin_country(df: pd.DataFrame) -> pd.DataFrame:
    """Conserve uniquement le premier pays d'origine pour chaque film.
    
    Supprime les lignes où la liste des pays d'origine est vide.
    """
    result = df[
        df["origin_country"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    result["origin_country"] = result["origin_country"].apply(lambda x: x[0])
    return result


def _build_poster_url(df: pd.DataFrame) -> pd.DataFrame:
    """Changes the way the poster path is encoded

    Parameters:
        df: a pandas data-frame whith a 'poster_path' column

    Returns:
        a pandas data frame with no 'poster_path' column but a 'full_poster_path' column containing the fulle poster url
    """
    result = df.copy()
    result["full_poster_path"] = TMDB_IMAGE_BASE_URL + result["poster_path"]
    return result.drop(columns=["poster_path"])


def _add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le nombre de caractères du synopsis et du titre."""
    result = df.copy()
    result["overview_count"] = result["overview"].str.len()
    result["title_count"] = result["title"].str.len()
    return result


def _add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit la date de sortie en timestamp Unix (secondes)."""
    result = df.copy()
    result["release_date"] = pd.to_datetime(result["release_date"])
    result["timestamp"] = result["release_date"].astype("int64") // 10**9
    return result


if __name__ == "__main__":
    print(0)