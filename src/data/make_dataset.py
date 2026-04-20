"""Extraction et nettoyage des données cinématographiques via l'API TMDB.

Ce module contient les fonctions pour :
- Récupérer les identifiants de films via l'endpoint Discover
- Récupérer les détails de chaque film via l'endpoint Movie Details
- Nettoyer et transformer les données brutes en DataFrame exploitable

Il a pour fonction de remplacer tmdbdata_extraction.py
"""

import ast
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.models.config import get_tmdb_headers


_DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
_MOVIE_DETAIL_URL = "https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
_GENRE_URL = "https://api.themoviedb.org/3/genre/movie/list?language=en"
_POSTER_BASE_URL = "https://image.tmdb.org/t/p/original"

_COLUMNS_TO_DROP: list[str] = [
    "adult",
    "backdrop_path",
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


def get_genre_dictionary() -> list[dict]:
    """Récupère le dictionnaire des genres depuis l'API TMDB.

    Returns:
        Liste de dictionnaires contenant les genres (id, name).
    """
    headers = get_tmdb_headers()
    response = requests.get(_GENRE_URL, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()["genres"]


def get_movie_ids(
    nb_pages: int = 1,
    starting_date: Optional[str] = None,
    ending_date: Optional[str] = None,
    minimal_vote_count: int = 100,
    ascending: bool = False) -> list[int]:
    """Récupère une liste d'IDs de films via l'API TMDB Discover.

    Remplace l'ancien `get_movie_ids_list`, `get_movie_ids_list_map`, `get_balanced_movie_list`

    Arguments:
        nb_pages: Nombre de pages à récupérer. Chaque page contient 20 films.
        starting_date: Date de sortie minimale au format ``"YYYY-MM-DD"``.
            Si ``None``, pas de borne inférieure.
        ending_date: Date de sortie maximale au format ``"YYYY-MM-DD"``.
            Si ``None``, pas de borne supérieure.
        minimal_vote_count: Nombre minimum de votes requis.
        ascending: Si ``True``, tri par date de sortie croissante ;
            sinon décroissante.

    Returns
        Liste des identifiants TMDB des films.
    """
    return _get_ids_by_date(
            nb_pages=nb_pages,
            starting_date=starting_date,
            ending_date=ending_date,
            minimal_vote_count=minimal_vote_count,
            ascending=ascending,
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
    movies: list[dict] = []
    print("Fetching movie details...")

    for movie_id in tqdm(ids):
        url = _MOVIE_DETAIL_URL.format(movie_id=movie_id)
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            movies.append(response.json())
        except requests.exceptions.RequestException as exc:
            print(f"Error on movie {movie_id}: {exc}")
        finally:
            time.sleep(0.1)

    return pd.json_normalize(movies)


def clean_dataset(df: pd.DataFrame, drop_original_title: bool = True) -> pd.DataFrame:
    """Orchestrate the full cleaning pipeline on a raw movie DataFrame.

    Steps applied in order:

    1. Drop rows with API errors (presence of ``status_message``).
    2. Drop rows with no synopsis (``overview``).
    3. Drop irrelevant columns (:func:`drop_useless_columns`).
    4. Keep only the main genre (:func:`keep_main_genre`).
    5. Keep only the first country of origin (:func:`keep_first_origin_country`).
    6. Build the full poster URL (:func:`build_full_poster_path`).
    7. Add character-count columns for title and overview (:func:`add_text_length_columns`).
    8. Convert release date to a Unix timestamp (:func:`convert_date_to_timestamp`).

    Args:
        df: Raw DataFrame produced by :func:`get_movies_info`.
        drop_original_title: If ``True``, the ``original_title`` column is dropped.

    Returns:
        Cleaned and enriched DataFrame.

    Raises:
        AssertionError: If ``df`` is ``None``.
    """ 
    assert df is not None, "No DataFrame was provided"

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

def _get_ids_by_date(
    nb_pages: int,
    starting_date: Optional[str] = None,
    ending_date: Optional[str] = None,
    minimal_vote_count: int = 100,
    ascending: bool = False,
) -> list[int]:
    """Récupère les IDs de films filtrés par date et nombre de votes.

    remplace la fonction `get_movie_ids_list` dans tmdb_extraction.py

    Arguments:
        nb_pages: Nombre de pages à récupérer (20 films par page).
        starting_date: Date de sortie minimale (``YYYY-MM-DD``).
        ending_date: Date de sortie maximale (``YYYY-MM-DD``).
        minimal_vote_count: Nombre minimum de votes requis.
        ascending: Si ``True``, tri par date croissante.

    Returns:
        Liste des identifiants TMDB des films.
    """
    headers = get_tmdb_headers()
    params: dict[str, object] = {
        "include_adult": "false",
        "include_video": "false",
        "language": "en-US",
        "vote_count.gte": minimal_vote_count,
        "sort_by": (
            "primary_release_date.asc" if ascending
            else "primary_release_date.desc"
        ),
    }

    if starting_date is not None:
        params["primary_release_date.gte"] = starting_date
    if ending_date is not None:
        params["primary_release_date.lte"] = ending_date

    ids: list[int] = []
    print("Fetching movie IDs...")

    for page in tqdm(range(1, nb_pages + 1)):
        params["page"] = page
        try:
            response = requests.get(
                _DISCOVER_URL, headers=headers, params=params, timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            ids.extend(movie["id"] for movie in results if "id" in movie)
        except requests.exceptions.RequestException as exc:
            print(f"Erreur page {page} : {exc}")

    return ids


# ---------------------------------------------------------------------------
# Fonctions nettoyage
# ---------------------------------------------------------------------------


def _drop_useless_columns(df: pd.DataFrame, drop_original_title: bool = True) -> pd.DataFrame:
    """Supprime les colonnes inutiles pour l'analyse.
    
    Cette fonction remplace `drop_useless_info`
    """
    columns = list(_COLUMNS_TO_DROP)
    if drop_original_title:
        columns.append("original_title")
    existing = [col for col in columns if col in df.columns] 
    return df.drop(columns=existing)


def _extract_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the genre list with the movie's primary genre.

    The ``genres`` column contains a list of ``{id, name}`` dictionaries.
    This function extracts the first entry and creates two new columns:
    ``main_genre_id`` and ``main_genre_name``.

    Note:
        Handles the case where ``genres`` has been serialised as a string
        (e.g. after reading from a CSV) by applying :func:`ast.literal_eval`.


    Arguments:
        df: DataFrame containing a ``genres`` column.

    Returns:
        DataFrame without the ``genres`` column, replaced by ``main_genre_id``
        and ``main_genre_name``.
    """
    result = df.copy()

    # Convertir la chaîne en liste si nécessaire (cas CSV)
    if isinstance(result["genres"].iloc[0], str):
        result["genres"] = result["genres"].apply(ast.literal_eval)

    result = result.dropna(subset=["genres"])
    result = result[result["genres"].apply(lambda x: len(x) > 0)].copy()
    result["main_genre_id"] = result["genres"].map(lambda x: x[0]["id"])
    result["main_genre_name"] = result["genres"].map(lambda x: x[0]["name"])
    return result.drop(columns=["genres"])


def _keep_first_origin_country(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only the first country of origin for each movie.

    Rows whose ``origin_country`` is not a non-empty list are dropped.

    Args:
        df: DataFrame containing an ``origin_country`` column of list type.

    Returns:
        DataFrame with ``origin_country`` reduced to a single string value.
    """
    result = df[
        df["origin_country"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    result["origin_country"] = result["origin_country"].apply(lambda x: x[0])
    return result


def _build_poster_url(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full poster URL from the relative TMDB path.

    Replaces the ``poster_path`` column (relative path, e.g. ``/abc.jpg``)
    with ``full_poster_path`` (absolute TMDB CDN URL).

    Args:
        df: DataFrame containing a ``poster_path`` column.

    Returns:
        DataFrame with ``full_poster_path`` in place of ``poster_path``.
    """

    result = df.copy()
    result["full_poster_path"] = _POSTER_BASE_URL + result["poster_path"]
    return result.drop(columns=["poster_path"])


def _add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``release_date`` to a Unix timestamp (seconds since 1970-01-01).

    The original ``release_date`` column is kept as ``datetime64``; a new
    ``timestamp`` column (64-bit integer) is added alongside it.

    Args:
        df: DataFrame containing a ``release_date`` column parseable by
            :func:`pandas.to_datetime`.

    Returns:
        DataFrame with ``release_date`` converted and ``timestamp`` added.
    """
    result = df.copy()
    result["title_char_count"] = result["title"].str.len()
    result["overview_char_count"] = result["overview"].str.len()
    return result


def _add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit la date de sortie en timestamp Unix (secondes)."""
    result = df.copy()
    result["release_date"] = pd.to_datetime(result["release_date"])
    result["timestamp"] = result["release_date"].astype("int64") // 10**9
    return result


if __name__ == "__main__":
    print("Module make_dataset chargé avec succès.")