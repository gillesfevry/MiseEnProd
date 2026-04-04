"""
tmdb_extraction.py

Extraction and cleaning of movie data from the TMDB API (The Movie Database).

Typical workflow::

    ids = get_movie_ids_list(headers=HEADERS, nb_pages=5,
                             starting_date="2020-01-01", ending_date="2023-12-31")
    raw_df   = get_movies_info(ids=ids, headers=HEADERS)
    clean_df = clean_data(raw_df)
"""

from __future__ import annotations

import ast
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: HTTP headers for the TMDB API (Gilles' account).
#: Replace the Bearer token if it expires.
HEADERS: dict[str, str] = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI0MmM0YzIxMjQyOTMwZDAwYzBkMWJhYmVhY2IwMGZlMyIsIm5iZiI6MTczMjI5MjUwNS43NjMsInN1YiI6IjY3NDBhZjk5ZGQ1M2ZhMDA2MDMzMzdjNSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.rNcPPSpBYoqNQ5frYu4B8Hai3cjeF9KsjyIqOSNXCa4",
}

#: List of genre dictionaries ``{id, name}`` provided by the TMDB API.
GENRE_DICTIONARY: list[dict] = requests.get(
    "https://api.themoviedb.org/3/genre/movie/list?language=en",
    headers=HEADERS,
    timeout=10,
).json()["genres"]

# ---------------------------------------------------------------------------
# Raw data fetching
# ---------------------------------------------------------------------------

_DISCOVER_URL = "https://api.themoviedb.org/3/discover/movie"
_MOVIE_DETAIL_URL = "https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
_POSTER_BASE_URL = "https://image.tmdb.org/t/p/original"

#: Columns systematically removed during the cleaning step.
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


def get_movie_ids_list(
    headers: dict[str, str],
    nb_pages: int = 1,
    starting_date: Optional[str] = None,
    ending_date: Optional[str] = None,
    minimal_vote_count: int = 100,
    ascending: bool = False,
) -> list[int]:
    """Fetch a list of TMDB movie IDs via the ``/discover/movie`` endpoint.

    Args:
        headers: TMDB authentication HTTP headers.
        nb_pages: Number of pages to fetch (20 movies per page).
        starting_date: Earliest release date filter in ``"YYYY-MM-DD"`` format.
            If ``None``, no lower date bound is applied.
        ending_date: Latest release date filter in ``"YYYY-MM-DD"`` format.
            If ``None``, no upper date bound is applied.
        minimal_vote_count: Minimum number of votes required to include a movie.
        ascending: If ``True``, results are sorted by release date ascending;
            otherwise descending.

    Returns:
        List of integer TMDB movie IDs.

    Raises:
        AssertionError: If ``nb_pages`` is less than 1.

    Example:
        >>> ids = get_movie_ids_list(headers=HEADERS, nb_pages=3,
        ...                          starting_date="2022-01-01")
        >>> len(ids) > 0
        True
    """
    assert isinstance(nb_pages, int) and nb_pages >= 1, "nb_pages must be >= 1"

    params: dict[str, object] = {
        "include_adult": "false",
        "include_video": "false",
        "language": "en-US",
        "vote_count.gte": minimal_vote_count,
        "sort_by": (
            "primary_release_date.asc" if ascending else "primary_release_date.desc"
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
                _DISCOVER_URL, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            ids.extend(movie["id"] for movie in results if "id" in movie)
        except requests.exceptions.RequestException as exc:
            print(f"Error on page {page}: {exc}")

    return ids


def get_movies_info(ids: list[int], headers: dict[str, str]) -> pd.DataFrame:
    """Fetch details for each movie and return a raw DataFrame.

    A 100 ms delay is introduced between requests to respect TMDB API
    rate limits.

    Args:
        ids: List of TMDB movie IDs (obtained via :func:`get_movie_ids_list`).
        headers: TMDB authentication HTTP headers.

    Returns:
        Un-cleaned DataFrame produced by ``pd.json_normalize``.
        Each row corresponds to one movie; columns reflect the JSON
        structure returned by the API.

    Raises:
        AssertionError: If ``ids`` is empty.

    Example:
        >>> df = get_movies_info(ids=[550], headers=HEADERS)
    """
    assert len(ids) >= 1, "The list of IDs is empty"

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


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame, drop_original_title: bool = True) -> pd.DataFrame:
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

    if "status_message" in df.columns:
        df = df[df["status_message"].isna()]
    else:
        df = df.copy()

    df = df.dropna(subset=["overview"])
    df = drop_useless_columns(df, drop_original_title)
    df = keep_main_genre(df)
    df = keep_first_origin_country(df)
    df = build_full_poster_path(df)
    df = add_text_length_columns(df)
    df = convert_date_to_timestamp(df)
    return df


def drop_useless_columns(
    df: pd.DataFrame, drop_original_title: bool = True
) -> pd.DataFrame:
    """Drop columns that are not relevant for analysis.

    Only columns actually present in ``df`` are dropped; missing columns
    are silently ignored.

    Args:
        df: DataFrame to clean.
        drop_original_title: If ``True``, the ``original_title`` column is
            also dropped.

    Returns:
        New DataFrame without the irrelevant columns.

    Raises:
        AssertionError: If ``df`` is ``None``.
    """
    assert df is not None, "No DataFrame was provided"

    columns_to_drop = list(_COLUMNS_TO_DROP)
    if drop_original_title:
        columns_to_drop.append("original_title")

    present_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=present_columns)


def keep_main_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the genre list with the movie's primary genre.

    The ``genres`` column contains a list of ``{id, name}`` dictionaries.
    This function extracts the first entry and creates two new columns:
    ``main_genre_id`` and ``main_genre_name``.

    Note:
        Handles the case where ``genres`` has been serialised as a string
        (e.g. after reading from a CSV) by applying :func:`ast.literal_eval`.

    Args:
        df: DataFrame containing a ``genres`` column.

    Returns:
        DataFrame without the ``genres`` column, replaced by ``main_genre_id``
        and ``main_genre_name``.
    """
    if isinstance(df["genres"].iloc[0], str):
        df["genres"] = df["genres"].apply(ast.literal_eval)

    df = df.dropna(subset=["genres"])
    df = df[df["genres"].apply(lambda x: len(x) > 0)].copy()
    df["main_genre_id"] = df["genres"].map(lambda x: x[0]["id"])
    df["main_genre_name"] = df["genres"].map(lambda x: x[0]["name"])
    return df.drop(columns=["genres"])


def keep_first_origin_country(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only the first country of origin for each movie.

    Rows whose ``origin_country`` is not a non-empty list are dropped.

    Args:
        df: DataFrame containing an ``origin_country`` column of list type.

    Returns:
        DataFrame with ``origin_country`` reduced to a single string value.
    """
    df = df[
        df["origin_country"].apply(lambda x: isinstance(x, list) and len(x) > 0)
    ].copy()
    df["origin_country"] = df["origin_country"].apply(lambda x: x[0])
    return df


def build_full_poster_path(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full poster URL from the relative TMDB path.

    Replaces the ``poster_path`` column (relative path, e.g. ``/abc.jpg``)
    with ``full_poster_path`` (absolute TMDB CDN URL).

    Args:
        df: DataFrame containing a ``poster_path`` column.

    Returns:
        DataFrame with ``full_poster_path`` in place of ``poster_path``.
    """
    df = df.copy()
    df["full_poster_path"] = _POSTER_BASE_URL + df["poster_path"]
    return df.drop(columns=["poster_path"])


def add_text_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add character-count columns for the movie title and overview.

    Args:
        df: DataFrame containing ``title`` and ``overview`` columns.

    Returns:
        DataFrame enriched with ``title_char_count`` and
        ``overview_char_count`` columns.
    """
    df = df.copy()
    df["title_char_count"] = df["title"].str.len()
    df["overview_char_count"] = df["overview"].str.len()
    return df


def convert_date_to_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``release_date`` to a Unix timestamp (seconds since 1970-01-01).

    The original ``release_date`` column is kept as ``datetime64``; a new
    ``timestamp`` column (64-bit integer) is added alongside it.

    Args:
        df: DataFrame containing a ``release_date`` column parseable by
            :func:`pandas.to_datetime`.

    Returns:
        DataFrame with ``release_date`` converted and ``timestamp`` added.
    """
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["timestamp"] = df["release_date"].astype("int64") // 10**9
    return df
