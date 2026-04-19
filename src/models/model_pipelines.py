"""
model_pipelines.py

Scikit-learn pipeline factories and cross-validation utilities for movie
revenue prediction.

Two model families are supported:

* **Random Forest** — :func:`create_random_forest_pipeline`
* **Elastic Net**   — :func:`create_elastic_net_pipeline`

Both share the same preprocessing strategy: TF-IDF on combined text fields,
median imputation + standard scaling on numeric features, and one-hot
encoding on categorical features.

The Elastic Net pipeline wraps the regressor in a
:class:`~sklearn.compose.TransformedTargetRegressor` that standard-scales
the target (``revenue``) before fitting. This prevents convergence issues
caused by the very large magnitude of revenue values (~10⁸).

Typical workflow::

    pipeline = create_random_forest_pipeline(n_estimators=200, max_depth=20)
    scores   = model_cross_validation(data=df, pipeline=pipeline)
    print(f"RMSE: {(-scores).mean():,.0f}")

    # Or run a full grid search:
    results = grid_search_random_forest(data=df)
"""

from __future__ import annotations

import itertools
import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

#: Text columns combined and vectorised with TF-IDF.
TEXT_FEATURES: list[str] = ["overview", "title"]

#: Categorical columns one-hot encoded.
CATEGORICAL_FEATURES: list[str] = ["main_genre_name", "origin_country"]

#: Numeric columns imputed and scaled.
NUMERIC_FEATURES: list[str] = [
    "budget",
    "popularity",
    "runtime",
    "vote_count",
    "vote_average",
    "timestamp",
]

#: Regression target column.
TARGET: str = "revenue"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _combine_text(X: pd.DataFrame) -> pd.Series:
    """Concatenate ``overview`` and ``title`` into a single text series.

    Args:
        X: DataFrame containing at least ``overview`` and ``title`` columns.

    Returns:
        Series of strings with the two fields joined by a space.
    """
    return X["overview"].fillna("") + " " + X["title"].fillna("")


def _build_preprocessor(
    tfidf_max_features: int,
    tfidf_ngram_range: tuple[int, int],
    tfidf_min_df: int,
) -> ColumnTransformer:
    """Build the shared :class:`~sklearn.compose.ColumnTransformer` preprocessor.

    The preprocessor applies three sub-pipelines in parallel:

    * **text** — combines ``overview`` + ``title`` then applies TF-IDF.
    * **num**  — median imputation followed by standard scaling.
    * **cat**  — most-frequent imputation followed by one-hot encoding.

    Args:
        tfidf_max_features: Maximum number of TF-IDF vocabulary terms to keep.
        tfidf_ngram_range: N-gram range passed to
            :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        tfidf_min_df: Minimum document frequency for a term to be included.

    Returns:
        Configured :class:`~sklearn.compose.ColumnTransformer` instance.
    """
    text_pipeline = Pipeline(
        [
            ("combine", FunctionTransformer(_combine_text, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=tfidf_max_features,
                    stop_words="english",
                    ngram_range=tfidf_ngram_range,
                    min_df=tfidf_min_df,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("text", text_pipeline, TEXT_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------


def create_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    tfidf_max_features: int = 5000,
    tfidf_ngram_range: tuple[int, int] = (1, 1),
    tfidf_min_df: int = 1,
    random_state: int = 0,
) -> Pipeline:
    """Build a full preprocessing + Random Forest regression pipeline.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree. ``None`` means nodes are
            expanded until all leaves are pure or contain fewer than
            ``min_samples_split`` samples.
        tfidf_max_features: Maximum number of TF-IDF vocabulary terms.
        tfidf_ngram_range: N-gram range for the TF-IDF vectoriser.
        tfidf_min_df: Minimum document frequency for a TF-IDF term.
        random_state: Random seed for reproducibility.

    Returns:
        Scikit-learn :class:`~sklearn.pipeline.Pipeline` ready to be fitted.

    Example:
        >>> pipeline = create_random_forest_pipeline(n_estimators=200, max_depth=20)
        >>> pipeline.fit(X_train, y_train)
    """
    preprocessor = _build_preprocessor(
        tfidf_max_features, tfidf_ngram_range, tfidf_min_df
    )

    return Pipeline(
        [
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_elastic_net_pipeline(
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    tfidf_max_features: int = 1000,
    tfidf_ngram_range: tuple[int, int] = (1, 1),
    tfidf_min_df: int = 1,
    max_iter: int = 10000,
    tol: float = 1e-3,
) -> Pipeline:
    """Build a full preprocessing + Elastic Net regression pipeline.

    Elastic Net combines L1 and L2 regularisation. ``l1_ratio=1.0`` is
    equivalent to Lasso; ``l1_ratio=0.0`` is equivalent to Ridge.

    The regressor is wrapped in a
    :class:`~sklearn.compose.TransformedTargetRegressor` that applies
    :class:`~sklearn.preprocessing.StandardScaler` to the target before
    fitting. This is critical for convergence: revenue values (~10⁸) are
    too large for the coordinate descent solver to converge reliably without
    target scaling.

    Args:
        alpha: Overall regularisation strength. Higher values shrink
            coefficients more aggressively. Values below 0.1 are not
            recommended as they tend to cause convergence issues even with
            target scaling.
        l1_ratio: Mix ratio between L1 and L2 penalties (0 ≤ l1_ratio ≤ 1).
            ``1.0`` is pure Lasso, ``0.0`` is pure Ridge.
        tfidf_max_features: Maximum number of TF-IDF vocabulary terms.
        tfidf_ngram_range: N-gram range for the TF-IDF vectoriser.
        tfidf_min_df: Minimum document frequency for a TF-IDF term.
        max_iter: Maximum number of iterations for the coordinate descent
            solver. Increased from the sklearn default (1000) to 10 000 to
            accommodate high-dimensional TF-IDF feature spaces.
        tol: Convergence tolerance for the solver. Relaxed from the sklearn
            default (1e-4) to 1e-3 to speed up convergence on sparse data
            without meaningfully affecting solution quality.

    Returns:
        Scikit-learn :class:`~sklearn.pipeline.Pipeline` ready to be fitted.

    Example:
        >>> pipeline = create_elastic_net_pipeline(alpha=0.1, l1_ratio=0.7)
        >>> pipeline.fit(X_train, y_train)
    """
    preprocessor = _build_preprocessor(
        tfidf_max_features, tfidf_ngram_range, tfidf_min_df
    )

    regressor = TransformedTargetRegressor(
        regressor=ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
        ),
        transformer=StandardScaler(),
    )

    return Pipeline(
        [
            ("preprocessing", preprocessor),
            ("model", regressor),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def model_cross_validation(
    data: pd.DataFrame,
    pipeline: Pipeline,
    n_folds: int = 10,
    scoring: str = "neg_root_mean_squared_error",
) -> np.ndarray:
    """Evaluate a pipeline with k-fold cross-validation.

    :class:`~sklearn.exceptions.ConvergenceWarning` are suppressed during
    cross-validation to keep logs readable. If convergence is a concern,
    increase ``max_iter`` or ``tol`` in :func:`create_elastic_net_pipeline`.

    Args:
        data: DataFrame containing all feature and target columns.
        pipeline: Fitted or unfitted scikit-learn pipeline to evaluate.
        n_folds: Number of cross-validation folds.
        scoring: Scikit-learn scoring metric string. Defaults to
            ``"neg_root_mean_squared_error"`` so that negating the scores
            yields RMSE values.

    Returns:
        Array of ``n_folds`` scores. For ``neg_*`` metrics, negate the
        array to obtain positive error values.

    Example:
        >>> scores = model_cross_validation(data=df, pipeline=pipeline)
        >>> print(f"RMSE: {(-scores).mean():,.0f} (+/- {(-scores).std():,.0f})")
    """
    X = data[TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = data[TARGET]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        scores = cross_val_score(pipeline, X, y, cv=n_folds, scoring=scoring, n_jobs=-1)

    return scores


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

#: Default hyperparameter grid for Random Forest grid search.
_DEFAULT_RF_PARAM_GRID: dict[str, list] = {
    "n_estimators": [100, 200],
    "max_depth": [None],
    "tfidf_max_features": [1000, 3000, 5000],
    "tfidf_ngram_range": [(1, 1), (1, 2), (2, 3)],
    "tfidf_min_df": [2, 5],
}


#: Default hyperparameter grid for Elastic Net grid search.
#: Note: ``alpha=0.01`` is intentionally excluded — very low regularisation
#: causes convergence failures even with target scaling and high ``max_iter``.
_DEFAULT_EN_PARAM_GRID: dict[str, list] = {
    "alpha": [0.01, 0.1, 1.0],
    "l1_ratio": [0.5, 0.7, 0.9],
    "tfidf_max_features": [5000, 10000],
    "tfidf_ngram_range": [(1, 1), (1, 2)],
    "tfidf_min_df": [2, 5, 10, 20, 30],
}


def grid_search_random_forest(
    data: pd.DataFrame,
    param_grid: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """Exhaustively search hyperparameter combinations for Random Forest.

    For each combination in ``param_grid``, a pipeline is built and
    evaluated with :func:`model_cross_validation`. Results are sorted by
    ascending mean RMSE.

    Args:
        data: DataFrame containing all feature and target columns.
        param_grid: Dictionary mapping hyperparameter names to lists of
            candidate values. Keys must match the arguments of
            :func:`create_random_forest_pipeline`. If ``None``, falls back
            to :data:`_DEFAULT_RF_PARAM_GRID`.

    Returns:
        DataFrame with one row per hyperparameter combination, sorted by
        ``rmse_mean`` ascending. Includes ``rmse_mean``, ``rmse_std``, and
        one column per hyperparameter.

    Example:
        >>> results = grid_search_random_forest(data=df)
        >>> print(results.head())
    """
    if param_grid is None:
        param_grid = _DEFAULT_RF_PARAM_GRID

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    logger.info("%d combinations to test...", len(combinations))
    results: list[dict] = []

    for combo in combinations:
        params = dict(zip(keys, combo))
        pipeline = create_random_forest_pipeline(**params)
        scores = model_cross_validation(data=data, pipeline=pipeline)
        rmse = (-scores).mean()
        std = (-scores).std()

        logger.info("RMSE: %s (+/- %s) | %s", f"{rmse:,.0f}", f"{std:,.0f}", params)
        results.append({**params, "rmse_mean": rmse, "rmse_std": std})

    return pd.DataFrame(results).sort_values("rmse_mean")


def grid_search_elastic_net(
    data: pd.DataFrame,
    param_grid: Optional[dict[str, list]] = None,
) -> pd.DataFrame:
    """Exhaustively search hyperparameter combinations for Elastic Net.

    For each combination in ``param_grid``, a pipeline is built and
    evaluated with :func:`model_cross_validation`. Results are sorted by
    ascending mean RMSE.

    Args:
        data: DataFrame containing all feature and target columns.
        param_grid: Dictionary mapping hyperparameter names to lists of
            candidate values. Keys must match the arguments of
            :func:`create_elastic_net_pipeline`. If ``None``, falls back
            to :data:`_DEFAULT_EN_PARAM_GRID`.

    Returns:
        DataFrame with one row per hyperparameter combination, sorted by
        ``rmse_mean`` ascending. Includes ``rmse_mean``, ``rmse_std``, and
        one column per hyperparameter.

    Example:
        >>> results = grid_search_elastic_net(data=df)
        >>> print(results.head())
    """
    if param_grid is None:
        param_grid = _DEFAULT_EN_PARAM_GRID

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*param_grid.values()))

    logger.info("%d combinations to test...", len(combinations))
    results: list[dict] = []

    for combo in combinations:
        params = dict(zip(keys, combo))
        pipeline = create_elastic_net_pipeline(**params)
        scores = model_cross_validation(data=data, pipeline=pipeline)
        rmse = (-scores).mean()
        std = (-scores).std()

        logger.info("RMSE: %s (+/- %s) | %s", f"{rmse:,.0f}", f"{std:,.0f}", params)
        results.append({**params, "rmse_mean": rmse, "rmse_std": std})

    return pd.DataFrame(results).sort_values("rmse_mean")
