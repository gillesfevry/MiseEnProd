"""
train.py
~~~~~~~~
MLflow training script for movie revenue prediction.

Workflow:
1. Load cleaned data from ``--data-path`` if the file exists; otherwise
   fetch fresh data from the TMDB API and save it to ``--data-path``.
2. Run a grid search over both Random Forest and Elastic Net pipelines,
   logging every candidate as a nested MLflow run.
3. Select the best model across both families and log it as the parent run,
   together with the serialised sklearn pipeline.

Usage::

    python train.py                          # all defaults
    python train.py --data-path data/movies.csv --nb-pages 20
    python train.py --experiment-name my_exp --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn

from src.models.model_pipelines import (
    create_elastic_net_pipeline,
    create_random_forest_pipeline,
    grid_search_elastic_net,
    grid_search_random_forest,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    TEXT_FEATURES,
)
from src.data.make_dataset import (
    clean_dataset,
    get_movie_ids,
    get_movies_details,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH      = "data/movies_clean.csv"
DEFAULT_EXPERIMENT     = "movie_revenue_prediction"
DEFAULT_NB_PAGES       = 20
DEFAULT_N_FOLDS        = 10
DEFAULT_STARTING_DATE  = "2000-01-01"
DEFAULT_ENDING_DATE    = "2023-12-31"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_or_fetch_data(
    data_path: str,
    nb_pages: int,
    starting_date: str,
    ending_date: str,
) -> pd.DataFrame:
    """Load cleaned movie data from CSV, or fetch it from TMDB if unavailable.

    If ``data_path`` points to an existing file it is loaded directly,
    skipping the API calls entirely. Otherwise the TMDB API is queried,
    the result is cleaned, and the DataFrame is saved to ``data_path`` for
    future runs.

    Args:
        data_path: Path to the CSV cache file.
        nb_pages: Number of TMDB discover pages to fetch when calling the API
            (20 movies per page).
        starting_date: Earliest release date filter (``"YYYY-MM-DD"``).
        ending_date: Latest release date filter (``"YYYY-MM-DD"``).

    Returns:
        Cleaned DataFrame ready for modelling.
    """
    path = Path(data_path)

    if path.exists():
        logger.info("Loading cached data from %s", path)
        return pd.read_csv(path)

    logger.info("No cache found at %s — fetching from TMDB API...", path)
    ids = get_movie_ids(
        nb_pages=nb_pages,
        starting_date=starting_date,
        ending_date=ending_date,
    )
    raw_df   = get_movies_details(ids=ids)
    clean_df = clean_dataset(raw_df)

    path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(path, index=False)
    logger.info("Cleaned data saved to %s (%d rows)", path, len(clean_df))

    return clean_df


# ---------------------------------------------------------------------------
# Grid search + MLflow logging helpers
# ---------------------------------------------------------------------------


def _log_grid_search_runs(
    results: pd.DataFrame,
    model_type: str,
    data: pd.DataFrame,
) -> tuple[dict, object]:
    """Log every grid search candidate as a nested MLflow run.

    Each row in ``results`` becomes one child run. The row with the lowest
    ``rmse_mean`` is also returned so the caller can compare across model
    families.

    Args:
        results: DataFrame produced by :func:`grid_search_random_forest` or
            :func:`grid_search_elastic_net`, sorted by ``rmse_mean``.
        model_type: Human-readable label (``"random_forest"`` or
            ``"elastic_net"``), used to tag each run.
        data: Full cleaned DataFrame, used to refit the best pipeline.

    Returns:
        Tuple of ``(best_params_dict, fitted_pipeline)`` for the row with
        the lowest ``rmse_mean``.
    """
    metric_cols = {"rmse_mean", "rmse_std"}
    best_row = results.iloc[0]
    best_params = {k: v for k, v in best_row.items() if k not in metric_cols}
    best_pipeline = None

    for _, row in results.iterrows():
        params = {k: v for k, v in row.items() if k not in metric_cols}
        with mlflow.start_run(run_name=f"{model_type}_candidate", nested=True):
            mlflow.set_tag("model_type", model_type)
            mlflow.log_params(params)
            mlflow.log_metric("rmse_mean", row["rmse_mean"])
            mlflow.log_metric("rmse_std",  row["rmse_std"])

    # Refit the best pipeline on all data
    X = data[TEXT_FEATURES + CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = data[TARGET]

    if model_type == "random_forest":
        best_pipeline = create_random_forest_pipeline(**best_params)
    else:
        best_pipeline = create_elastic_net_pipeline(**best_params)

    best_pipeline.fit(X, y)
    return best_params, best_pipeline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    data_path: str,
    experiment_name: str,
    nb_pages: int,
    n_folds: int,
    starting_date: str,
    ending_date: str,
    ) -> None:
    """Execute the full training and logging pipeline.

    Steps:

    1. Load or fetch cleaned data.
    2. Open a parent MLflow run for the full experiment.
    3. Run grid searches for both model families, logging each candidate as
       a nested run.
    4. Select the overall best model, log its params, metrics, and
       serialised pipeline to the parent run.

    Args:
        data_path: Path to CSV cache (loaded if present, saved otherwise).
        experiment_name: MLflow experiment name.
        nb_pages: TMDB pages to fetch if the cache is missing.
        n_folds: Number of cross-validation folds.
        starting_date: Earliest TMDB release date filter.
        ending_date: Latest TMDB release date filter.
    """
    # -- Data ------------------------------------------------------------------
    df = load_or_fetch_data(data_path, nb_pages, starting_date, ending_date)
    logger.info("Dataset ready: %d rows, %d columns", *df.shape)

    # -- MLflow experiment -----------------------------------------------------
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="grid_search_all_models"):
        mlflow.log_param("data_path",     data_path)
        mlflow.log_param("n_folds",       n_folds)
        mlflow.log_param("nb_rows",       len(df))
        mlflow.log_param("starting_date", starting_date)
        mlflow.log_param("ending_date",   ending_date)

        # -- Random Forest grid search -----------------------------------------
        logger.info("Starting Random Forest grid search...")
        rf_results = grid_search_random_forest(data=df)
        rf_params, rf_pipeline = _log_grid_search_runs(rf_results, "random_forest", df)

        # -- Elastic Net grid search -------------------------------------------
        logger.info("Starting Elastic Net grid search...")
        en_results = grid_search_elastic_net(data=df)
        en_params, en_pipeline = _log_grid_search_runs(en_results, "elastic_net", df)

        # -- Select best model across families ---------------------------------
        best_rf_rmse = rf_results.iloc[0]["rmse_mean"]
        best_en_rmse = en_results.iloc[0]["rmse_mean"]

        if best_rf_rmse <= best_en_rmse:
            best_model_type = "random_forest"
            best_params = rf_params
            best_pipeline = rf_pipeline
            best_rmse = best_rf_rmse
            best_std = rf_results.iloc[0]["rmse_std"]
        else:
            best_model_type = "elastic_net"
            best_params = en_params
            best_pipeline = en_pipeline
            best_rmse = best_en_rmse
            best_std = en_results.iloc[0]["rmse_std"]

        logger.info(
            "Best model: %s | RMSE: %,.0f (+/- %,.0f)",
            best_model_type, best_rmse, best_std,
        )

        # -- Log best model to parent run -------------------------------------
        mlflow.set_tag("best_model_type", best_model_type)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_rmse_mean", best_rmse)
        mlflow.log_metric("best_rmse_std",  best_std)
        mlflow.sklearn.log_model(best_pipeline, 
            artifact_path="best_model",
            pip_requirements=[f"scikit-learn=={sklearn.__version__}"]
        )

        logger.info("Run complete. Launch `mlflow ui` to explore results.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search + MLflow logging for movie revenue prediction."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"CSV cache path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--nb-pages",
        type=int,
        default=DEFAULT_NB_PAGES,
        help=f"TMDB pages to fetch if cache is missing (default: {DEFAULT_NB_PAGES})",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=DEFAULT_N_FOLDS,
        help=f"Cross-validation folds (default: {DEFAULT_N_FOLDS})",
    )
    parser.add_argument(
        "--starting-date",
        default=DEFAULT_STARTING_DATE,
        help=f"Earliest release date YYYY-MM-DD (default: {DEFAULT_STARTING_DATE})",
    )
    parser.add_argument(
        "--ending-date",
        default=DEFAULT_ENDING_DATE,
        help=f"Latest release date YYYY-MM-DD (default: {DEFAULT_ENDING_DATE})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        data_path = args.data_path,
        experiment_name = args.experiment_name,
        nb_pages = args.nb_pages,
        n_folds = args.n_folds,
        starting_date = args.starting_date,
        ending_date = args.ending_date,
    )