"""A simple API to expose our trained model."""

from datetime import datetime
from fastapi import FastAPI
import mlflow
import pandas as pd

from src.data.make_dataset import get_movie_ids, get_movies_details, clean_dataset

path = "data/test.csv"

mlflow.set_tracking_uri("file:./mlruns")
EXPERIMENT_NAME = "movie_revenue_prediction"
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
exp_id = exp.experiment_id
runs = mlflow.search_runs(
    experiment_ids=[exp_id], order_by=["metrics.best_rmse_mean ASC"]
)
best_run_id = runs.iloc[0].run_id
model_uri = f"runs:/{best_run_id}/best_model"
model = mlflow.sklearn.load_model(model_uri)


app = FastAPI(
    title="Prédiction du revenu d'un film",
    description="Application de prédiction du revenu d'un film",
)


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API de prédiction du revenu d'un film",
        "Model_name": "Revenu ML",
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
async def predict() -> str:
    """ """

    today = datetime.now()
    starting_time = today.strftime("%Y-%m-%d")

    ids = get_movie_ids(
        4, starting_date=starting_time, ascending=True, minimal_vote_count=0
    )
    print(ids)
    raw_df = get_movies_details(ids=ids)
    new_df = clean_dataset(raw_df)

    new_df.to_csv(path, index=False)

    df_to_predict = pd.read_csv("data/test.csv")
    prediction = model.predict(df_to_predict)

    return prediction
