"""A simple API to expose our trained model."""
from datetime import datetime
from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import skops.io as sio


from src.data.make_dataset import (
    get_movie_ids,
    get_movies_details,
    clean_dataset
)
path = "data/test.csv"

app = FastAPI(
    title="Prédiction du revenu d'un film",
    description="Application de prédiction du revenu d'un film 🎬​",
)

MODEL_PATH = Path("models/best_model.skops")

def load_model():
    trusted_types = sio.get_untrusted_types(file=MODEL_PATH)
    return sio.load(MODEL_PATH, trusted=trusted_types)


model = load_model()


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
async def predict(
) -> list:
    """ """

    today = datetime.now()
    starting_time = today.strftime("%Y-%m-%d")

    ids = get_movie_ids(4, starting_date=starting_time, ascending=True, minimal_vote_count=0)
    print(ids)
    raw_df = get_movies_details(ids=ids)
    new_df = clean_dataset(raw_df)

    new_df.to_csv(path, index=False)

    df_to_predict = pd.read_csv("data/test.csv")
    prediction = model.predict(df_to_predict)

    return prediction