"""A simple API to expose our trained model."""

from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import skops.io as sio

app = FastAPI(
    title="Prédiction du revenu d'un film",
    description="Application de prédiction du revenu d'un film",
)

# ---------------------------------------------------------------------
# Load model (SKOPS only → simple & stable)
# ---------------------------------------------------------------------

MODEL_PATH = Path("models/best_model.skops")

def load_model():
    trusted_types = [
        "numpy.dtype",
        "src.models.model_pipelines._combine_text",
    ]

    return sio.load(MODEL_PATH, trusted=trusted_types)


model = load_model()

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.get("/", tags=["Welcome"])
def show_welcome_page():
    return {
        "Message": "API de prédiction du revenu d'un film",
        "Model_name": "Revenu ML",
        "Model_version": "0.1",
    }


@app.get("/predict", tags=["Predict"])
def predict(
    title: str,
    overview: str,
    main_genre_name: str,
    original_language: str,
    origin_country: str,
    timestamp: int,
    runtime: float,
    budget: float,
    popularity: float,
    vote_average: float,
    vote_count: float,
):
    X = pd.DataFrame([{
        "title": title,
        "overview": overview,
        "main_genre_name": main_genre_name,
        "original_language": original_language,
        "origin_country": origin_country,
        "timestamp": timestamp,
        "runtime": runtime,
        "budget": budget,
        "popularity": popularity,
        "vote_average": vote_average,
        "vote_count": vote_count,
    }])

    prediction = model.predict(X)[0]

    return {"prediction": float(prediction)}