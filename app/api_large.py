"""A simple API to expose our trained model."""

from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import skops.io as sio
import logging
import mlflow

logging.basicConfig(
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.DEBUG,
    handlers=[logging.FileHandler("api.log"), logging.StreamHandler()],
)

# Preload model -------------------

logging.info(
    "Getting model from MLFlow"
)

model_name = "production"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)


# Define app -------------------------

app = FastAPI(
    title="Prédiction du revenu d'un film",
    description="Application de prédiction du revenu d'un film",
)


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
    logger.info(f"Requête de prédiction reçue pour le movie nommé '{title}'")
    X = pd.DataFrame(
        [
            {
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
            }
        ]
    )

    prediction = model.predict(X)[0]
    logger.info(f"Prédiction réussie pour le film nommé '{title}'")
    return {"prediction": float(prediction)}
