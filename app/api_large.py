"""A simple API to expose our trained model."""

from fastapi import FastAPI
import pandas as pd
from pathlib import Path
import skops.io as sio
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api-movie")

app = FastAPI(
    title="Prédiction du revenu d'un film",
    description="Application de prédiction du revenu d'un film",
)


MODEL_PATH = Path("models/best_model.skops")


def load_model():
    try:
        logger.info(
            f"Tentative de chargement du modèle depuis l'emplacement {MODEL_PATH}"
        )
        trusted_types = sio.get_untrusted_types(file=MODEL_PATH)
        m = sio.load(MODEL_PATH, trusted=trusted_types)
        logger.info("Modèle chargé.")
        return m
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise e


model = load_model()


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
