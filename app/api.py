"""A simple API to expose our trained model."""

from fastapi import FastAPI
from pathlib import Path
import skops.io as sio
import logging

from src.data.make_dataset import get_movies_details, clean_dataset

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
def predict(ID: int):
    logger.info(f"Requête de prédiction reçue pour le film d'ID {ID}")

    df = get_movies_details([ID])
    df = clean_dataset(df)

    prediction = model.predict(df)
    logger.info(f"Prédiction réussie pour le film d'ID {ID}")
    return {"prediction": float(prediction)}
