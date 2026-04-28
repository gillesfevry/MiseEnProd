"""A simple API to expose our trained model."""

from fastapi import FastAPI
import logging
import mlflow

from src.data.make_dataset import get_movies_details, clean_dataset

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

# Preload model -------------------

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
def predict(ID: int):
    logging.info(f"Requête de prédiction reçue pour le film dont l'ID est : {ID}")

    df = get_movies_details([ID])
    df = clean_dataset(df)

    prediction = model.predict(df)
    logging.info(f"Prédiction réussie pour le film dont l'ID est : {ID}")
    return {"prediction": float(prediction)}
