"""Chargement des configurations sensibles depuis secrets.yaml.
Pour ne pas avoir à mettre les clefs en brut dans le code
"""

from pathlib import Path
import yaml
import os

SECRETS_PATH = Path(__file__).resolve().parents[2] / "secrets.yaml"


def load_secrets() -> dict:
    """Charge les secrets depuis le fichier secrets.yaml.

    Returns :
        dict
            Dictionnaire contenant les secrets du projet.
    """
    token_env = os.getenv("JETON_API")
    if token_env:
        return {"tmdb": {"bearer_token": token_env}}

    if not SECRETS_PATH.exists():
        raise FileNotFoundError(f"Fichier {SECRETS_PATH} introuvable. ")

    with open(SECRETS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_tmdb_headers() -> dict:
    """Retourne les headers d'authentification pour l'API TMDB.

    Returns:
        dict
            Headers HTTP contenant le token Bearer TMDB.
    """

    secrets = load_secrets()
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {secrets['tmdb']['bearer_token']}",
    }


if __name__ == "__main__":
    print(f"Chemin cherché pour les secrets : {SECRETS_PATH}")

    try:
        mes_secrets = load_secrets()
        print(mes_secrets)

        headers = get_tmdb_headers()
        print(headers)

    except FileNotFoundError as e:
        print(f" erreur : {e}")
    except KeyError as e:
        print(f" erreur : Clé manquante dans le YAML. Il manque : {e}")
