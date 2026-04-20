# ENSAE Mise en production — Prédiction de revenus de films à partir de données TMDB

## Contexte

Ce projet a été réalisé dans le cadre du cours **"Mise en production"** en 3A à l'ENSAE. Il s'inscrit dans la continuité du projet de **"Python pour la data science"** (2A) de Gilles Févry, Clément Morelière et Gabin Simmonet.

Le projet initial explorait les données cinématographiques de [TMDB](https://www.themoviedb.org/) pour prédire la note moyenne et le genre des films. Cette version restructure et industrialise le projet en appliquant les bonnes pratiques de mise en production.

## Objectifs

- **Récupérer et nettoyer** les données cinématographiques via l'API TMDB
- **Construire des features** pertinentes pour la modélisation
- **Entraîner des modèles** de prédiction du revenu d'un film
- **Tracer les expérimentations** avec MLflow
- **Déployer une API** de prédiction

## Déploiement

Pour le déploiement de l'application, se rendre sur [https://movie.lab.sspcloud.fr/](https://movie.lab.sspcloud.fr/). Puis, pour faire une request, faire [https://movie.lab.sspcloud.fr/docs](https://movie.lab.sspcloud.fr/docs).

> **Note** : il se peut qu'il s'affiche `no available server`, dans ce cas actualiser la page pour refaire fonctionner le site.

> **Note** : le déploiement de l'application [https://movie.lab.sspcloud.fr/](https://movie.lab.sspcloud.fr/) est contrôlé par un autre dépôt ([https://github.com/RebeccaBle/application-deployment](https://github.com/RebeccaBle/application-deployment)).


## Structure du projet

```
MiseEnProd/
├── src/                          # Code source modulaire
│   ├── __init__.py
│   ├── data/                     # Récupération et nettoyage des données
│   │   ├── __init__.py
│   │   ├── make_dataset.py       # Nettoyage des données brutes
│   │   └── download_from_s3.py   # Téléchargement des données depuis S3
│   ├── features/                 # Construction des features
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                   # Entraînement et prédiction
│   │   ├── __init__.py
│   │   ├── config.py             # Chargement des secrets (API token)
│   │   ├── model_pipelines.py    # Définition des pipelines scikit-learn
│   │   ├── predict_genre.py      # Prédiction du genre
│   │   ├── predict_rating.py     # Prédiction de la note
│   │   └── train.py              # Entraînement (prédiction du revenu)
│   └── visualization/            # Visualisations
│       ├── __init__.py
│       └── visualize.py
├── app/                          # API de prédiction
│   ├── api.py                    # Point d'entrée de l'API
│   └── run.sh                    # Script de lancement
├── notebooks/                    # Notebooks d'exploration et anciens scripts
│   ├── Notebook_Main.ipynb
│   └── ...
├── deployment/                   # Configuration Kubernetes
│   ├── deployment.yaml
│   ├── service.yaml
│   └── ingress.yaml
├── .github/workflows/            # CI/CD GitHub Actions
│   ├── test.yaml                 # Tests automatisés
│   └── prod.yml                  # Déploiement en production
├── .gitignore
├── requirements.txt              # Dépendances Python (avec versions exactes)
├── pyproject.toml                # Configuration du projet et de Ruff
├── uv.lock                       # Lockfile uv
├── install.sh                    # Script d'installation
├── Dockerfile                    # Conteneurisation de l'application
├── LICENSE                       # Licence MIT
└── README.md
```

## Installation

### Prérequis

- Python 3.12+
- Git
- [uv](https://docs.astral.sh/uv/) (recommandé) ou pip

### Étapes

**1. Cloner le dépôt**

```bash
git clone https://github.com/gillesfevry/MiseEnProd.git
cd MiseEnProd
```

**2. Créer l'environnement et installer les dépendances**

Avec `uv` (recommandé) :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Avec `pip` :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Avec `conda` :

```bash
conda env create -f environment.yml
conda activate miseenprod
```

**3. Configurer les secrets**

Créer un fichier `secrets.yaml` à la racine du projet. Ce fichier ne doit **jamais** être commité :

```yaml
tmdb:
  bearer_token: "VOTRE_TOKEN_TMDB"
```

Pour obtenir un token, créer un compte sur [themoviedb.org](https://www.themoviedb.org/) puis générer un Bearer Token dans Settings > API.

**4. Récupérer les données**

Les données sont récupérées automatiquement via l'API TMDB au premier lancement si elles ne sont pas présentes localement. Elles peuvent aussi être téléchargées depuis S3 :

```bash
uv run python -m src.data.download_from_s3
```

## Utilisation

### Lancer l'entraînement

```bash
uv run python -m src.models.train
```

Ou sans uv :

```bash
python -m src.models.train
```

### Visualiser les résultats avec MLflow

```bash
uv run mlflow ui
```

Ouvrir le navigateur à l'adresse `http://localhost:5000`.

> **Note (SSP Cloud)** : vérifier que le port 5000 est bien activé dans la configuration du service VSCode (onglet Networking).

## Qualité du code

Le projet utilise [Ruff](https://docs.astral.sh/ruff/) comme linter et formatter :

```bash
ruff check src/ --fix
ruff format src/
```

## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).

## Auteurs

Projet réalisé dans le cadre du cours "MISE EN PRODUCTION" en A à l'ENSAE.

## Automatisation

Le deploiement de l'application (https://movie.lab.sspcloud.fr/) est contrôlé par un autre dépôt.
