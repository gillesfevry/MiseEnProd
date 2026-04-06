# ENSAE-Data-Science-project

Ce dépôt contient le projet de programmation de Gilles Févry, Clément Morelière et Gabin Simmonet dans le cadre du cours "Python pour la dat science"en 2A à l'ENSAE et encadré par Mr Daniel Marin

--Contenu du projet--

Grâce à l'API de TMDB, nous explorons les données cinématographiques et cherchons à prédire certaines caractéristiques de films, en particulier, la note moyenne laissée par les utilisateurs ainsi que le genre d'un film. Dans le premier cas nous obtenons une erreur moyenne de 0,4 sur 10 et dans le second nous parvenons à prédire le bon genre dans plus de 60% des cas
Enfin en croisant cette base de donnée avec celle des lieux de tournages à Paris, nous étudions le lien entre lieu de tournage et genre d'un film. 

NB: Le projet est davantage introduit et problématisé dans "Notebook Main". 

--Fichiers--

Sous Python 3.12.7 le code s'exécute normalement sans problème

Le projet se compose essentiellement du Notebook "Notebook Main". Pour rendre le projet plu lisible, ce Notebook fait appel à deux fichiers Python: "tmdbdata" et "genderprediction". Ces fichiers contiennent respectivement les fonctions permettant de nettoyer les données et de prédire le genre des films. 

Le fichier "requirements" recense les packages Python externes à installer afin de pouvoir lancer les 3 premières parties du Notebook. La première cellule du Notebook permet de les installer directement. La deuxième cellule du Notebbok contient les packages nécessaires pour la dernière partie du projet. 

Des fichiers CSV sont également directement stockés dans ce repo. La manière dont ils sont obtenus est explicité dans le Notebook, mais ils permettent essentiellement de ne pas constamment relancer les requêtes de l'API qui peuvent s'avérer être un peu longues. Enfin nous avons directement importé le fichier des lieux de tournage depuis data.gouv.fr et l'avons glissé sur le repo Github. 

Enfin, il est important de noter que la clef API utilisée pour les requêtes est directement inscrite dans le fichier tmdbdata. Il est donc important que ce repo restent en l'Etat privé. 

Nouveau README : 


# MiseEnProd — Analyse de données cinématographiques

## Description du projet

Ce projet explore les données cinématographiques issues de **TMDB** et de **data.gouv** (lieux de tournage à Paris). Son objectif ets de savoir ce qu'on peut diree d'un film sans l'avoir vu.

### Objectifs
- **Récupération et nettoyage** des données via l'API TMDB
- **Statistiques descriptives** sur les films 
- **Prédiction de la note** d'un film via modele
- **Prédiction du genre** d'un film via  modele

## Structure du projet

```
MiseEnProd-main/
├── License
├── data/
│   ├── raw/              # The original, immutable data dump. (on S3)
│   └── processed/        # The final, canonical data sets for modeling. (on S3)
│
├── models/               # Trained and serialized models, model predictions, or model summaries
│
├── notebooks/            # Jupyter notebooks for exploration.
│
├── reports/              # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/          # Graphics.
│
├── src/                  # Source code for use in this project.
│   ├── config.py         # Charge of secret if needed
│
│   ├── data/                     # Scripts to download or generate data.
│   │   ├── make_dataset.py       # for cleaning the data
│   │   └── download_from_s3.py   # to download the data
│
│   ├── features/                 # Scripts to turn raw data into features for modeling.
│   │   └── build_features.py     # Construction des features
│
│   ├── models/                   # Scripts to train models and then use trained models to make predictions.
│   │   ├── predict_genre.py      # Genre's prediction 
│   │   └── predict_rating.py     # Note prediction 
│
│   └── visualization/            # Scripts to create exploratory and results oriented visualizations.
│       └── visualize.py          
│
├── .gitignore
├── pyproject.toml                # Configuration Ruff (linter/formatter)
├── requirements.txt              # Dépendances Python
├── secrets.yaml.example          # Template pour les secrets
└── README.md
```


## Installation

### Prérequis
- Python 3.12+
- Git

### Étapes

1. **Cloner le dépôt**


2. **Créer un environnement virtuel**


3. **Installer les dépendances**


4. **Configurer les secrets**
Dans un secrets.yaml mettre cette structure : 

```
tmdb:
  bearer_token: "Notre_TOKEN"

```

5. **Télécharger les données**
(Voir s'il vaut mieux pas importer directement les données quand on en a besoin, plutot que de les télécharger, mais le faire à la fin).
Utiliser le script `src\data\download_from_s3.py` en lançant : `python -m src.data.download_from_s3`.
Attention, il faut revoir comment implémenter : 

DATA_DIR, BUCKET, FILES. (Pour le moment en brut dans le script, j'ai testé en mettant bien els csv sur s3 et ça marche mais il faudra plutot les mettre dans un fichier de config et expliquer ici quoi mettre dans la config pour lancer)



## Utilisation


## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).

## Auteurs

Projet réalisé dans le cadre du cours "MISE EN PRODUCTION" en A à l'ENSAE.
