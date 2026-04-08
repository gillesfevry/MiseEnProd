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

Ce projet explore les données cinématographiques issues de **TMDB** et de **data.gouv** (lieux de tournage à Paris). Son objectif est de prédire les revenus d'un film à partir de ces données.

### Objectifs
- **Récupération et nettoyage** des données via l'API TMDB
- **Prédiction du revenu** d'un film via  modele

## Structure du projet

```
MiseEnProd-main/
├── License
├── data/
│   ├── movies_clean.csv              # Ce télécharge après la première utilisation, sinon va directement requeter l'API
│
├── notebooks/            # Jupyter notebooks for exploration and previous codes.
│
├── reports/              # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/          # Graphics.
│
├── src/                  # Source code for use in this project.

│
│   ├── data/                     # Scripts to download or generate data.
│   │   ├── make_dataset.py       # for cleaning the data
│   │   └── download_from_s3.py   # to download the data
│
│   ├── features/                 # Scripts to turn raw data into features for modeling.
│   │   └── build_features.py     # Construction des features
│
│   ├── models/                   # Scripts to train models and then use trained models to make predictions.
│   │   ├── config.py             # Charge of secret if needed
│   │   ├── model_pipelines.py    # 
│   │   └── train.py              # Revenu prediction
│
│   └── visualization/            # Scripts to create exploratory and results oriented visualizations.       
│
├── .gitignore
├── pyproject.toml                # Configuration Ruff (linter/formatter)
├── requirements.txt              # Python ependancies
├── secrets.yaml                  # Secret with token for the API
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
Où notre token est le token de l'API TMDB ( pour le moement il est encore hardcodé dans l'ancien code dans \notebooks\tmdb_extraction.py il faut donc le mettre dans le secrets.yaml

5. **Télécharger les données**
(Voir s'il vaut mieux pas importer directement les données quand on en a besoin, plutot que de les télécharger, mais le faire à la fin).
Utiliser le script `src\data\download_from_s3.py` en lançant : `python -m src.data.download_from_s3`.
Attention, il faut revoir comment implémenter : 

DATA_DIR, BUCKET, FILES. (Pour le moment en brut dans le script, j'ai testé en mettant bien els csv sur s3 et ça marche mais il faudra plutot les mettre dans un fichier de config et expliquer ici quoi mettre dans la config pour lancer)

Mais je crois que du coup même sans avoir les données ça va directment les télécharger depuis l'API si on a bien rempli le yaml.

## Utilisation

`python -m src.models.train`

Puis sur Onyxia run : 

`mlflow ui` dans le bash et ensuite 

Puis dans les ports sur VScode il y a le lien (n'a pas marché pour moi)



## Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE).

## Auteurs

Projet réalisé dans le cadre du cours "MISE EN PRODUCTION" en A à l'ENSAE.
