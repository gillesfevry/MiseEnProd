# ENSAE-Data-Science-project

Ce dépôt contient le projet de programmation de Gilles Févry, Clément Morelière et Gabin Simmonet dans le cadre du cours "Python pour la dat science"en 2A à l'ENSAE et encadré par Mr Daniel Marin

--Contenu du projet--

Grâce à l'API de TMDB, nous explorons les données cinématographiques et cherchons à prédire certaines caractéristiques de film, en particulier, la note moyenne laissée par les utilisateurs ainsi que le genre d'un film. Dans le premier cas nous obtenons une erreur moyenne de 0,4 sur 10 et dans le second nous parvenons à prédire le bon genre dans plus de 60% des cas
Enfin en croisant cette base de donnée avec celle des lieux de tournages à Paris, nous étudions le lien entre lieu de tournage et genre d'un film. 

NB: Le projet est davantage introduit et problématisé dans "Notebook Main". 

--Fichiers--

Le projet se compose essentiellement du Notebook "Notebook Main". Pour rendre le projet plu lisible, ce Notebook fait appel à deux fichiers Python: "tmdbdata" et "genderprediction". Ces fichiers contiennent respectivement les fonctions permettant de nettoyer les données et de prédire le genre des films. 

Le fichier "requirements" recense les packages Python externes à installer afin de pouvoir lancer les 3 premières parties du Notebook. La première cellule du Notebook permet de les installer directement. La deuxième cellule du Notebbok contient les packages nécessaires pour la dernière partie du projet. 

Des fichiers CSV sont également directement stockés dans ce repo. La manière dont ils sont obtenus est explicité dans le Notebook, mais ils permettent essentiellement de ne pas constamment relancer les requêtes de l'API qui peuvent s'avérer être un peu longues. 

Enfin, il est important de noter que la clef API utilisée pour les requêtes est directement inscrite dans le fichier tmdbdata. Il est donc important que ce repo restent en l'Etat privé. 