"""Téléchargement des données brutes depuis le stockage S3 sur SSP Cloud.

Usage
-----
    python -m src.data.download_from_s3

Ce script télécharge les fichiers CSV depuisS3 vers le dossier ``data/raw/``. 
Voir pour peut être changer ça et faire en sorte de directement importer les données sans avoir à les télécharger.
"""

from pathlib import Path
from tqdm import tqdm
import s3fs

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" #Il faudra retirer ça pour ne pas l'avoir en brut dans nos dossiers

BUCKET = "s3://arnaud2701/MEP/Projet"  #Il faudra retirer ça pour ne pas l'avoir en brut dans nos dossiers

FILES = [
    "usbestmovies.csv",
    "balancedmovies.csv",
    "randommovies.csv",
    "data_tmdb_map.csv",
    "lieux-de-tournage-a-paris.csv",
] #Il faudra retirer ça pour ne pas l'avoir en brut dans nos dossiers


def download_data(bucket: str = BUCKET, dest: Path = DATA_DIR) -> None:
    """Télécharge les fichiers de données depuis S3.

    Arguments:
        bucket : str
            Chemin du bucket S3 (ex: ``s3://mon-bucket/data/raw``).
        dest : Path
            Dossier local de destination.
    """
    dest.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": "https://minio.lab.sspcloud.fr"}
    )

    for filename in tqdm(FILES, desc="Téléchargement"):
        remote = f"{bucket}/{filename}"
        local = dest / filename

        if local.exists():
            print(f"  {filename} existe déjà, skip.")
            continue

        fs.get(remote, str(local))
    print("Téléchargement terminé.")


if __name__ == "__main__":
    download_data()
