import s3fs
import os
import h5py
import numpy as np


def download_s3_folder(
    bucket_name='projet-slums-detection/',
    s3_folder='challenge_mexique/',
    local_dir='../data/'
):
    """
    Télécharge tous les fichiers d'un dossier S3 dans un répertoire local.

    :param bucket_name: Nom du bucket S3.
    :param s3_folder: Chemin du dossier sur S3 à télécharger.
    :param local_dir: Chemin local où télécharger les fichiers.
    """
    print("*****Téléchargement des données*****")
    fs = s3fs.S3FileSystem(
        client_kwargs={"endpoint_url": f'https://{os.environ["AWS_S3_ENDPOINT"]}'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"],
    )

    files = fs.ls(f"{bucket_name}/{s3_folder}")

    for file in files:
        file_path = file.replace(bucket_name+s3_folder, "")
        local_file_path = os.path.join(local_dir, file_path)

        local_file_dir = os.path.dirname(local_file_path)
        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        if not os.path.exists(local_file_path):
            print(f"Téléchargement de {file} vers {local_file_path}")
            fs.get(file, local_file_path)

        else:
            print(f"Le fichier {file} a déjà été téléchargé ici {local_file_path}")


def load_data(filepath="../data/train_data.h5", has_labels=True):
    print("*****Ouverture des données*****")

    with h5py.File(filepath, 'r') as hdf:
        # Extract the images (X)
        X = np.array(hdf['images'])
        if has_labels:
            # Extract the labels (y)
            y = np.array(hdf['labels'])
        else:
            y = np.zeros(X.shape[0])

    return X, y
