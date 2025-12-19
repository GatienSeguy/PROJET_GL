import json
import requests
from pathlib import Path

# URL du serveur
URL = "http://127.0.0.1:8001"

BASE_DIR = Path(__file__).resolve().parent
CODE_DIR = BASE_DIR.parent
DATASET_DIR = CODE_DIR / "SERVEUR_DATA" / "datasets"
print(f"DATASET_DIR = {DATASET_DIR}")
def send_dataset(file_path: Path, dataset_name: str):
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    # Lire le contenu du fichier JSON
    with file_path.open("r", encoding="utf-8") as f:
        time_series_data = json.load(f)

    # Construire le payload JSON attendu par /datasets/data_add
    payload = {
        "name": dataset_name,
        "data": time_series_data
    }

    # Envoyer le JSON directement
    response = requests.post(f"{URL}/datasets/data_add", json=payload)

    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except ValueError:
        print("Response text:", response.text)

    if response.status_code != 200:
        raise RuntimeError("Échec de l'upload du dataset")

    # ✅ Vérification locale si le dataset a été enregistré
    dataset_nom = dataset_name + ".json" 
    path_new_dataset = DATASET_DIR / dataset_nom
    if not path_new_dataset.exists():
        raise FileNotFoundError(
            f"Le dataset '{dataset_nom}' n'existe pas dans {DATASET_DIR}"
        )

    print(f"✅ Dataset '{dataset_name}' envoyé et enregistré avec succès.")

def delete_dataset(dataset_name: str):
    payload = {
        "name": dataset_name
    }

    response = requests.post(f"{URL}/datasets/data_supression", json=payload)

    print("Status code:", response.status_code)
    try:
        print("Response:", response.json())
    except ValueError:
        print("Response text:", response.text)

    if response.status_code != 200:
        raise RuntimeError("Échec de la suppression du dataset")

    # ✅ Vérification locale si le dataset a été supprimé
    dataset_nom = dataset_name + ".json" 
    path_dataset = DATASET_DIR / dataset_nom
    if path_dataset.exists():
        raise RuntimeError(
            f"Le dataset '{dataset_nom}' existe toujours dans {DATASET_DIR} après tentative de suppression"
        )

    print(f"✅ Dataset '{dataset_name}' supprimé avec succès.")


# Exemple d'appel
send_dataset(BASE_DIR / "Dataset_test_global.json", "sofiane_est_un_dieu")
delete_dataset("sofiane_est_un_dieu")
