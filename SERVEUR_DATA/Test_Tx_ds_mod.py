# client_ds_model.py
import requests
import json
from pathlib import Path

# ==========================
# CONFIGURATION GLOBALE
# ==========================
URL = "http://192.168.27.66:8000"

# ==========================
#  ENVOYER DATASET
# ==========================
def envoyer_dataset(nom_dataset: str, fichier_json: Path | None = None, payload: dict | None = None):
    """Envoie un dataset (soit depuis un fichier .json, soit via un payload Python)."""
    if fichier_json and Path(fichier_json).exists():
        print(f">>> Envoi du dataset depuis le fichier : {fichier_json}")
        with open(fichier_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif payload:
        print(">>> Envoi du dataset depuis un payload Python")
        data = payload
    else:
        raise ValueError("Spécifie un fichier JSON ou un payload pour le dataset.")
    
    data.setdefault("name", nom_dataset)
    r = requests.post(f"{URL}/datasets", json=data, timeout=60)
    if r.status_code == 409:
        print(f"⚠️ Le nom '{nom_dataset}' est déjà utilisé pour un dataset.")
        return None
    r.raise_for_status()
    print("✅ Dataset envoyé avec succès.")
    return r.json()


# ==========================
# ENVOYER MODELE
# ==========================
def envoyer_modele(nom_modele: str, fichier_modele: Path):
    """Envoie un modèle binaire (.pth, .pt, .onnx...)"""
    fichier_modele = Path(fichier_modele)
    if not fichier_modele.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {fichier_modele}")
    
    print(f">>> Envoi du modèle : {fichier_modele}")
    with open(fichier_modele, "rb") as f:
        files = {"file": (fichier_modele.name, f, "application/octet-stream")}
        data = {"name": nom_modele}
        r = requests.post(f"{URL}/models/upload", files=files, data=data, timeout=120)
    if r.status_code == 409:
        print(f"⚠️ Le nom '{nom_modele}' est déjà utilisé pour un modèle.")
        return None
    r.raise_for_status()
    print("✅ Modèle envoyé avec succès.")
    return r.json()


# ==========================
#  DEMANDER DATASET
# ==========================
def demander_dataset(nom_dataset: str, dossier_sortie: Path = Path("./downloads")):
    """Télécharge le dernier dataset enregistré sous ce nom."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    print(f">>> Téléchargement du dataset '{nom_dataset}'")
    r = requests.get(f"{URL}/datasets/by-name/{nom_dataset}/download", timeout=60)
    r.raise_for_status()
    out_path = dossier_sortie / f"{nom_dataset}_latest.json"
    out_path.write_bytes(r.content)
    print(f"✅ Dataset sauvegardé sous {out_path}")
    return out_path


# ==========================
#  DEMANDER MODELE
# ==========================
def demander_modele(nom_modele: str, dossier_sortie: Path = Path("./downloads")):
    """Télécharge le dernier modèle enregistré sous ce nom."""
    dossier_sortie.mkdir(parents=True, exist_ok=True)
    print(f">>> Téléchargement du modèle '{nom_modele}'")
    r = requests.get(f"{URL}/models/by-name/{nom_modele}/download", timeout=120)
    r.raise_for_status()
    out_path = dossier_sortie / f"{nom_modele}_latest.pth"
    out_path.write_bytes(r.content)
    print(f"✅ Modèle sauvegardé sous {out_path}")
    return out_path


# ==========================
#  EXEMPLE D’UTILISATION
# ==========================
if __name__ == "__main__":
    dataset_name = "EURO"
    model_name = "EURO_MLP"
    
    dataset_file = Path("/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/Datas/EURO.json")
    model_file = Path("/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/mon_modele.pth")

    # Payload alternatif si fichier absent
    payload_dataset = {
        "name": dataset_name,
        "timestamps": ["2025-11-01", "2025-11-02", "2025-11-03"],
        "values": [18.5, 19.2, 18.9],
        "meta": {"source": "capteur_XYZ", "unite": "°C"},
    }

    # ---- Envoi ----
    envoyer_dataset(dataset_name, fichier_json=dataset_file, payload=payload_dataset)
    envoyer_modele(model_name, model_file)

    # ---- Téléchargement ----
    demander_dataset(dataset_name)
    demander_modele(model_name)

    print("\n✅ Terminé.")