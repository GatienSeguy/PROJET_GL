import requests
import json
import os

# URL du serveur FastAPI
URL = "http://127.0.0.1:8000/datasets/"

# Dossier où se trouvent tes fichiers JSON
DATA_DIR = "/home/sofsoflefoufou/Documents/Code/PROJET_GL/SERVEUR_DATA"

# Liste des fichiers à envoyer
FILES = ["CACAO.json", "EUROS.json"]

for filename in FILES:
    filepath = os.path.join(DATA_DIR, filename)

    # Vérifie que le fichier existe
    if not os.path.exists(filepath):
        print(f"❌ Fichier introuvable : {filepath}")
        continue

    # Lecture du JSON
    with open(filepath, "r") as f:
        data = json.load(f)

    # Envoi au serveur
    response = requests.post(URL, json=data)

    # Vérifie la réponse
    if response.status_code == 200:
        print(f"✅ {filename} envoyé avec succès")
        print("→ ID attribué :", response.json()["dataset_id"])
    else:
        print(f"⚠️ Erreur pour {filename} :", response.status_code, response.text)
