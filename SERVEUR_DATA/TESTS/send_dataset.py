import requests
import json
import os

# URL du serveur
URL = "http://127.0.0.1:8000/datasets/"
SOURCE_DIR = "/home/sofsoflefoufou/Documents/Code/PROJET_GL/SERVEUR_DATA/Datas"

for filename in os.listdir(SOURCE_DIR):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(SOURCE_DIR, filename)

    with open(filepath, "r") as f:
        data = json.load(f)

    try:
        response = requests.post(URL, json=data, timeout=10)
        if response.status_code == 200:
            print(f"✅ {filename} envoyé avec succès")
        else:
            print(f"⚠️ Échec pour {filename} : {response.status_code}")
    except Exception as e:
        print(f"⚠️ Erreur pour {filename} :", e)
