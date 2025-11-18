import requests
import json

# URL de ton serveur FastAPI (remplace le pcort si nécessaire)
url = "http://127.0.0.1:8001/choix_dataset/"

# Envoi de la requête POST
response = requests.post(url)

# Vérification du code HTTP
if response.status_code == 200:
    # Affichage du JSON reçu de façon lisible
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print(f"Erreur {response.status_code}: {response.text}")
