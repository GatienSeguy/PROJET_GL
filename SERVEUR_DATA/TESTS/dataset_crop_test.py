import requests
import json

# URL de ton serveur FastAPI (remplace le pcort si nécessaire)
url = "http://127.0.0.1:8000/dataset/data_solo/"

# Envoi de la requête POST

#Envoi d'un message pour demander EURO.json entre le 2020-01-01 et le 2020-06-01 avec un pas de 1 jour
response = requests.post(url, json={"message": "choix dataset", "name": "EURO", "date_debut": "2020-01-01", "date_fin": "2020-06-01", "pas": "40d"})


# Vérification du code HTTP
if response.status_code == 200:
    # Affichage du JSON reçu de façon lisible
    data = response.json()
    print(json.dumps(data, indent=2))
else:
    print(f"Erreur {response.status_code}: {response.text}")
