import os
import requests



# URL du serveur FastAPI
URL = "http://127.0.0.1:8000/models/upload/"

# Envoie du fichier

files = {"file": ("mon_modele.pth", "application/octet-stream")}
response = requests.post(URL, files=files)

print("wesh")
print("Status code:", response.status_code)
print("Response:", response.json())
