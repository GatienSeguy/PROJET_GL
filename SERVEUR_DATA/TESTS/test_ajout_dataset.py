import requests
import json

url = "http://127.0.0.1:8000/datasets/data_add"

filename = "/home/sofsoflefoufou/Documents/Code/PROJET_GL/SERVEUR_DATA/TESTS/bruno_mars.json"

with open(filename, "r") as f:
    content = json.load(f)

payload = {
    "name": "bruno_mars",
    "data": content
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Réponse:", response.text)
