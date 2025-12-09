import requests

url = "http://127.0.0.1:8001/models/model_delete"

payload = {
    "name": "nouveau_modele_test"
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Réponse:", response.text)
