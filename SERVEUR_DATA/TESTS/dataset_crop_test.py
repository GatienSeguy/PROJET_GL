import requests
import json

URL = "http://127.0.0.1:8000/datasets/info_all"

payload = {
    "message": "choix dataset"
}

response = requests.post(URL, json=payload)

print("Status code :", response.status_code)
try:
    print(json.dumps(response.json(), indent=2))
except Exception:
    print("Réponse brute :", response.text)
