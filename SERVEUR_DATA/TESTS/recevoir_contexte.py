import requests
import json

url = "http://127.0.0.1:8001/contexte/obtenir_solo"

payload = {
    "name": "dataset_test"   # <-- mets ton nom ici
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)

try:
    data = response.json()
    print("\n=== PAYLOAD DU CONTEXTE ===")
    print(json.dumps(data, indent=4, ensure_ascii=False))
except Exception:
    print("\nRéponse brute :", response.text)
