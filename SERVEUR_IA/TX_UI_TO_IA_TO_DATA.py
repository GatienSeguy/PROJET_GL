import requests
import json
import pydantic
from classes import newDatasetRequest
url = "http://192.168.27.66:8000/datasets/data_add_proxy"

filename = "/Users/gatienseguy/Documents/VSCode/PROJET_GL/SERVEUR_DATA/TESTS/bruno_mars.json"

with open(filename, "r") as f:
    content = json.load(f)

payload = {
    "name": "bruno_mars",
    "data": content
}
# payload = newDatasetRequest(payload["name"],payload["data"])
response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Réponse:", response.text)
