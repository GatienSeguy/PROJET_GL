import requests

url = "http://127.0.0.1:8000/datasets/data_supression/"

payload = {
    "name": "bruno_mars"
}

response = requests.post(url, json=payload)

print("Status:", response.status_code)
print("Réponse:", response.text)
