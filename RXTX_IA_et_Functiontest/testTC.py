import requests

url = "http://192.168.27.66:8000/datasets/info_all"
payload = {"message": "choix dataset"}

r = requests.post(url, json=payload)
print(r.status_code)
print(r.text)