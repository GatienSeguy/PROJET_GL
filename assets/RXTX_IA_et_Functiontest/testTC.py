import requests

r = requests.post(
    "http://192.168.27.66:8000/datasets/info_all",
    json={"message": "choix dataset"}
)

print(r.status_code)
print(r.text)