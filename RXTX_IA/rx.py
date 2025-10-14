import requests, json

URL = "http://138.231.149.81:8000" 

payload = {
    "horizon": 8,
    "dates": ["2025-01-01", "2025-01-31"],
    "pas_temporel": 1,
    "split_train": 0.9,
    "freq": "H"
}

r = requests.post(f"{URL}/tempoconfig", json=payload)
print("POST /tempoconfig ->", r.status_code)
print(json.dumps(r.json(), indent=2))

payload_series = {
    "timestamps": [
        "2025-01-01T00:00:00Z",
        "2025-01-01T01:00:00Z",
        "2025-01-01T02:00:00Z"
    ],
    "values": [10.0, 12.5, 11.8]
}

r = requests.post(f"{URL}/timeseries", json=payload_series)
print("POST /timeseries ->", r.status_code)
print(json.dumps(r.json(), indent=2))