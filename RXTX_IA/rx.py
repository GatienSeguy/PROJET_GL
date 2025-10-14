import requests, json
URL_SERVEUR = "http://localhost:8000/train"

payload_Test_ConfigTempo = {
    "horizon": 8,
    "dates": ["2025-01-01", "2025-01-31"],
    "pas_temporel": 1,
    "split_train": 0.9,
    "freq": "H"
  }

r = requests.post(URL_SERVEUR, json=payload_Test_ConfigTempo)
print("Code:", r.status_code)
print(json.dumps(r.json(), indent=2))



payload_Test_TimeSeriesData= {
    "timestamps": [
      "2025-01-01T00:00:00Z",
      "2025-01-01T01:00:00Z",
      "2025-01-01T02:00:00Z",
      "2025-01-01T03:00:00Z",
      "2025-01-01T04:00:00Z",
      "2025-01-01T05:00:00Z"
      
    ],
    "values":[1,2,4,8,16,32]
  }


r = requests.post(URL_SERVEUR, json=payload_Test_TimeSeriesData)
print("Code:", r.status_code)
print(json.dumps(r.json(), indent=2))