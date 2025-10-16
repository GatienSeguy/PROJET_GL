import requests, json

URL = "http://192.168.1.94:8000" 

payload = {
    "horizon": 8,
    "dates": ["2025-01-01", "2025-01-31"],
    "pas_temporel": 1,
    "split_train": 0.9,
    "freq": "H"
}

# r = requests.post(f"{URL}/tempoconfig", json=payload)
# print("POST /tempoconfig ->", r.status_code)
# print(json.dumps(r.json(), indent=2))


payload2 = {
    "Parametres_temporels": {
        "horizon": 24,
        "dates": ["2025-01-01", "2025-02-01"],
        "pas_temporel": 1,
        "portion_decoupage": 0.9
    },
    "Parametres_choix_reseau_neurones": {
        "modele": "MLP"
    },
    "Parametres_archi_reseau": {
        "nb_couches": 4,
        "hidden_size": 128,
        "dropout_rate": 0.2,
        "fonction_activation": "ReLU"
    },
    "Parametres_choix_loss_fct": {"fonction_perte": "MAE",
            "params" : None},

    "Parametres_optimisateur": {"optimisateur": "Adam", "learning_rate": 0.001, "decroissance" : None, "scheduler" : None, "patience": None},
    "Parametres_entrainement": {"nb_epochs": 10, "batch_size": 64, "clip_gradient" : 1},
    "Parametres_visualisation_suivi": {"metriques": ["MAE", "MSE"]}
} 


# r = requests.post(f"{URL}/train_full", json=payload2)
# print("POST /train_full ->", r.status_code)
# print(json.dumps(r.json(), indent=2))


payload3={
  "series": {
    "timestamps": [
      "2025-01-01T00:00:00",
      "2025-01-01T01:00:00",
      "2025-01-01T02:00:00",
      "2025-01-01T03:00:00",
      "2025-01-01T04:00:00",
      "2025-01-01T05:00:00",
      "2025-01-01T06:00:00",
      "2025-01-01T07:00:00",
      "2025-01-01T08:00:00",
      "2025-01-01T09:00:00",
      "2025-01-01T10:00:00",
      "2025-01-01T11:00:00"
    ],
    "values": [12.4, 12.7, 13.0, 12.9, 13.2, 13.5, 13.4, 13.7, 14.0, 13.9, 14.2, 14.5]
  },
  "config": {
    "Parametres_temporels": {
      "horizon": 1,
      "dates": ["2025-01-01", "2025-01-01"],
      "pas_temporel": 60,
      "portion_decoupage": 0.8
    },
    "Parametres_choix_reseau_neurones": {
      "modele": "MLP"
    },
    "Parametres_archi_reseau": {
      "nb_couches": 2,
      "hidden_size": 64,
      "dropout_rate": 0.0,
      "fonction_activation": "ReLU"
    },
    "Parametres_choix_loss_fct": {
      "fonction_perte": "MSE"
    },
    "Parametres_optimisateur": {
      "optimisateur": "Adam",
      "learning_rate": 0.001,
      "decroissance": 0.0,
      "scheduler": "None",
      "patience": 5
    },
    "Parametres_entrainement": {
      "nb_epochs": 1000,
      "batch_size": 4
    },
    "Parametres_visualisation_suivi": {
      "metriques": ["loss"]
    }
  }
}

# r = requests.post(f"{URL}/training", json=payload3)
# print("POST /training ->", r.status_code)
# print(json.dumps(r.json(), indent=2))


payload4={'Parametres_temporels': {'horizon': 1, 'dates': ['2025-01-01', '2025-01-01'], 'pas_temporel': 60, 'portion_decoupage': 0.8}, 
          'Parametres_choix_reseau_neurones': {'modele': 'MLP'},
            'Parametres_archi_reseau': {'nb_couches': 2, 'hidden_size': 64, 'dropout_rate': 0.0, 'fonction_activation': 'ReLU'},
            'Parametres_choix_loss_fct': {'fonction_perte': 'MSE', 'params': None},
            'Parametres_optimisateur': {'optimisateur': 'Adam', 'learning_rate': 0.001, 'decroissance': 0.0, 'scheduler': None, 'patience': 5},
              'Parametres_entrainement': {'nb_epochs': 1000, 'batch_size': 4, 'clip_grandient': None},
                'Parametres_visualisation_suivi': {'metriques': ['loss']}}




# payload4={
#     "Parametres_temporels": {
#       "horizon": 1,
#       "dates": ["2025-01-01", "2025-01-01"],
#       "pas_temporel": 60,
#       "portion_decoupage": 0.8
#     },
#     "Parametres_choix_reseau_neurones": {
#       "modele": "MLP"
#     },
#     "Parametres_archi_reseau": {
#       "nb_couches": 2,
#       "hidden_size": 64,
#       "dropout_rate": 0.0,
#       "fonction_activation": "ReLU"
#     },
#     "Parametres_choix_loss_fct": {
#       "fonction_perte": "MSE"
#     },
#     "Parametres_optimisateur": {
#       "optimisateur": "Adam",
#       "learning_rate": 0.001,
#       "decroissance": 0.0,
#       "scheduler": "None",
#       "patience": 5
#     },
#     "Parametres_entrainement": {
#       "nb_epochs": 1000,
#       "batch_size": 4
#     },
#     "Parametres_visualisation_suivi": {
#       "metriques": ["loss"]
#     }
# }

with requests.post(f"{URL}/train_full", json=payload4, stream=True) as r:
    r.raise_for_status()
    print("Content-Type:", r.headers.get("content-type"))  # doit être text/event-stream
    for line in r.iter_lines():
        if not line:
            continue
        if line.startswith(b"data: "):
            msg = json.loads(line[6:].decode("utf-8"))
            # msg = {"epoch": i, "avg_loss": ...} puis {"done": True, "final_loss": ...}
            print("EVENT:", msg)
            if msg.get("done"):
                break