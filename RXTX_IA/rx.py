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


payload2 = {
    "Parametres_temporels": {
        "horizon": 24,
        "dates": ["2025-01-01", "2025-02-01"],
        "pas_temporel": 1,
        "portion_decoupage": 0.9
    },
    "Parametres_choix_reseau_neurones": {
        "modele": "LSTM"
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


r = requests.post(f"{URL}/train_full", json=payload2)
print("POST /train_full ->", r.status_code)
print(json.dumps(r.json(), indent=2))
