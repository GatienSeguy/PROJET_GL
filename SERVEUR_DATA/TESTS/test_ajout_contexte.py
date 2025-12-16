import requests

url = "http://127.0.0.1:8001/contexte/add_solo"

PAQUET = {
    "payload": {
        "Parametres_temporels": {
            "horizon": 1,
            "portion_decoupage": 0.8
        },
        "Parametres_choix_reseau_neurones": {
            "modele": "CNN"
        },
        "Parametres_choix_loss_fct": {
            "fonction_perte": "MSE",
            "params": None
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
            "batch_size": 4,
            "clip_gradient": None
        },
        "Parametres_visualisation_suivi": {
            "metriques": ["loss"]
        }
    },

    "payload_model": {
        "Parametres_archi_reseau": {
            "nb_couches": 2,
            "hidden_size": 64,
            "dropout_rate": 0.0,
            "fonction_activation": "ReLU",
            "kernel_size": 3,
            "stride": 1,
            "padding": 0
        }
    },

    "payload_dataset": {
        "name": "dataset_test",
        "dates": ["2001-01-01", "2025-01-02"],
        "pas_temporel": 1
    }, 
    "payload_name_model":{"name":"test_cnn"}

}

# --- Envoi au serveur ---
response = requests.post(url, json=PAQUET)

# --- Affichage du résultat ---
print("Status code:", response.status_code)
try:
    print("Réponse JSON:", response.json())
except:
    print("Réponse brute:", response.text)
