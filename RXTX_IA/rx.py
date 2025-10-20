import requests, json

URL = "http://138.231.149.81:8000"

# Séparation du payload global et du payload modèle
payload_global = {
    'Parametres_temporels': 
    {'horizon': 1,
    'dates': ['2024-01-01', '2026-01-01'],
    'pas_temporel': 1, 
    'portion_decoupage': 0.8},

    'Parametres_choix_reseau_neurones': 
    {'modele': 'MLP'},

    'Parametres_choix_loss_fct': 
    {'fonction_perte': 'MSE',
     'params': None},

    'Parametres_optimisateur': 
    {'optimisateur': 'Adam', 
     'learning_rate': 0.01, 
     'decroissance': 0.0, 
     'scheduler': None, 
     'patience': 5},

    'Parametres_entrainement': 
    {'nb_epochs': 1000, 
     'batch_size': 4, 
     'clip_gradient': None},

    'Parametres_visualisation_suivi': 
    {'metriques': ['loss']}
}

payload_model = {
    'nb_couches': 5,
    'hidden_size': 64,
    'bidirectional': False,
    'batch_first': True
}

# Envoi au serveur
with requests.post(f"{URL}/train_full", json={"payload": payload_global, "payload_model": payload_model}, stream=True) as r:
    r.raise_for_status()
    print("Content-Type:", r.headers.get("content-type"))
    for line in r.iter_lines():
        if not line:
            continue
        if line.startswith(b"data: "):
            msg = json.loads(line[6:].decode("utf-8"))
            print("EVENT:", msg)
            if msg.get("done"):
                break