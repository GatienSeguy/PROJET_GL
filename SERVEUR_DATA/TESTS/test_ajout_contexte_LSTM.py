import requests
url = "http://127.0.0.1:8001/contexte/add_solo"
PAQUET = {'payload': {'Parametres_temporels': {'horizon': 1, 'portion_decoupage': 0.8}, 'Parametres_choix_reseau_neurones': {'modele': 'LSTM'}, 'Parametres_choix_loss_fct': {'fonction_perte': 'MSE', 'params': None}, 'Parametres_optimisateur': {'optimisateur': 'Adam', 'learning_rate': 0.001, 'decroissance': 0.0, 'scheduler': 'None', 'patience': 5}, 'Parametres_entrainement': {'nb_epochs': 1000, 'batch_size': 4, 'clip_gradient': None}, 'Parametres_visualisation_suivi': {'metriques': ['loss']}}, 'payload_model': {'Parametres_archi_reseau': {'nb_couches': 2, 'hidden_size': 64, 'bidirectional': False, 'batch_first': False}}, 'payload_dataset': {'name': '', 'dates': ['2001-01-01', '2025-01-02'], 'pas_temporel': 1}}

# --- Envoi au serveur ---
response = requests.post(url, json=PAQUET)
# --- Affichage du résultat ---
print("Status code:", response.status_code)
