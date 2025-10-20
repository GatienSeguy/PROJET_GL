import requests, json

URL = "http://138.231.149.81:8000" 

payload4={'Parametres_temporels': {'horizon': 1,
                                    'dates': ['2024-01-01', '2026-01-01'], 
                                   'pas_temporel': 1, 
                                   'portion_decoupage': 0.8}, 
          'Parametres_choix_reseau_neurones': {'modele': 'CNN'},
            'Parametres_choix_loss_fct': {'fonction_perte': 'MSE', 
                                          'params': None},
          'Parametres_optimisateur': {'optimisateur': 'Adam',
                                         'learning_rate': 0.001,
                                           'decroissance': 0.0,
                                             'scheduler': None,
                                               'patience': 5},
          'Parametres_entrainement': {'nb_epochs': 1000,
                                           'batch_size': 4, 
                                           'clip_grandient': None},
          'Parametres_visualisation_suivi': {'metriques': ['loss']},
          
          'Parametres_archi_reseau_MLP': {'nb_couches': 5, 
                          'hidden_size': 64,
                            'dropout_rate': 0.0,
                              'fonction_activation': 'ReLU'},
          
          'Parametres_archi_reseau_CNN': {'nb_couches': 5, 
                          'hidden_size': 64,
                              'fonction_activation': 'ReLU',
                              'kernel_size':3,
                              'stride':1,
                              'padding':1},
                }




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