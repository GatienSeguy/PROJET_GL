##Structure souhaitée (au début)
├── main.py                 # Serveur FastAPI
├── send_model.py           # Script d'envoi de modèles
├── request_data.py         # Script de requête de données
├── create_model.py         # Script de création de modèle test
├── data/
│   └── timeseries_data.json  # Données temporelles
└── saved_models/           # Dossier des modèles (créé auto)


Sofiane tu trouveras dans le dossier TESTS :
├── main.py                 
├── send_model.py    # Envoie un model déja enregistré en .pth       
├── request_data.py         
├── create_model.py  

Ce qu'il faut faire :
1) Il faut vérifier que tout fonctionne. (partie model et partie data)

2) Essayer de créer un model, l'enregistrer en .pth puis l'envoyer au serveur.

3) Essayer de trouver une solution pour le nom des fichiers enregistrer.

4) Gestionnaire de fichier ?