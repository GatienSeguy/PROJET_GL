import requests
import json
import base64
import os

URL = "http://127.0.0.1:8001/models/model_all"
PATH_MODELS_TESTS = "/home/sofsoflefoufou/Documents/Code/projet2/PROJET_GL/SERVEUR_DATA/Models_Tests/"

# Assurons-nous que le dossier de destination existe
os.makedirs(PATH_MODELS_TESTS, exist_ok=True)

payload = {
    "message": "choix_models"
}   

print(f"Envoi de la requête à {URL}...")
response = requests.post(URL, json=payload)
print("Status code :", response.status_code)

# Vérification avant de traiter le JSON
if response.status_code == 200:
    reponse_json = response.json()
    
    if reponse_json is None:
        print("ERREUR: Le serveur a renvoyé 'null'. Vérifiez le return de la fonction serveur.")
    elif not reponse_json:
        print("Le serveur a renvoyé une liste vide (aucun .pth trouvé dans le dossier source).")
    else:
        # Traitement des modèles
        for nom_modele, modele_info in reponse_json.items():
            encoded_data = modele_info.get('model_state_dict')
            
            if encoded_data:
                chemin_sauvegarde = os.path.join(PATH_MODELS_TESTS, nom_modele + ".pth")
                
                # Décodage Base64 et écriture
                try:
                    data_bytes = base64.b64decode(encoded_data)
                    with open(chemin_sauvegarde, "wb") as f:
                        f.write(data_bytes)
                    print(f"✅ Modèle {nom_modele} sauvegardé dans {chemin_sauvegarde}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde de {nom_modele} : {e}")
            else:
                print(f"⚠️ Pas de données pour le modèle {nom_modele}")

else:
    print("Erreur HTTP:", response.status_code)
    print("Réponse brute :", response.text)