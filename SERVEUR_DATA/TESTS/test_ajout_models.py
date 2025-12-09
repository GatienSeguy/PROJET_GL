import requests
import base64
import os
from pathlib import Path

# Configuration
URL = "http://127.0.0.1:8001/models/model_add"

# Mettez ici le chemin de votre fichier .pth local à envoyer
FILE_TO_UPLOAD = "/home/sofsoflefoufou/Documents/Code/projet2/PROJET_GL/SERVEUR_DATA/Models_Tests/mon_nouveau_modele.pth"
# Nom que vous voulez donner au modèle sur le serveur (sans l'extension .pth, car le serveur l'ajoute)
MODEL_NAME = "nouveau_modele_test"

def envoyer_modele(file_path: str, model_name: str):
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Erreur : Le fichier {file_path} n'existe pas.")
        return

    print(f"Lecture et encodage de {path.name}...")

    # 1. Lecture du fichier en binaire et encodage en Base64
    try:
        with open(path, "rb") as f:
            file_content = f.read()
            # Encodage en bytes base64 puis décodage en string utf-8 pour le JSON
            encoded_string = base64.b64encode(file_content).decode('utf-8')
    except Exception as e:
        print(f"❌ Erreur lors de la lecture/encodage du fichier : {e}")
        return

    # Vérification de la taille des données avant l'envoi
    original_size_kb = len(file_content) / 1024
    encoded_size_kb = len(encoded_string) / 1024
    print(f"Taille du fichier original : {original_size_kb:.2f} KB")
    print(f"Taille des données Base64 à envoyer : {encoded_size_kb:.2f} KB")
    
    if original_size_kb == 0:
        print("⚠️ Le fichier lu est vide (0 KB). L'envoi ne créera qu'un fichier vide.")

    # 2. Préparation du payload (doit correspondre à newModelRequest côté serveur)
    payload = {
        "name": model_name,
        "data": encoded_string
    }

    # 3. Envoi de la requête
    print(f"\nTentative d'envoi vers {URL} pour créer le modèle: '{model_name}.pth'")
    try:
        response = requests.post(URL, json=payload)
        
        print(f"Status Code : {response.status_code}")
        
        if response.status_code == 200:
            print(f"✅ Succès : Le modèle a été envoyé et devrait être créé sous le nom '{model_name}.pth'.")
            print(f"Réponse serveur : {response.json()}")
        else:
            # Affiche l'erreur renvoyée par le serveur (y compris les 400 ou 500)
            print("❌ Erreur serveur : Le fichier n'a probablement pas été créé.")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au serveur. Vérifiez que le serveur FastAPI est lancé.")

if __name__ == "__main__":
    # Pour tester, créez un fichier bidon si vous n'en avez pas, ou changez le chemin FILE_TO_UPLOAD
    if not os.path.exists(FILE_TO_UPLOAD):
        print(f"⚠️ Le fichier test {FILE_TO_UPLOAD} n'existe pas. Création d'un fichier de test (100 octets)...")
        os.makedirs(os.path.dirname(FILE_TO_UPLOAD), exist_ok=True)
        # Crée un fichier de 100 octets non vide
        with open(FILE_TO_UPLOAD, "wb") as f:
            f.write(b"0" * 100)

    envoyer_modele(FILE_TO_UPLOAD, MODEL_NAME)