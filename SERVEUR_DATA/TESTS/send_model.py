import requests
import os
import sys

# ====================================
# CONFIGURATION
# ====================================
SERVER_URL = "http://:8000"
MODEL_FILE = "mon_modele.pth"  # Pour l'instant le .pth est deja la


def send_model(model_path: str):
    """Envoie un modèle .pth au serveur"""
    
    # Vérifier que le fichier existe
    if not os.path.exists(model_path):
        print(f"Erreur : {model_path} introuvable")
        return False
    
    # Préparer l'envoi
    url = f"{SERVER_URL}/models/upload/"
    
    with open(model_path, 'rb') as f:
        files = {'file': (os.path.basename(model_path), f, 'application/octet-stream')}
        
        try:
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Modèle envoyé avec succès !")
                print(f"Nom sauvegardé : {data['filename']}")
                print(f"Chemin : {data['path']}")
                return True
            else:
                print(f"Erreur {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"Impossible de se connecter au serveur {SERVER_URL}")
            print("   Vérifiez que le serveur est lancé avec : uvicorn main:app --reload")
            return False

if __name__ == "__main__":
    # Permet d'utiliser : python send_model.py mon_autre_modele.pth
    model_file = sys.argv[1] if len(sys.argv) > 1 else MODEL_FILE
    send_model(model_file)