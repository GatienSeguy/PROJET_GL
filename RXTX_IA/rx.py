import requests
import json

# ====================================
# CONFIGURATION DU CLIENT
# ====================================

URL_SERVEUR = "http://localhost:8000/train"


# ====================================
# FONCTION D'ENVOI
# ====================================

def envoyer_configuration(config_dict):
    """
    Envoie une configuration au serveur IA
    et affiche la réponse
    """
    print("\n" + "="*70)
    print("ENVOI DE LA CONFIGURATION AU SERVEUR IA")
    print("="*70)
    print("\nConfiguration envoyée :")
    print(json.dumps(config_dict, indent=2, ensure_ascii=False))
    
    try:
        # Envoi de la requête POST
        response = requests.post(URL_SERVEUR, json=config_dict)
        
        print(f"\nCode de réponse : {response.status_code}")
        
        if response.status_code == 200:
            print("SUCCES - Configuration acceptée !\n")
            print("Réponse du serveur :")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        else:
            print("ERREUR - Configuration rejetée !\n")
            print("Détails de l'erreur :")
            erreurs = response.json()
            if 'detail' in erreurs:
                for erreur in erreurs['detail']:
                    print(f"   Champ : {' -> '.join(map(str, erreur['loc']))}")
                    print(f"   Problème : {erreur['msg']}")
                    print(f"   Type : {erreur['type']}\n")
    
    except requests.exceptions.ConnectionError:
        print("ERREUR : Impossible de se connecter au serveur")
        print("Assure-toi que le serveur tourne sur http://localhost:8000")
    except Exception as e:
        print(f"ERREUR inattendue : {e}")
    
    print("="*70 + "\n")


# ====================================
# TON JSON ICI - MODIFIE CETTE PARTIE
# ====================================

ma_configuration = {
    "action": "start",
    "architecture": {
        "model_type": "lstm",
        "batch_size": 32,
        "num_layers": 3
    },
    "optimisation": {
        "optimizer": "adam",
        "loss_function": "mse",
        "learning_rate": 0.001,
        "epochs": 100
    },
    "horizon": 24,
    "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}


# ====================================
# ENVOI
# ====================================

if __name__ == "__main__":
    print("\nDémarrage du client...")
    envoyer_configuration(ma_configuration)